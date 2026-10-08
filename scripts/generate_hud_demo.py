"""
Stage 9: Decision-Support HUD Demonstration Generator.
Executes the crowd risk prediction pipeline with the integrated Decision-Support HUD
across benchmark video clips (UMN and real crowd concert footage).

Outputs:
1. Annotated surveillance videos with HUD:
   - results/hud_demo_umn_indoor.mp4
   - results/hud_demo_concert.mp4
2. High-resolution representative frame screenshots:
   - results/hud_demo_frames/01_umn_calm_monitoring.png
   - results/hud_demo_frames/02_umn_elevated_panic_crda.png
   - results/hud_demo_frames/03_umn_critical_evacuation_intervention.png
   - results/hud_demo_frames/04_concert_dense_crowd_attribution.png
3. Machine-readable telemetry JSON:
   - results/hud_demo_telemetry.json
"""

import argparse
import json
from pathlib import Path
import sys

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np
from tqdm import tqdm

from src.features.cell_grid import (
    SameSceneCalmReferenceManager,
    SpatialCellFeatureExtractor,
    SpatialCellFeatureVector,
    SpatialGridFeatureMap,
)
from src.pipeline import CrowdRiskPipeline
from src.utils.config import load_config
from src.utils.logger import setup_logger
from src.utils.video_io import VideoReader, VideoWriter
from src.visualization.decision_hud import DecisionSupportHUD, HUDTelemetryFrame


def parse_args():
    parser = argparse.ArgumentParser(description="Stage 9: Decision-Support HUD Demo Generator")
    parser.add_argument(
        "--umn-video",
        type=str,
        default="data/videos/indoor_clip4.mp4",
        help="Path to UMN benchmark video clip."
    )
    parser.add_argument(
        "--concert-video",
        type=str,
        default="data/videos/pexels-timo-volz-5544073 (1080p).mp4",
        help="Path to real dense concert video footage."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="Configuration YAML path."
    )
    parser.add_argument(
        "--max-frames-umn",
        type=int,
        default=250,
        help="Maximum frames to process for UMN benchmark video."
    )
    parser.add_argument(
        "--max-frames-concert",
        type=int,
        default=120,
        help="Maximum frames to process for concert footage."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Output directory for generated demo videos, screenshots, and telemetry JSON."
    )
    parser.add_argument(
        "--target-width",
        type=int,
        default=960,
        help="Target upscale width for low-resolution UMN video to ensure high HUD legibility."
    )
    parser.add_argument(
        "--target-height",
        type=int,
        default=720,
        help="Target upscale height for low-resolution UMN video."
    )
    return parser.parse_args()


def process_video_with_hud(
    pipeline: CrowdRiskPipeline,
    source_path: str,
    output_video_path: str,
    max_frames: int,
    target_canvas_size: tuple,
    screenshots_dir: Path,
    screenshot_prefix: str,
    logger
):
    """Processes a video clip, applies the DecisionSupportHUD, captures key frames, and records telemetry."""
    source_p = Path(source_path)
    if not source_p.exists():
        logger.warning(f"Video file not found: {source_path}. Skipping.")
        return []

    logger.info(f"Processing: {source_path}")
    reader = VideoReader(source=source_path)
    total_frames = min(reader.total_frames, max_frames) if reader.total_frames > 0 else max_frames

    out_w, out_h = target_canvas_size
    writer = VideoWriter(output_path=output_video_path, fps=reader.fps, width=out_w, height=out_h)

    # Initialize HUD instance
    hud = DecisionSupportHUD(
        high_risk_threshold=pipeline.config.get("risk", {}).get("levels", {}).get("moderate_cutoff", 0.60),
        moderate_risk_threshold=pipeline.config.get("risk", {}).get("levels", {}).get("low_cutoff", 0.30)
    )

    telemetry_records = []
    pbar = tqdm(total=total_frames, desc=f"Rendering {source_p.name}", unit="frame")

    saved_normal = False
    saved_elevated = False
    saved_peak = False
    peak_risk = -1.0
    peak_frame_img = None
    peak_telemetry = None

    try:
        for frame_idx, frame in reader:
            if frame_idx >= max_frames:
                break

            timestamp = float(frame_idx / max(1.0, reader.fps))

            # Step 1: Detect persons
            detections = pipeline.detector.detect(frame)

            # Step 2: Track objects
            active_tracks = pipeline.tracker.update(detections)

            # Step 3: Trajectories
            for track in active_tracks:
                pipeline.trajectory_mgr.update(
                    track_id=track.track_id,
                    frame_id=frame_idx,
                    raw_center=track.center,
                    fps=reader.fps
                )
            pipeline.trajectory_mgr.cleanup_old_tracks(
                active_track_ids=[t.track_id for t in active_tracks],
                current_frame=frame_idx
            )

            # Step 4: Motion Features
            motion_feat = pipeline.motion_extractor.extract(
                frame_id=frame_idx,
                timestamp=timestamp,
                active_tracks=active_tracks,
                trajectory_mgr=pipeline.trajectory_mgr,
                frame_width=reader.width,
                frame_height=reader.height
            )

            # Step 5: Optical Flow
            flow_feat = pipeline.flow_extractor.compute(frame) if pipeline.flow_enabled else None

            # Step 6: Spatial Density Grid
            density_grid = pipeline.density_estimator.compute_grid(
                frame_width=reader.width,
                frame_height=reader.height,
                active_tracks=active_tracks,
                trajectory_mgr=pipeline.trajectory_mgr
            )

            # Step 6b: Spatial Localized Cell Grid [D, O, B, K]
            flow_field = getattr(pipeline.flow_extractor, "last_flow", None)
            grid_feature_map = pipeline.cell_feature_extractor.extract_grid(
                frame_idx=frame_idx,
                timestamp=timestamp,
                frame_width=reader.width,
                frame_height=reader.height,
                active_tracks=active_tracks,
                trajectory_mgr=pipeline.trajectory_mgr,
                flow_field=flow_field
            )
            is_calm = frame_idx < pipeline.early_warning.warmup_frames
            pipeline.calm_reference_mgr.update(grid_feature_map, is_calm_window=is_calm)

            # Step 7: Bottleneck
            bottleneck_res = pipeline.bottleneck_detector.evaluate(
                motion_features=motion_feat,
                density_grid=density_grid,
                flow_features=flow_feat
            )

            # Step 8: Panic
            panic_res = pipeline.panic_detector.evaluate(
                motion_features=motion_feat,
                flow_features=flow_feat
            )

            # Step 9: Risk Fusion
            risk_res = pipeline.risk_engine.compute_risk(
                panic_result=panic_res,
                bottleneck_result=bottleneck_res,
                motion_features=motion_feat
            )

            # Step 10: Early Warning Monitor
            warning_res = pipeline.early_warning.update(
                frame_id=frame_idx,
                timestamp=timestamp,
                current_risk=risk_res.overall_risk
            )

            # Step 11: CRDA Counterfactual Driver Attribution
            crda_report = pipeline.crda_engine.explain_grid(
                grid_map=grid_feature_map,
                calm_mgr=pipeline.calm_reference_mgr,
                only_elevated=False
            )

            # Base bounding boxes & trajectory rendering at native resolution
            annotated_base = frame.copy()
            if pipeline.visualizer.draw_trajectories:
                annotated_base = pipeline.visualizer._render_trajectories(
                    annotated_base, active_tracks, pipeline.trajectory_mgr
                )
            if pipeline.visualizer.draw_boxes:
                annotated_base = pipeline.visualizer._render_detections(
                    annotated_base, active_tracks, pipeline.trajectory_mgr
                )

            # Resize canvas to target dimensions for crystal-clear HUD text legibility
            if (out_w, out_h) != (reader.width, reader.height):
                annotated_scaled = cv2.resize(annotated_base, (out_w, out_h), interpolation=cv2.INTER_LINEAR)
                # Remap cell coordinates to target canvas
                scale_x = out_w / float(reader.width)
                scale_y = out_h / float(reader.height)
                scaled_cells = []
                for c in grid_feature_map.cells:
                    c_dict = c.to_dict()
                    scaled_c = SpatialCellFeatureVector(
                        cell_id=c.cell_id, row=c.row, col=c.col,
                        x1=int(round(c.x1 * scale_x)), y1=int(round(c.y1 * scale_y)),
                        x2=int(round(c.x2 * scale_x)), y2=int(round(c.y2 * scale_y)),
                        area_px=float(c.area_px * scale_x * scale_y),
                        D=c.D, O=c.O, B=c.B, K=c.K,
                        person_count=c.person_count
                    )
                    scaled_cells.append(scaled_c)
                matrix = [[None for _ in range(grid_feature_map.cols)] for _ in range(grid_feature_map.rows)]
                for sc in scaled_cells:
                    matrix[sc.row][sc.col] = sc
                render_grid_map = SpatialGridFeatureMap(
                    frame_idx=grid_feature_map.frame_idx,
                    timestamp=grid_feature_map.timestamp,
                    rows=grid_feature_map.rows,
                    cols=grid_feature_map.cols,
                    frame_width=out_w,
                    frame_height=out_h,
                    cells=scaled_cells,
                    cell_matrix=matrix
                )
            else:
                annotated_scaled = annotated_base
                render_grid_map = grid_feature_map

            # Render DecisionSupportHUD
            hud_frame, tel = hud.render(
                frame=annotated_scaled,
                frame_idx=frame_idx,
                timestamp=timestamp,
                risk_assessment=risk_res,
                warning_status=warning_res,
                grid_feature_map=render_grid_map,
                crda_report=crda_report
            )

            writer.write(hud_frame)
            telemetry_records.append(tel.to_dict())

            # Capture representative screenshots
            if not saved_normal and frame_idx >= 15 and risk_res.overall_risk < 0.35:
                cv2.imwrite(str(screenshots_dir / f"{screenshot_prefix}_01_normal_monitoring.png"), hud_frame)
                saved_normal = True

            if not saved_elevated and risk_res.overall_risk >= 0.45:
                cv2.imwrite(str(screenshots_dir / f"{screenshot_prefix}_02_elevated_risk_crda.png"), hud_frame)
                saved_elevated = True

            if risk_res.overall_risk > peak_risk:
                peak_risk = risk_res.overall_risk
                peak_frame_img = hud_frame.copy()
                peak_telemetry = tel

            pbar.update(1)

        # Save peak crisis frame screenshot
        if peak_frame_img is not None:
            cv2.imwrite(str(screenshots_dir / f"{screenshot_prefix}_03_peak_crisis_intervention.png"), peak_frame_img)

    finally:
        reader.release()
        writer.release()
        pbar.close()

    logger.info(f"Video saved: {output_video_path} (Peak Risk: {peak_risk:.2f})")
    return telemetry_records


def main():
    args = parse_args()
    logger = setup_logger("HUDDemo")
    config = load_config(args.config)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    screenshots_dir = out_dir / "hud_demo_frames"
    screenshots_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 80)
    logger.info("STAGE 9: DECISION-SUPPORT HUD DEMONSTRATION")
    logger.info("AI-Based Crowd Risk & Evacuation Safety Prediction System")
    logger.info("=" * 80)

    # 1. Process Official UMN Benchmark Clip (Indoor Clip 4)
    logger.info("\n[1/2] Processing UMN Benchmark Clip (indoor_clip4.mp4)...")
    pipeline_umn = CrowdRiskPipeline(config)
    umn_video_out = str(out_dir / "hud_demo_umn_indoor.mp4")
    umn_telemetry = process_video_with_hud(
        pipeline=pipeline_umn,
        source_path=args.umn_video,
        output_video_path=umn_video_out,
        max_frames=args.max_frames_umn,
        target_canvas_size=(args.target_width, args.target_height),
        screenshots_dir=screenshots_dir,
        screenshot_prefix="umn_indoor",
        logger=logger
    )

    # 2. Process Real Concert Dense Crowd Footage
    logger.info("\n[2/2] Processing Real Dense Crowd Concert Footage...")
    concert_path = Path(args.concert_video)
    concert_telemetry = []
    if concert_path.exists():
        pipeline_concert = CrowdRiskPipeline(config)
        concert_video_out = str(out_dir / "hud_demo_concert.mp4")
        concert_telemetry = process_video_with_hud(
            pipeline=pipeline_concert,
            source_path=str(concert_path),
            output_video_path=concert_video_out,
            max_frames=args.max_frames_concert,
            target_canvas_size=(1280, 720),
            screenshots_dir=screenshots_dir,
            screenshot_prefix="real_concert",
            logger=logger
        )
    else:
        logger.warning(f"Concert video not found at: {concert_path}. Skipping.")

    # 3. Save Consolidated Machine-Readable JSON Telemetry
    telemetry_path = out_dir / "hud_demo_telemetry.json"
    demo_manifest = {
        "metadata": {
            "stage": "Stage 9",
            "title": "Decision-Support HUD Telemetry & Audit Log",
            "date": "October 2026",
            "target_safe_threshold_R_safe": 0.60,
            "moderate_risk_threshold": 0.30,
            "crda_alpha": 1.0,
            "scientific_disclaimer": "All risk scores are model-derived non-probabilistic indices. CRDA attributions are non-causal sensitivity indicators."
        },
        "umn_benchmark_summary": {
            "source_video": args.umn_video,
            "frames_processed": len(umn_telemetry),
            "output_video": umn_video_out,
            "sample_telemetry": umn_telemetry[:10] if umn_telemetry else []
        },
        "concert_footage_summary": {
            "source_video": str(concert_path),
            "frames_processed": len(concert_telemetry),
            "sample_telemetry": concert_telemetry[:10] if concert_telemetry else []
        }
    }

    with open(telemetry_path, "w", encoding="utf-8") as f:
        json.dump(demo_manifest, f, indent=4)
    logger.info(f"Consolidated telemetry written to: {telemetry_path}")

    # Summary
    print("\n" + "=" * 60)
    print("STAGE 9 HUD DEMO GENERATION COMPLETE")
    print("=" * 60)
    print(f"UMN Demo Video:          {umn_video_out}")
    if concert_telemetry:
        print(f"Concert Demo Video:      {out_dir / 'hud_demo_concert.mp4'}")
    print(f"Screenshots Directory:   {screenshots_dir}")
    print(f"Machine Telemetry JSON:  {telemetry_path}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
