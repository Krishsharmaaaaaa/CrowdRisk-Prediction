"""
Crowd Risk Prediction System Master Pipeline.
Orchestrates detection, tracking, feature extraction, threat inference, warning, and rendering.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from .bottleneck.bottleneck_detector import BottleneckDetector
from .detection.yolo_detector import YOLOPersonDetector
from .features.density import SpatialDensityEstimator
from .features.cell_grid import SpatialCellFeatureExtractor, SameSceneCalmReferenceManager
from .features.motion import MotionFeatureExtractor
from .features.optical_flow import OpticalFlowExtractor
from .explainability.crda_engine import CRDAEngine
from .panic.panic_detector import PanicDetector
from .risk.risk_fusion import RiskFusionEngine
from .tracking.bytetrack_tracker import ByteTrackTracker
from .tracking.trajectory import TrajectoryManager
from .utils.logger import setup_logger
from .utils.video_io import VideoReader, VideoWriter
from .visualization.visualizer import CrowdVisualizer
from .warning.early_warning import EarlyWarningSystem


class CrowdRiskPipeline:
    """
    End-to-end processing pipeline for video analysis and evacuation decision-support.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = setup_logger("CrowdRiskPipeline")

        # Initialize Subsystems
        self.logger.info("Initializing YOLO Person Detector...")
        det_cfg = config.get("detection", {})
        self.detector = YOLOPersonDetector(
            model_name=det_cfg.get("model_name", "yolov8n.pt"),
            confidence_threshold=det_cfg.get("confidence_threshold", 0.35),
            iou_threshold=det_cfg.get("iou_threshold", 0.45),
            device=config.get("system", {}).get("device", "auto"),
            person_class_id=det_cfg.get("person_class_id", 0)
        )

        self.logger.info("Initializing ByteTrack Tracker...")
        track_cfg = config.get("tracking", {})
        self.tracker = ByteTrackTracker(
            track_thresh=track_cfg.get("track_thresh", 0.45),
            match_thresh=track_cfg.get("match_thresh", 0.80),
            track_buffer=track_cfg.get("track_buffer", 30),
            min_box_area=track_cfg.get("min_box_area", 100.0)
        )

        traj_cfg = config.get("trajectory", {})
        self.trajectory_mgr = TrajectoryManager(
            window_size=traj_cfg.get("window_size", 15),
            min_track_length=traj_cfg.get("min_track_length", 5),
            smoothing_factor=traj_cfg.get("smoothing_factor", 0.3)
        )

        dens_cfg = config.get("density", {})
        self.motion_extractor = MotionFeatureExtractor(
            reference_density_per_mpx=dens_cfg.get("reference_density_per_mpx", 200.0)
        )

        flow_cfg = config.get("optical_flow", {})
        self.flow_enabled = flow_cfg.get("enabled", True)
        self.flow_extractor = OpticalFlowExtractor(
            pyr_scale=flow_cfg.get("pyr_scale", 0.5),
            levels=flow_cfg.get("levels", 3),
            winsize=flow_cfg.get("winsize", 15),
            iterations=flow_cfg.get("iterations", 3),
            poly_n=flow_cfg.get("poly_n", 5),
            poly_sigma=flow_cfg.get("poly_sigma", 1.2),
            fast_downscale=flow_cfg.get("fast_downscale", 2),
            flow_mag_threshold=flow_cfg.get("flow_mag_threshold", 2.0)
        )

        self.density_estimator = SpatialDensityEstimator(
            grid_rows=dens_cfg.get("grid_rows", 4),
            grid_cols=dens_cfg.get("grid_cols", 4),
            saturation_capacity=dens_cfg.get("saturation_capacity", 15)
        )

        grid_cfg = config.get("spatial_grid", {})
        self.cell_feature_extractor = SpatialCellFeatureExtractor(
            grid_rows=grid_cfg.get("grid_rows", dens_cfg.get("grid_rows", 4)),
            grid_cols=grid_cfg.get("grid_cols", dens_cfg.get("grid_cols", 4)),
            reference_density_per_mpx=dens_cfg.get("reference_density_per_mpx", 200.0),
            free_flow_speed_px=grid_cfg.get("free_flow_speed_px", 4.0)
        )
        self.calm_reference_mgr = SameSceneCalmReferenceManager(
            rows=grid_cfg.get("grid_rows", dens_cfg.get("grid_rows", 4)),
            cols=grid_cfg.get("grid_cols", dens_cfg.get("grid_cols", 4)),
            warmup_frames=grid_cfg.get("calm_reference_window", 30)
        )

        bottle_cfg = config.get("bottleneck", {})
        self.bottleneck_detector = BottleneckDetector(
            weights=bottle_cfg.get("weights"),
            speed_drop_threshold=bottle_cfg.get("speed_drop_threshold", 0.40),
            low_threshold=bottle_cfg.get("thresholds", {}).get("low", 0.40),
            critical_threshold=bottle_cfg.get("thresholds", {}).get("critical", 0.70)
        )

        panic_cfg = config.get("panic", {})
        self.panic_detector = PanicDetector(
            mode=panic_cfg.get("mode", "heuristic"),
            model_path=panic_cfg.get("model_path", "models/panic_rf.joblib"),
            heuristic_weights=panic_cfg.get("heuristic_weights"),
            low_threshold=panic_cfg.get("thresholds", {}).get("low", 0.35),
            high_threshold=panic_cfg.get("thresholds", {}).get("high", 0.65)
        )

        risk_cfg = config.get("risk", {})
        self.risk_engine = RiskFusionEngine(
            weights=risk_cfg.get("weights"),
            low_cutoff=risk_cfg.get("levels", {}).get("low_cutoff", 0.30),
            moderate_cutoff=risk_cfg.get("levels", {}).get("moderate_cutoff", 0.60),
            high_cutoff=risk_cfg.get("levels", {}).get("high_cutoff", 0.80)
        )

        self.crda_engine = CRDAEngine(
            high_risk_threshold=risk_cfg.get("levels", {}).get("moderate_cutoff", 0.60),
            moderate_risk_threshold=risk_cfg.get("levels", {}).get("low_cutoff", 0.30)
        )

        warn_cfg = config.get("early_warning", {}) or config.get("warning", {})
        self.early_warning = EarlyWarningSystem(
            warmup_frames=warn_cfg.get("warmup_frames", 20),
            risk_threshold=warn_cfg.get("risk_threshold", 0.60),
            sustained_window=warn_cfg.get("sustained_window", 10),
            trend_window=warn_cfg.get("trend_window", 15),
            lead_time_fps=warn_cfg.get("lead_time_fps", 30.0)
        )

        viz_cfg = config.get("visualization", {})
        self.visualizer = CrowdVisualizer(
            draw_boxes=viz_cfg.get("draw_boxes", True),
            draw_trajectories=viz_cfg.get("draw_trajectories", True),
            draw_density_grid=viz_cfg.get("draw_density_grid", True),
            draw_bottleneck_zones=viz_cfg.get("draw_bottleneck_zones", True),
            draw_dashboard=viz_cfg.get("draw_dashboard", True),
            trail_length=viz_cfg.get("trail_length", 20),
            dashboard_alpha=viz_cfg.get("dashboard_alpha", 0.75)
        )

    def process_video(
        self,
        source: str,
        output_video_path: Optional[str] = None,
        max_frames: Optional[int] = None,
        display: bool = False
    ) -> Dict[str, Any]:
        """
        Executes end-to-end processing loop on input video or stream.
        """
        # Reset internal states
        self.tracker.reset()
        self.trajectory_mgr = TrajectoryManager(
            window_size=self.config.get("trajectory", {}).get("window_size", 15),
            min_track_length=self.config.get("trajectory", {}).get("min_track_length", 5),
            smoothing_factor=self.config.get("trajectory", {}).get("smoothing_factor", 0.3)
        )
        dens_cfg = self.config.get("density", {})
        self.motion_extractor = MotionFeatureExtractor(
            reference_density_per_mpx=dens_cfg.get("reference_density_per_mpx", 200.0)
        )
        self.flow_extractor.reset()
        self.bottleneck_detector.reset()
        self.panic_detector.reset()
        self.early_warning.reset()

        paths_cfg = self.config.get("paths", {})
        out_vid_path = output_video_path or paths_cfg.get("output_video", "results/annotated_output.mp4")
        features_csv_path = paths_cfg.get("features_csv", "results/features.csv")
        timeline_csv_path = paths_cfg.get("timeline_csv", "results/risk_timeline.csv")
        summary_json_path = paths_cfg.get("summary_json", "results/summary.json")

        self.logger.info(f"Opening video source: {source}")
        reader = VideoReader(
            source=source,
            max_dimension=self.config.get("video", {}).get("max_dimension", 1280)
        )

        writer = VideoWriter(
            output_path=out_vid_path,
            fps=reader.fps,
            width=reader.width,
            height=reader.height
        )

        total_frames = reader.total_frames if reader.total_frames > 0 else None
        if max_frames and total_frames:
            total_frames = min(total_frames, max_frames)

        pbar = tqdm(total=total_frames, desc="Processing Video", unit="frame")

        feature_records: List[Dict[str, Any]] = []
        timeline_records: List[Dict[str, Any]] = []
        cell_grid_samples: List[Dict[str, Any]] = []
        crda_explanations: List[Dict[str, Any]] = []

        people_counts = []
        risk_scores = []
        panic_scores = []
        bottleneck_scores = []

        warning_triggered = False
        warning_first_time: Optional[float] = None
        warning_first_frame: Optional[int] = None

        try:
            for frame_idx, frame in reader:
                if max_frames is not None and frame_idx >= max_frames:
                    break

                timestamp = float(frame_idx / max(1.0, reader.fps))

                # Step 1: Detect Persons
                detections = self.detector.detect(frame)

                # Step 2: Track Multi-Object
                active_tracks = self.tracker.update(detections)

                # Step 3: Trajectory Updates
                for track in active_tracks:
                    self.trajectory_mgr.update(
                        track_id=track.track_id,
                        frame_id=frame_idx,
                        raw_center=track.center,
                        fps=reader.fps
                    )
                self.trajectory_mgr.cleanup_old_tracks(
                    active_track_ids=[t.track_id for t in active_tracks],
                    current_frame=frame_idx
                )

                # Step 4: Motion Features
                motion_feat = self.motion_extractor.extract(
                    frame_id=frame_idx,
                    timestamp=timestamp,
                    active_tracks=active_tracks,
                    trajectory_mgr=self.trajectory_mgr,
                    frame_width=reader.width,
                    frame_height=reader.height
                )

                # Step 5: Optical Flow
                if self.flow_enabled:
                    flow_feat = self.flow_extractor.compute(frame)
                else:
                    flow_feat = None

                # Step 6: Spatial Density Grid
                density_grid = self.density_estimator.compute_grid(
                    frame_width=reader.width,
                    frame_height=reader.height,
                    active_tracks=active_tracks,
                    trajectory_mgr=self.trajectory_mgr
                )

                # Step 6b (Research Layer): Spatial Localized Cell Grid [D, O, B, K]
                flow_field = getattr(self.flow_extractor, "last_flow", None)
                grid_feature_map = self.cell_feature_extractor.extract_grid(
                    frame_idx=frame_idx,
                    timestamp=timestamp,
                    frame_width=reader.width,
                    frame_height=reader.height,
                    active_tracks=active_tracks,
                    trajectory_mgr=self.trajectory_mgr,
                    flow_field=flow_field
                )
                is_calm = frame_idx < self.early_warning.warmup_frames
                self.calm_reference_mgr.update(grid_feature_map, is_calm_window=is_calm)

                # Step 7: Bottleneck Detection
                bottleneck_res = self.bottleneck_detector.evaluate(
                    motion_features=motion_feat,
                    density_grid=density_grid,
                    flow_features=flow_feat
                )

                # Step 8: Panic / Anomaly Detection
                panic_res = self.panic_detector.evaluate(
                    motion_features=motion_feat,
                    flow_features=flow_feat
                )

                # Step 9: Risk Fusion
                risk_res = self.risk_engine.compute_risk(
                    panic_result=panic_res,
                    bottleneck_result=bottleneck_res,
                    motion_features=motion_feat
                )

                # Step 10: Early Warning Temporal Monitor
                warning_res = self.early_warning.update(
                    frame_id=frame_idx,
                    timestamp=timestamp,
                    current_risk=risk_res.overall_risk
                )

                if warning_res.is_warning_active and not warning_triggered:
                    warning_triggered = True
                    warning_first_time = timestamp
                    warning_first_frame = frame_idx
                    self.logger.warning(
                        f"==> EARLY WARNING TRIGGERED at Frame {frame_idx} (Time {timestamp:.2f}s) | Risk: {risk_res.overall_risk:.2f}"
                    )

                # Record Telemetry
                people_counts.append(motion_feat.person_count)
                risk_scores.append(risk_res.overall_risk)
                panic_scores.append(panic_res.score)
                bottleneck_scores.append(bottleneck_res.score)

                # CSV Record: Motion & Flow Features
                f_dict = motion_feat.to_dict()
                if flow_feat:
                    f_dict["optical_flow_mean"] = flow_feat.mean_magnitude
                    f_dict["optical_flow_variance"] = flow_feat.magnitude_variance
                    f_dict["high_flow_ratio"] = flow_feat.high_flow_ratio
                feature_records.append(f_dict)

                # CSV Record: Risk & Warning Timeline
                timeline_records.append({
                    "frame": frame_idx,
                    "timestamp": timestamp,
                    "person_count": motion_feat.person_count,
                    "panic_score": panic_res.score,
                    "panic_level": panic_res.level,
                    "bottleneck_score": bottleneck_res.score,
                    "bottleneck_level": bottleneck_res.level,
                    "density_score": motion_feat.density,
                    "overall_risk": risk_res.overall_risk,
                    "risk_level": risk_res.risk_level,
                    "early_warning_active": warning_res.is_warning_active,
                    "risk_trend": warning_res.risk_trend
                })

                # Step 11: Compute CRDA Attribution Report & Render Visualization Overlay
                crda_report = self.crda_engine.explain_grid(
                    grid_map=grid_feature_map,
                    calm_mgr=self.calm_reference_mgr,
                    only_elevated=False
                )

                annotated_frame = self.visualizer.render(
                    frame=frame,
                    active_tracks=active_tracks,
                    trajectory_mgr=self.trajectory_mgr,
                    motion_features=motion_feat,
                    density_grid=density_grid,
                    bottleneck_result=bottleneck_res,
                    panic_result=panic_res,
                    risk_assessment=risk_res,
                    warning_status=warning_res,
                    flow_features=flow_feat,
                    grid_feature_map=grid_feature_map,
                    crda_report=crda_report
                )

                if frame_idx % 15 == 0 or warning_res.is_warning_active:
                    cell_grid_samples.append(grid_feature_map.to_dict())
                    if crda_report.elevated_cells_count > 0:
                        crda_explanations.append(crda_report.to_dict())

                writer.write(annotated_frame)

                if display:
                    cv2.imshow("AI Crowd Safety Monitor", annotated_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

                pbar.update(1)

        finally:
            reader.release()
            writer.release()
            pbar.close()
            if display:
                cv2.destroyAllWindows()

        # Save CSV Logs
        Path(features_csv_path).parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(feature_records).to_csv(features_csv_path, index=False)

        Path(timeline_csv_path).parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(timeline_records).to_csv(timeline_csv_path, index=False)

        # Save Cell Grid Feature Samples
        cell_grid_path = Path(paths_cfg.get("cell_grid_json", "results/cell_grid_samples.json"))
        cell_grid_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cell_grid_path, "w", encoding="utf-8") as f:
            json.dump(cell_grid_samples, f, indent=4)

        # Save CRDA Counterfactual Attribution Reports
        crda_output_path = Path(paths_cfg.get("crda_json", "results/crda_explanations.json"))
        crda_output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(crda_output_path, "w", encoding="utf-8") as f:
            json.dump(crda_explanations, f, indent=4)

        # Summary JSON
        processed_frames = len(risk_scores)
        duration_sec = float(processed_frames / max(1.0, reader.fps))

        summary = {
            "video_source": str(source),
            "output_video": str(out_vid_path),
            "features_csv": str(features_csv_path),
            "timeline_csv": str(timeline_csv_path),
            "fps": reader.fps,
            "processed_frames": processed_frames,
            "duration_seconds": duration_sec,
            "resolution": f"{reader.width}x{reader.height}",
            "max_people": int(np.max(people_counts)) if people_counts else 0,
            "average_people": float(np.mean(people_counts)) if people_counts else 0.0,
            "maximum_risk": float(np.max(risk_scores)) if risk_scores else 0.0,
            "average_risk": float(np.mean(risk_scores)) if risk_scores else 0.0,
            "maximum_panic": float(np.max(panic_scores)) if panic_scores else 0.0,
            "maximum_bottleneck": float(np.max(bottleneck_scores)) if bottleneck_scores else 0.0,
            "warning_triggered": warning_triggered,
            "warning_first_frame": warning_first_frame,
            "warning_first_timestamp": warning_first_time
        }

        Path(summary_json_path).parent.mkdir(parents=True, exist_ok=True)
        with open(summary_json_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=4)

        return summary
