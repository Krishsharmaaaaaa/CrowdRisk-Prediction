"""
Dataset Feature Extraction Script.
Processes video files through detector, tracker, optical flow, and motion extractors
to compile tabular ground-truth-labeled training data.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List
import pandas as pd
from tqdm import tqdm

# Add root directory to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.detection.yolo_detector import YOLOPersonDetector
from src.features.motion import MotionFeatureExtractor
from src.features.optical_flow import OpticalFlowExtractor
from src.tracking.bytetrack_tracker import ByteTrackTracker
from src.tracking.trajectory import TrajectoryManager
from src.utils.config import load_config
from src.utils.video_io import VideoReader


def extract_features_from_dataset(metadata_file: str, output_csv: str, config_path: str = "config/config.yaml"):
    config = load_config(config_path)
    meta_path = Path(metadata_file)
    if not meta_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_file}")

    with open(meta_path, "r", encoding="utf-8") as f:
        metadata: Dict[str, Dict] = json.load(f)

    detector = YOLOPersonDetector(
        model_name=config["detection"]["model_name"],
        confidence_threshold=config["detection"]["confidence_threshold"],
        device=config["system"]["device"]
    )
    tracker = ByteTrackTracker(track_thresh=config["tracking"]["track_thresh"])
    trajectory_mgr = TrajectoryManager()
    dens_cfg = config.get("density", {})
    motion_extractor = MotionFeatureExtractor(
        reference_density_per_mpx=dens_cfg.get("reference_density_per_mpx", 200.0)
    )
    flow_extractor = OpticalFlowExtractor()

    all_rows: List[Dict] = []

    for vid_name, meta in metadata.items():
        vid_path = meta["file_path"]
        if not Path(vid_path).exists():
            print(f"Skipping missing video: {vid_path}")
            continue

        normal_start, normal_end = meta.get("normal_range", [0, 0])
        abnormal_start, abnormal_end = meta.get("abnormal_range", [0, 0])

        print(f"\nProcessing {vid_name}...")
        tracker.reset()
        flow_extractor.reset()
        trajectory_mgr = TrajectoryManager()
        motion_extractor = MotionFeatureExtractor(
            reference_density_per_mpx=dens_cfg.get("reference_density_per_mpx", 200.0)
        )

        reader = VideoReader(vid_path, max_dimension=config["video"]["max_dimension"])
        for frame_idx, frame in tqdm(reader, desc=f"Extracting {vid_name}", total=reader.total_frames):
            timestamp = frame_idx / max(1.0, reader.fps)

            # Ground truth label: 0 for normal, 1 for abnormal/panic
            if abnormal_start <= frame_idx <= abnormal_end:
                label = 1
            else:
                label = 0

            # Pipeline processing
            dets = detector.detect(frame)
            tracks = tracker.update(dets)

            for t in tracks:
                trajectory_mgr.update(t.track_id, frame_idx, t.center, fps=reader.fps)
            trajectory_mgr.cleanup_old_tracks([t.track_id for t in tracks], current_frame=frame_idx)

            mf = motion_extractor.extract(
                frame_idx,
                timestamp,
                tracks,
                trajectory_mgr,
                frame_width=reader.width,
                frame_height=reader.height
            )
            flow = flow_extractor.compute(frame)

            row = mf.to_dict()
            row["flow_mean"] = flow.mean_magnitude
            row["flow_variance"] = flow.magnitude_variance
            row["high_flow_ratio"] = flow.high_flow_ratio
            row["video_name"] = vid_name
            row["video_id"] = vid_name
            row["clip_id"] = meta.get("clip_id", vid_name)
            row["scene"] = meta.get("scene", "unknown")
            row["scene_name"] = meta.get("scene_name", "unknown")
            row["label"] = label
            all_rows.append(row)

        reader.release()

    df = pd.DataFrame(all_rows)
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"\nSuccessfully compiled feature dataset with {len(df)} frames -> {output_csv}")


def main():
    parser = argparse.ArgumentParser(description="Extract Dataset Feature Vectors for Training.")
    parser.add_argument("--metadata", type=str, default="data/umn_metadata.json", help="Path to metadata JSON.")
    parser.add_argument("--output", type=str, default="data/processed/features_dataset.csv", help="Output CSV path.")
    parser.add_argument("--config", type=str, default="config/config.yaml", help="Path to config file.")
    args = parser.parse_args()

    extract_features_from_dataset(args.metadata, args.output, args.config)


if __name__ == "__main__":
    main()
