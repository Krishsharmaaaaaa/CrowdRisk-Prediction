"""
UMN Dataset Preparation and Ground Truth Segment Definition Script.
Segments the official University of Minnesota (UMN) Unusual Crowd Activity dataset
into 11 standardized benchmark video clips across 3 distinct scenes:
- Scene 1: Lawn (clips 1-2, 1452 frames)
- Scene 2: Indoor Hallway (clips 3-8, 4143 frames)
- Scene 3: Plaza Courtyard (clips 9-11, 2143 frames)
Total: 7,739 frames.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import cv2
from tqdm import tqdm

# Documented benchmark clip cuts and ground-truth temporal anomaly intervals
# in the official 7,739-frame UMN master recording (Crowd-Activity-All.avi)
UMN_CLIPS_SPEC = [
    # Scene 1: Lawn
    {
        "clip_id": "lawn_clip1",
        "scene": "lawn",
        "scene_name": "Lawn (Outdoor Grass)",
        "start_frame_master": 0,
        "end_frame_master": 625,
        "normal_range": [0, 497],
        "abnormal_range": [498, 624]
    },
    {
        "clip_id": "lawn_clip2",
        "scene": "lawn",
        "scene_name": "Lawn (Outdoor Grass)",
        "start_frame_master": 626,
        "end_frame_master": 1453,
        "normal_range": [0, 297],
        "abnormal_range": [298, 826]
    },
    # Scene 2: Indoor Hallway
    {
        "clip_id": "indoor_clip3",
        "scene": "indoor",
        "scene_name": "Indoor (Hallway/Corridor)",
        "start_frame_master": 1454,
        "end_frame_master": 2002,
        "normal_range": [0, 94],
        "abnormal_range": [95, 547]
    },
    {
        "clip_id": "indoor_clip4",
        "scene": "indoor",
        "scene_name": "Indoor (Hallway/Corridor)",
        "start_frame_master": 2003,
        "end_frame_master": 2687,
        "normal_range": [0, 95],
        "abnormal_range": [96, 683]
    },
    {
        "clip_id": "indoor_clip5",
        "scene": "indoor",
        "scene_name": "Indoor (Hallway/Corridor)",
        "start_frame_master": 2688,
        "end_frame_master": 3455,
        "normal_range": [0, 296],
        "abnormal_range": [297, 766]
    },
    {
        "clip_id": "indoor_clip6",
        "scene": "indoor",
        "scene_name": "Indoor (Hallway/Corridor)",
        "start_frame_master": 3456,
        "end_frame_master": 4034,
        "normal_range": [0, 297],
        "abnormal_range": [298, 577]
    },
    {
        "clip_id": "indoor_clip7",
        "scene": "indoor",
        "scene_name": "Indoor (Hallway/Corridor)",
        "start_frame_master": 4035,
        "end_frame_master": 4807,
        "normal_range": [0, 297],
        "abnormal_range": [298, 771]
    },
    {
        "clip_id": "indoor_clip8",
        "scene": "indoor",
        "scene_name": "Indoor (Hallway/Corridor)",
        "start_frame_master": 4808,
        "end_frame_master": 5596,
        "normal_range": [0, 119],
        "abnormal_range": [120, 787]
    },
    # Scene 3: Plaza Courtyard
    {
        "clip_id": "plaza_clip9",
        "scene": "plaza",
        "scene_name": "Plaza (Outdoor Courtyard)",
        "start_frame_master": 5597,
        "end_frame_master": 6195,
        "normal_range": [0, 554],
        "abnormal_range": [555, 597]
    },
    {
        "clip_id": "plaza_clip10",
        "scene": "plaza",
        "scene_name": "Plaza (Outdoor Courtyard)",
        "start_frame_master": 6196,
        "end_frame_master": 6931,
        "normal_range": [0, 56],
        "abnormal_range": [57, 734]
    },
    {
        "clip_id": "plaza_clip11",
        "scene": "plaza",
        "scene_name": "Plaza (Outdoor Courtyard)",
        "start_frame_master": 6932,
        "end_frame_master": 7739,
        "normal_range": [0, 766],
        "abnormal_range": [767, 806]
    }
]


def prepare_umn_dataset(
    master_video: str = "data/videos/Crowd-Activity-All.avi",
    output_dir: str = "data/videos",
    output_meta_path: str = "data/umn_metadata.json"
):
    master_path = Path(master_video)
    if not master_path.exists():
        raise FileNotFoundError(f"Master UMN video not found at: {master_video}. Please run download script first.")

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    meta_path = Path(output_meta_path)
    meta_path.parent.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(master_path))
    total_master_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"Master UMN Video: {master_path.name} ({total_master_frames} frames, {width}x{height} @ {fps:.1f} FPS)")
    print(f"Segmenting into 11 benchmark clips across 3 scenes...")

    records: Dict[str, Dict] = {}
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    for spec in UMN_CLIPS_SPEC:
        clip_id = spec["clip_id"]
        scene = spec["scene"]
        start_f = spec["start_frame_master"]
        end_f = spec["end_frame_master"]
        expected_len = end_f - start_f
        clip_filename = f"{clip_id}.mp4"
        clip_path = out_dir / clip_filename

        print(f"\nExtracting {clip_id} [Scene: {scene}, Frames {start_f}..{end_f} ({expected_len} frames)]...")
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)
        writer = cv2.VideoWriter(str(clip_path), fourcc, fps, (width, height))

        actual_frames = 0
        for _ in range(expected_len):
            ret, frame = cap.read()
            if not ret:
                break
            writer.write(frame)
            actual_frames += 1

        writer.release()
        print(f"Wrote {actual_frames} frames -> {clip_path}")

        records[clip_filename] = {
            "clip_id": clip_id,
            "scene": scene,
            "scene_name": spec["scene_name"],
            "file_path": str(clip_path.resolve()),
            "total_frames": actual_frames,
            "fps": fps,
            "normal_range": spec["normal_range"],
            "abnormal_range": [spec["abnormal_range"][0], min(spec["abnormal_range"][1], actual_frames - 1)]
        }

    cap.release()

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=4)

    print("\n" + "=" * 60)
    print(f"Successfully prepared UMN Dataset Metadata: {meta_path}")
    print(f"Registered {len(records)} benchmark clips across 3 scenes (Lawn, Indoor, Plaza).")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Prepare and segment official UMN benchmark dataset.")
    parser.add_argument("--master-video", type=str, default="data/videos/Crowd-Activity-All.avi", help="Path to Crowd-Activity-All.avi")
    parser.add_argument("--output-dir", type=str, default="data/videos", help="Output directory for segmented clips.")
    parser.add_argument("--output-meta", type=str, default="data/umn_metadata.json", help="Path for metadata JSON.")
    args = parser.parse_args()

    prepare_umn_dataset(args.master_video, args.output_dir, args.output_meta)


if __name__ == "__main__":
    main()
