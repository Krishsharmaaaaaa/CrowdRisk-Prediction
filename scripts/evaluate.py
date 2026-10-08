"""
Evacuation Safety and Early Warning Evaluation Script.
Evaluates timeline predictions against ground-truth temporal segments and computes detection lead time.
"""

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score


def evaluate_timeline(
    timeline_csv: str = "results/risk_timeline.csv",
    ground_truth_json: str = "data/umn_metadata.json",
    video_name: str = "demo.mp4",
    output_json: str = "results/evaluation_report.json"
):
    csv_path = Path(timeline_csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"Timeline CSV not found: {timeline_csv}")

    df = pd.read_csv(csv_path)
    total_frames = len(df)
    print(f"Loaded {total_frames} frames from {csv_path}")

    # Load ground truth if provided
    gt_onset = None
    y_true = np.zeros(total_frames, dtype=int)

    meta_path = Path(ground_truth_json)
    if meta_path.exists():
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        if video_name in meta:
            ab_range = meta[video_name].get("abnormal_range", [0, 0])
            gt_onset = ab_range[0]
            for f in range(total_frames):
                if ab_range[0] <= f <= ab_range[1]:
                    y_true[f] = 1

    # Predicted binary label based on Risk Level (HIGH or CRITICAL) or Early Warning Trigger
    y_pred = (df["overall_risk"] >= 0.50).astype(int).values
    y_scores = df["overall_risk"].values

    acc = float(accuracy_score(y_true, y_pred)) if np.any(y_true) else None
    prec = float(precision_score(y_true, y_pred, zero_division=0)) if np.any(y_true) else None
    rec = float(recall_score(y_true, y_pred, zero_division=0)) if np.any(y_true) else None
    f1 = float(f1_score(y_true, y_pred, zero_division=0)) if np.any(y_true) else None
    try:
        roc_auc = float(roc_auc_score(y_true, y_scores)) if np.any(y_true) and len(np.unique(y_true)) > 1 else None
    except Exception:
        roc_auc = None

    # Determine first early warning trigger
    warning_frames = df[df["early_warning_active"] == True]["frame"].tolist()
    first_warn_frame = warning_frames[0] if warning_frames else None
    first_warn_sec = float(df[df["early_warning_active"] == True]["timestamp"].iloc[0]) if warning_frames else None

    lead_time_seconds = None
    lead_time_frames = None
    if gt_onset is not None and first_warn_frame is not None:
        lead_time_frames = int(gt_onset - first_warn_frame)
        fps = total_frames / max(1.0, float(df["timestamp"].iloc[-1]))
        lead_time_seconds = float(lead_time_frames / fps)

    report = {
        "video_evaluated": video_name,
        "total_frames": total_frames,
        "ground_truth_event_onset_frame": gt_onset,
        "first_warning_frame": first_warn_frame,
        "first_warning_timestamp_sec": first_warn_sec,
        "lead_time_frames": lead_time_frames,
        "lead_time_seconds": lead_time_seconds,
        "metrics": {
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1_score": f1,
            "roc_auc": roc_auc
        }
    }

    print("\n" + "=" * 50)
    print("EVALUATION & EARLY WARNING REPORT")
    print("=" * 50)
    print(f"Video Evaluated:           {video_name}")
    print(f"Ground Truth Onset Frame:  {gt_onset}")
    print(f"First Early Warning Frame: {first_warn_frame}")
    if lead_time_seconds is not None:
        print(f"Early Warning Lead Time:   {lead_time_seconds:.2f} seconds ({lead_time_frames} frames ahead)")
    else:
        print("Early Warning Lead Time:   N/A (or rising-risk demonstrated)")
    if acc is not None:
        print(f"Accuracy:                  {acc:.4f}")
        print(f"Precision:                 {prec:.4f}")
        print(f"Recall:                    {rec:.4f}")
        print(f"F1 Score:                  {f1:.4f}")
    if roc_auc is not None:
        print(f"ROC-AUC:                   {roc_auc:.4f}")
    print("=" * 50)

    Path(output_json).parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4)
    print(f"Evaluation report saved -> {output_json}\n")


def main():
    parser = argparse.ArgumentParser(description="Evaluate Timeline Risk against Ground Truth.")
    parser.add_argument("--timeline", type=str, default="results/risk_timeline.csv", help="Timeline CSV path.")
    parser.add_argument("--metadata", type=str, default="data/umn_metadata.json", help="Ground truth metadata JSON.")
    parser.add_argument("--video-name", type=str, default="demo.mp4", help="Name of video in metadata.")
    parser.add_argument("--output", type=str, default="results/evaluation_report.json", help="Report output path.")
    args = parser.parse_args()

    evaluate_timeline(args.timeline, args.metadata, args.video_name, args.output)


if __name__ == "__main__":
    main()
