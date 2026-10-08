"""
Crowd Risk Prediction System - Main CLI Entry Point.
AI-Based Crowd Panic, Bottleneck and Risk Prediction System for Real-Time Evacuation Safety.
"""

import argparse
import sys
from pathlib import Path

from src.pipeline import CrowdRiskPipeline
from src.utils.config import load_config
from src.utils.logger import setup_logger


def parse_args():
    parser = argparse.ArgumentParser(
        description="AI-Based Crowd Panic, Bottleneck and Risk Prediction System for Real-Time Evacuation Safety."
    )
    parser.add_argument(
        "--source",
        type=str,
        default="data/videos/demo.mp4",
        help="Path to video file or webcam index (e.g., '0', 'data/videos/demo.mp4')."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="Path to configuration YAML file."
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional custom output path for annotated video."
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Maximum number of frames to process (useful for rapid testing)."
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run in Minor Project Demonstration Mode."
    )
    parser.add_argument(
        "--display",
        action="store_true",
        help="Display OpenCV live visualization window during processing."
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["auto", "cuda", "cpu"],
        help="Override computation device (auto, cuda, cpu)."
    )
    return parser.parse_args()


def main():
    args = parse_args()
    logger = setup_logger("Main")

    # Load configuration
    overrides = {}
    if args.device:
        overrides["system"] = {"device": args.device}
    if args.display:
        overrides["video"] = {"display_window": True}
    if args.demo:
        # Lightweight / snappy settings for live demonstration
        overrides["optical_flow"] = {"fast_downscale": 2}

    config = load_config(args.config, overrides=overrides)

    logger.info("=" * 60)
    logger.info("AI CROWD RISK & EVACUATION SAFETY PREDICTION SYSTEM")
    logger.info("Minor Research Project Prototype")
    logger.info("=" * 60)
    logger.info(f"Source: {args.source}")
    logger.info(f"Config: {args.config}")
    if args.demo:
        logger.info("Mode: DEMO MODE ENABLED")

    # Verify source video existence if not a webcam
    if not str(args.source).isdigit() and not Path(args.source).exists():
        logger.error(f"Error: Specified source file does not exist: {args.source}")
        logger.info("Hint: Run 'python scripts/download_demo.py' or provide a valid video.")
        sys.exit(1)

    # Initialize and execute pipeline
    pipeline = CrowdRiskPipeline(config)
    summary = pipeline.process_video(
        source=args.source,
        output_video_path=args.output,
        max_frames=args.max_frames,
        display=args.display
    )

    # Print Final Summary Banner
    print("\n" + "=" * 50)
    print("DEMO COMPLETE" if args.demo else "PROCESSING COMPLETE")
    print("=" * 50)
    print(f"Processed Frames:    {summary['processed_frames']}")
    print(f"Duration:            {summary['duration_seconds']:.2f}s")
    print(f"Maximum People:      {summary['max_people']}")
    print(f"Average People:      {summary['average_people']:.1f}")
    print(f"Maximum Risk:        {summary['maximum_risk']:.2f}")
    print(f"Maximum Panic:       {summary['maximum_panic']:.2f}")
    print(f"Maximum Bottleneck:  {summary['maximum_bottleneck']:.2f}")
    if summary['warning_triggered']:
        print(f"Warning Triggered:   YES (at {summary['warning_first_timestamp']:.2f}s / Frame {summary['warning_first_frame']})")
    else:
        print("Warning Triggered:   NO")
    print(f"Output Video:        {summary['output_video']}")
    print(f"Feature CSV:         {summary['features_csv']}")
    print(f"Risk Timeline CSV:   {summary['timeline_csv']}")
    print(f"Summary JSON:        {config.get('paths', {}).get('summary_json', 'results/summary.json')}")
    print("=" * 50 + "\n")


if __name__ == "__main__":
    main()
