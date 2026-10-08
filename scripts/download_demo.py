"""
Demo Video Acquisition Script.
Downloads or provides access to the official UMN Unusual Crowd Activity dataset,
or generates a standardized synthetic crowd evacuation video for immediate offline testing.
"""

import argparse
import math
from pathlib import Path
import cv2
import numpy as np
import requests
from tqdm import tqdm


OFFICIAL_UMN_PAGE = "https://www.crcv.ucf.edu/projects/Abnormal_Crowd/"


def generate_synthetic_crowd_demo(output_path: str, num_frames: int = 150, fps: int = 25):
    """
    Generates a realistic synthetic crowd evacuation video for offline demonstration and testing.
    Phase 1 (Frames 0-70): Normal wandering / slow walking motion.
    Phase 2 (Frames 70-150): Sudden panic escape towards bottom-right exit / bottleneck.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    width, height = 640, 480
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(path), fourcc, fps, (width, height))

    np.random.seed(42)
    num_people = 20

    # Initialize agent positions
    positions = np.zeros((num_people, 2), dtype=np.float32)
    velocities = np.zeros((num_people, 2), dtype=np.float32)

    positions[:, 0] = np.random.uniform(100, 500, size=num_people)
    positions[:, 1] = np.random.uniform(100, 350, size=num_people)

    # Wander angles
    angles = np.random.uniform(0, 2 * math.pi, size=num_people)
    wander_speed = 1.5

    print(f"Generating synthetic crowd evacuation video ({num_frames} frames)...")
    for f in range(num_frames):
        img = np.full((height, width, 3), 40, dtype=np.uint8)  # Dark ground

        # Draw corridor / exit choke point
        cv2.rectangle(img, (520, 360), (620, 460), (30, 80, 30), 2)
        cv2.putText(img, "EXIT / CHOKE POINT", (450, 345), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 220, 100), 1)

        is_panic_phase = f >= 65

        for i in range(num_people):
            if not is_panic_phase:
                # Normal wandering
                angles[i] += np.random.uniform(-0.2, 0.2)
                velocities[i, 0] = wander_speed * math.cos(angles[i])
                velocities[i, 1] = wander_speed * math.sin(angles[i])
            else:
                # Evacuation surge towards exit (570, 410) with increasing speed and congestion
                dx = 570 - positions[i, 0]
                dy = 410 - positions[i, 1]
                dist = math.hypot(dx, dy)
                if dist > 15:
                    run_speed = np.random.uniform(4.5, 7.5)
                    velocities[i, 0] = (dx / dist) * run_speed
                    velocities[i, 1] = (dy / dist) * run_speed
                else:
                    velocities[i] = np.random.uniform(-0.5, 0.5, size=2)

            positions[i] += velocities[i]
            # Keep within boundary
            positions[i, 0] = np.clip(positions[i, 0], 30, width - 30)
            positions[i, 1] = np.clip(positions[i, 1], 30, height - 30)

            # Draw simulated person (head + body + limbs)
            px, py = int(positions[i, 0]), int(positions[i, 1])
            body_color = (180, 200, 220) if not is_panic_phase else (120, 140, 255)

            # Draw person outline so YOLO recognizes person morphology
            # Head
            cv2.circle(img, (px, py - 24), 8, (200, 200, 200), -1)
            # Torso
            cv2.rectangle(img, (px - 10, py - 16), (px + 10, py + 12), body_color, -1)
            # Legs
            cv2.line(img, (px - 6, py + 12), (px - 8, py + 30), (100, 100, 100), 4)
            cv2.line(img, (px + 6, py + 12), (px + 8, py + 30), (100, 100, 100), 4)
            # Arms
            cv2.line(img, (px - 10, py - 10), (px - 18, py + 4), (100, 100, 100), 3)
            cv2.line(img, (px + 10, py - 10), (px + 18, py + 4), (100, 100, 100), 3)

        out.write(img)

    out.release()
    print(f"Synthetic demo video successfully created: {output_path}")


def download_file(url: str, output_path: str):
    """Downloads a file with a progress bar."""
    response = requests.get(url, stream=True, timeout=20)
    response.raise_for_status()
    total_size = int(response.headers.get("content-length", 0))

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f, tqdm(
        desc=f"Downloading {Path(output_path).name}",
        total=total_size,
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
                bar.update(len(chunk))


def main():
    parser = argparse.ArgumentParser(description="Acquire Demo Video for Crowd Risk System.")
    parser.add_argument(
        "--output",
        type=str,
        default="data/videos/demo.mp4",
        help="Destination path for demo video."
    )
    parser.add_argument(
        "--generate-synthetic",
        action="store_true",
        help="Generate synthetic evacuation demo video directly."
    )
    args = parser.parse_args()

    print("=" * 60)
    print("DEMO VIDEO ACQUISITION TOOL")
    print(f"Official UMN Research Dataset: {OFFICIAL_UMN_PAGE}")
    print("=" * 60)

    # If requested or default, generate clean synthetic demo for guaranteed offline capability
    generate_synthetic_crowd_demo(args.output, num_frames=160, fps=25)
    print("\nTo use an official UMN video:")
    print(f"1. Visit {OFFICIAL_UMN_PAGE}")
    print(f"2. Place the downloaded video into data/videos/umn_demo.avi")
    print(f"3. Run: python run.py --source data/videos/umn_demo.avi --demo")


if __name__ == "__main__":
    main()
