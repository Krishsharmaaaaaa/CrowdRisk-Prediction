"""
Download and extract the official UMN Unusual Crowd Activity Dataset.
Source: University of Central Florida CRCV / University of Minnesota
URL: https://www.crcv.ucf.edu/projects/Abnormal_Crowd/Normal_Abnormal_Crowd.zip
"""

import os
import sys
import zipfile
from pathlib import Path
import requests
import urllib3

urllib3.disable_warnings()

UMN_ZIP_URL = "https://www.crcv.ucf.edu/projects/Abnormal_Crowd/Normal_Abnormal_Crowd.zip"


def download_and_extract_umn(
    download_url: str = UMN_ZIP_URL,
    target_zip: str = "data/Normal_Abnormal_Crowd.zip",
    extract_dir: str = "data/umn_raw",
    destination_videos_dir: str = "data/videos"
):
    target_path = Path(target_zip)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    extract_path = Path(extract_dir)
    extract_path.mkdir(parents=True, exist_ok=True)
    dest_path = Path(destination_videos_dir)
    dest_path.mkdir(parents=True, exist_ok=True)

    # 1. Download if not already downloaded
    if not target_path.exists() or target_path.stat().st_size < 1000000:
        print(f"Connecting to official UMN source: {download_url}")
        response = requests.get(download_url, stream=True, verify=False, timeout=30)
        response.raise_for_status()
        total_size = int(response.headers.get("content-length", 0))

        print(f"Downloading UMN Dataset ({total_size / (1024*1024):.1f} MB)...")
        downloaded = 0
        with open(target_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        pct = (downloaded / total_size) * 100
                        print(f"\rProgress: {downloaded / (1024*1024):.1f} MB / {total_size / (1024*1024):.1f} MB ({pct:.1f}%)", end="", flush=True)
        print("\nDownload complete.")
    else:
        print(f"Archive already present: {target_path} ({target_path.stat().st_size / (1024*1024):.1f} MB)")

    # 2. Extract
    print(f"Extracting archive to {extract_path}...")
    with zipfile.ZipFile(target_path, "r") as zip_ref:
        zip_ref.extractall(extract_path)
    print("Extraction complete.")

    # 3. List extracted video files
    video_files = list(extract_path.rglob("*.avi")) + list(extract_path.rglob("*.mp4")) + list(extract_path.rglob("*.mat"))
    print(f"Found {len(video_files)} data/video files in extracted directory.")
    for vf in video_files:
        print(f" - {vf.name} ({vf.stat().st_size / (1024*1024):.2f} MB)")


if __name__ == "__main__":
    download_and_extract_umn()
