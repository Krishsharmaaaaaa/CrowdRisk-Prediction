import os
import psutil
import time
import gc
import cv2
import numpy as np

def get_rss_mb():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)

print(f"Initial Process RSS: {get_rss_mb():.1f} MB")

import torch
print(f"After importing torch: {get_rss_mb():.1f} MB")

from src.utils.config import load_config
from src.pipeline import CrowdRiskPipeline

config = load_config('config/config.yaml', overrides={
    'video': {'display_window': False},
    'system': {'device': 'cpu'}
})
print(f"Before creating pipeline: {get_rss_mb():.1f} MB")

pipeline = CrowdRiskPipeline(config)
print(f"After creating pipeline (YOLO + ByteTrack + RF): {get_rss_mb():.1f} MB")

source = 'data/videos/demo.mp4'
out_vid = 'results/test_profile_out.mp4'

# Profile frame by frame
reader = pipeline.process_video(source, output_video_path=out_vid, max_frames=60, display=False)
print(f"After processing 60 frames: {get_rss_mb():.1f} MB")

gc.collect()
print(f"After gc.collect(): {get_rss_mb():.1f} MB")
