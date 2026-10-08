"""
Configuration loader with default fallback merging and CLI overrides.
"""

from pathlib import Path
from typing import Any, Dict
import yaml


def _deep_update(base: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    for key, value in updates.items():
        if isinstance(value, dict) and key in base and isinstance(base[key], dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def get_default_config() -> Dict[str, Any]:
    """Returns sensible defaults in case config file is missing or partial."""
    return {
        "system": {
            "device": "auto",
            "seed": 42,
            "debug": False
        },
        "video": {
            "target_fps": None,
            "max_dimension": 1280,
            "display_window": False
        },
        "detection": {
            "model_name": "yolov8n.pt",
            "person_class_id": 0,
            "confidence_threshold": 0.35,
            "iou_threshold": 0.45
        },
        "tracking": {
            "tracker_type": "bytetrack",
            "track_thresh": 0.45,
            "match_thresh": 0.80,
            "track_buffer": 30,
            "min_box_area": 100
        },
        "trajectory": {
            "window_size": 15,
            "min_track_length": 5,
            "smoothing_factor": 0.3
        },
        "optical_flow": {
            "enabled": True,
            "pyr_scale": 0.5,
            "levels": 3,
            "winsize": 15,
            "iterations": 3,
            "poly_n": 5,
            "poly_sigma": 1.2,
            "fast_downscale": 2,
            "flow_mag_threshold": 2.0
        },
        "density": {
            "grid_rows": 4,
            "grid_cols": 4,
            "saturation_capacity": 15,
            "reference_density_per_mpx": 200.0
        },
        "bottleneck": {
            "weights": {
                "density": 0.30,
                "speed_reduction": 0.30,
                "flow_imbalance": 0.20,
                "irregularity": 0.20
            },
            "speed_drop_threshold": 0.40,
            "thresholds": {
                "low": 0.40,
                "critical": 0.70
            }
        },
        "panic": {
            "mode": "heuristic",
            "model_path": "models/panic_rf.joblib",
            "heuristic_weights": {
                "speed_surge": 0.25,
                "speed_variance": 0.20,
                "direction_entropy": 0.20,
                "flow_mean": 0.20,
                "flow_variance": 0.15
            },
            "thresholds": {
                "low": 0.35,
                "high": 0.65
            }
        },
        "risk": {
            "weights": {
                "panic": 0.45,
                "bottleneck": 0.40,
                "density": 0.15
            },
            "levels": {
                "low_cutoff": 0.30,
                "moderate_cutoff": 0.60,
                "high_cutoff": 0.80
            }
        },
        "early_warning": {
            "warmup_frames": 20,
            "sustained_window": 10,
            "risk_threshold": 0.60,
            "trend_window": 15,
            "lead_time_fps": 30
        },
        "visualization": {
            "draw_boxes": True,
            "draw_trajectories": True,
            "draw_density_grid": True,
            "draw_bottleneck_zones": True,
            "draw_dashboard": True,
            "trail_length": 20,
            "dashboard_alpha": 0.75
        },
        "paths": {
            "output_video": "results/annotated_output.mp4",
            "features_csv": "results/features.csv",
            "timeline_csv": "results/risk_timeline.csv",
            "summary_json": "results/summary.json"
        }
    }


def load_config(config_path: str = "config/config.yaml", overrides: Dict[str, Any] = None) -> Dict[str, Any]:
    """Loads configuration from YAML and merges with defaults and overrides."""
    config = get_default_config()
    path = Path(config_path)

    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            yaml_content = yaml.safe_load(f)
            if isinstance(yaml_content, dict):
                _deep_update(config, yaml_content)

    if overrides:
        _deep_update(config, overrides)

    return config
