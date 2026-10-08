"""
Analysis Job Manager for Asynchronous Video Risk Processing.
Provides thread-safe job tracking, background worker execution, isolated storage,
input sanitization, and result retrieval for the Render production backend.
"""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
import shutil
from typing import Any, Dict, List, Optional
import uuid

import cv2
import pandas as pd

from ..pipeline import CrowdRiskPipeline
from ..utils.config import load_config
from ..utils.logger import setup_logger

logger = setup_logger("JobManager")

MAX_UPLOAD_SIZE_BYTES = int(os.getenv("MAX_UPLOAD_SIZE_MB", "150")) * 1024 * 1024
ALLOWED_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}


@dataclass
class JobRecord:
    analysis_id: str
    status: str                         # "queued", "processing", "completed", "failed"
    original_filename: str
    created_at: str
    completed_at: Optional[str] = None
    progress: int = 0
    current_stage: str = "Queued"
    error_message: Optional[str] = None
    summary: Optional[Dict[str, Any]] = None
    is_demo: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class AnalysisJobManager:
    """
    Manages asynchronous analysis jobs, isolated execution directories,
    and access to processed video, timelines, spatial grids, and CRDA reports.
    """

    def __init__(self, base_jobs_dir: Optional[str] = None, max_workers: Optional[int] = None):
        self.jobs_dir = Path(base_jobs_dir or os.getenv("JOBS_DIR", "data/jobs")).resolve()
        self.jobs_dir.mkdir(parents=True, exist_ok=True)
        workers = max_workers or int(os.getenv("MAX_CONCURRENT_JOBS", "1"))
        self.executor = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="risk_worker")
        self._jobs: Dict[str, JobRecord] = {}
        self._load_existing_jobs()

    def _load_existing_jobs(self):
        """Discovers existing jobs from disk on service restart."""
        for job_folder in self.jobs_dir.iterdir():
            if job_folder.is_dir():
                meta_file = job_folder / "job_meta.json"
                if meta_file.exists():
                    try:
                        with open(meta_file, "r", encoding="utf-8") as f:
                            data = json.load(f)
                            self._jobs[data["analysis_id"]] = JobRecord(**data)
                    except Exception as e:
                        logger.warning(f"Could not load metadata for job {job_folder.name}: {e}")

    async def create_job_from_upload(self, file) -> JobRecord:
        """
        Streams uploaded video directly to disk in 64KB chunks, validating size
        and codec without holding the entire binary file in memory.
        """
        clean_name = Path(file.filename).name
        ext = Path(clean_name).suffix.lower()
        if ext not in ALLOWED_EXTENSIONS:
            raise ValueError(f"Unsupported file format '{ext}'. Allowed extensions: {', '.join(sorted(ALLOWED_EXTENSIONS))}")

        analysis_id = str(uuid.uuid4())
        job_dir = self.jobs_dir / analysis_id
        job_dir.mkdir(parents=True, exist_ok=True)
        input_video_path = job_dir / f"input{ext}"

        total_bytes = 0
        with open(input_video_path, "wb") as buffer:
            while True:
                chunk = await file.read(64 * 1024)
                if not chunk:
                    break
                total_bytes += len(chunk)
                if total_bytes > MAX_UPLOAD_SIZE_BYTES:
                    buffer.close()
                    shutil.rmtree(job_dir, ignore_errors=True)
                    max_mb = MAX_UPLOAD_SIZE_BYTES // (1024 * 1024)
                    raise ValueError(f"Uploaded file exceeds maximum allowed size of {max_mb} MB.")
                buffer.write(chunk)

        if total_bytes == 0:
            shutil.rmtree(job_dir, ignore_errors=True)
            raise ValueError("Uploaded file is empty.")

        cap = cv2.VideoCapture(str(input_video_path))
        if not cap.isOpened():
            cap.release()
            shutil.rmtree(job_dir, ignore_errors=True)
            raise ValueError("Uploaded video file could not be parsed by video decoder. Please check the codec.")

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        if frame_count <= 0:
            shutil.rmtree(job_dir, ignore_errors=True)
            raise ValueError("Uploaded video contains no readable frames.")

        now_iso = datetime.now(timezone.utc).isoformat()
        job = JobRecord(
            analysis_id=analysis_id,
            status="queued",
            original_filename=clean_name,
            created_at=now_iso,
            progress=0,
            current_stage="Job queued"
        )
        self._jobs[analysis_id] = job
        self._save_job_meta(job)

        self.executor.submit(self._run_job_pipeline, analysis_id, input_video_path)
        logger.info(f"Queued analysis job: {analysis_id} for file '{clean_name}' ({total_bytes / (1024*1024):.2f} MB streamed)")
        return job

    def create_job(self, original_filename: str, video_bytes: bytes) -> JobRecord:
        """
        Validates uploaded video bytes and creates an isolated analysis job (sync fallback).
        """
        clean_name = Path(original_filename).name
        ext = Path(clean_name).suffix.lower()
        if ext not in ALLOWED_EXTENSIONS:
            raise ValueError(f"Unsupported file format '{ext}'. Allowed extensions: {', '.join(sorted(ALLOWED_EXTENSIONS))}")

        if len(video_bytes) > MAX_UPLOAD_SIZE_BYTES:
            max_mb = MAX_UPLOAD_SIZE_BYTES // (1024 * 1024)
            raise ValueError(f"Uploaded file exceeds maximum allowed size of {max_mb} MB.")

        if len(video_bytes) == 0:
            raise ValueError("Uploaded file is empty.")

        analysis_id = str(uuid.uuid4())
        job_dir = self.jobs_dir / analysis_id
        job_dir.mkdir(parents=True, exist_ok=True)

        input_video_path = job_dir / f"input{ext}"
        with open(input_video_path, "wb") as f:
            f.write(video_bytes)

        cap = cv2.VideoCapture(str(input_video_path))
        if not cap.isOpened():
            cap.release()
            shutil.rmtree(job_dir, ignore_errors=True)
            raise ValueError("Uploaded video file could not be parsed by video decoder. Please check the codec.")

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        if frame_count <= 0:
            shutil.rmtree(job_dir, ignore_errors=True)
            raise ValueError("Uploaded video contains no readable frames.")

        now_iso = datetime.now(timezone.utc).isoformat()
        job = JobRecord(
            analysis_id=analysis_id,
            status="queued",
            original_filename=clean_name,
            created_at=now_iso,
            progress=0,
            current_stage="Job queued"
        )
        self._jobs[analysis_id] = job
        self._save_job_meta(job)

        self.executor.submit(self._run_job_pipeline, analysis_id, input_video_path)
        logger.info(f"Queued analysis job: {analysis_id} for file '{clean_name}'")
        return job

    def _save_job_meta(self, job: JobRecord):
        meta_file = self.jobs_dir / job.analysis_id / "job_meta.json"
        try:
            with open(meta_file, "w", encoding="utf-8") as f:
                json.dump(job.to_dict(), f, indent=4)
        except Exception as e:
            logger.error(f"Failed to persist job metadata for {job.analysis_id}: {e}")

    def _run_job_pipeline(self, analysis_id: str, input_video_path: Path):
        """Worker executing the existing AI risk pipeline in isolated job folder."""
        job = self._jobs.get(analysis_id)
        if not job:
            return

        job_dir = self.jobs_dir / analysis_id
        try:
            job.status = "processing"
            job.progress = 15
            job.current_stage = "Initializing AI risk pipeline and models"
            self._save_job_meta(job)

            # Load project configuration with job-isolated output paths
            out_video_path = job_dir / "annotated_output.mp4"
            features_csv_path = job_dir / "features.csv"
            timeline_csv_path = job_dir / "risk_timeline.csv"
            summary_json_path = job_dir / "summary.json"
            cell_grid_json_path = job_dir / "cell_grid_samples.json"
            crda_json_path = job_dir / "crda_explanations.json"

            overrides = {
                "system": {"device": os.getenv("TORCH_DEVICE", "auto")},
                "video": {"display_window": False},
                "paths": {
                    "output_video": str(out_video_path),
                    "features_csv": str(features_csv_path),
                    "timeline_csv": str(timeline_csv_path),
                    "summary_json": str(summary_json_path),
                    "cell_grid_json": str(cell_grid_json_path),
                    "crda_json": str(crda_json_path)
                }
            }

            config = load_config("config/config.yaml", overrides=overrides)
            pipeline = CrowdRiskPipeline(config)

            job.progress = 30
            job.current_stage = "Running detection, tracking, optical flow, and risk fusion"
            self._save_job_meta(job)

            # Process video (limit frames in demo/cloud if MAX_FRAMES is set)
            max_f = int(os.getenv("MAX_ANALYSIS_FRAMES", "0"))
            max_frames = max_f if max_f > 0 else None

            summary = pipeline.process_video(
                source=str(input_video_path),
                output_video_path=str(out_video_path),
                max_frames=max_frames,
                display=False
            )

            job.progress = 100
            job.status = "completed"
            job.current_stage = "Analysis complete"
            job.completed_at = datetime.now(timezone.utc).isoformat()
            job.summary = summary
            self._save_job_meta(job)
            logger.info(f"Successfully completed analysis job: {analysis_id}")

        except Exception as e:
            logger.error(f"Error executing analysis job {analysis_id}: {e}", exc_info=True)
            job.status = "failed"
            job.current_stage = "Analysis failed"
            job.error_message = "The video could not be processed. Please check the video format and try again."
            job.completed_at = datetime.now(timezone.utc).isoformat()
            self._save_job_meta(job)
        finally:
            import gc
            if "pipeline" in locals():
                del pipeline
            gc.collect()

    def get_job(self, analysis_id: str) -> Optional[JobRecord]:
        return self._jobs.get(analysis_id)

    def get_timeline(self, analysis_id: str) -> List[Dict[str, Any]]:
        """Returns parsed risk timeline rows as dictionaries."""
        csv_path = self.jobs_dir / analysis_id / "risk_timeline.csv"
        if not csv_path.exists():
            return []
        try:
            df = pd.read_csv(csv_path)
            return df.to_dict(orient="records")
        except Exception as e:
            logger.error(f"Failed to read timeline for {analysis_id}: {e}")
            return []

    def get_spatial_grid(self, analysis_id: str) -> List[Dict[str, Any]]:
        """Returns cell grid snapshots from JSON."""
        json_path = self.jobs_dir / analysis_id / "cell_grid_samples.json"
        if not json_path.exists():
            return []
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to read spatial grid for {analysis_id}: {e}")
            return []

    def get_crda(self, analysis_id: str) -> List[Dict[str, Any]]:
        """Returns CRDA counterfactual explanations."""
        json_path = self.jobs_dir / analysis_id / "crda_explanations.json"
        if not json_path.exists():
            return []
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to read CRDA for {analysis_id}: {e}")
            return []

    def get_file_path(self, analysis_id: str, file_type: str) -> Optional[Path]:
        """Resolves output file path safely, preventing directory traversal."""
        job_dir = (self.jobs_dir / analysis_id).resolve()
        if not job_dir.exists() or not job_dir.is_relative_to(self.jobs_dir):
            return None

        type_map = {
            "video": "annotated_output.mp4",
            "timeline": "risk_timeline.csv",
            "features": "features.csv",
            "summary": "summary.json",
            "crda": "crda_explanations.json",
            "spatial": "cell_grid_samples.json"
        }
        filename = type_map.get(file_type)
        if not filename:
            return None

        target = (job_dir / filename).resolve()
        if target.exists() and target.is_relative_to(job_dir):
            return target
        return None

    def get_demo_manifest(self) -> List[Dict[str, Any]]:
        """Returns precomputed benchmark demo items."""
        return [
            {
                "sample_id": "umn_indoor_clip4",
                "title": "UMN Indoor Corridor Benchmark (Clip 4)",
                "scene": "Indoor Evacuation Benchmark",
                "description": "Real surveillance benchmark of rapid crowd corridor evacuation with spatial bottleneck choking.",
                "duration_seconds": 8.33,
                "peak_risk": 0.72,
                "dominant_driver": "Disorder & Bottleneck",
                "is_demo": True
            },
            {
                "sample_id": "concert_crowd",
                "title": "Dense Crowd Concert Footage",
                "scene": "High-Density Event Arena",
                "description": "High-density crowd telemetry under elevated density and directional convergence.",
                "duration_seconds": 4.0,
                "peak_risk": 0.58,
                "dominant_driver": "Density",
                "is_demo": True
            }
        ]

    def get_demo_sample_data(self, sample_id: str) -> Optional[Dict[str, Any]]:
        """Loads precomputed demo results for instantaneous demonstration mode."""
        if sample_id in ["umn_indoor_clip4", "concert_crowd"]:
            # Load from results directory
            summary_path = Path("results/summary.json")
            timeline_path = Path("results/risk_timeline.csv")
            crda_path = Path("results/crda_explanations.json")
            spatial_path = Path("results/cell_grid_samples.json")

            summary = {}
            if summary_path.exists():
                try:
                    with open(summary_path, "r", encoding="utf-8") as f:
                        summary = json.load(f)
                except Exception:
                    pass

            timeline = []
            if timeline_path.exists():
                try:
                    df = pd.read_csv(timeline_path)
                    timeline = df.to_dict(orient="records")
                except Exception:
                    pass

            crda = []
            if crda_path.exists():
                try:
                    with open(crda_path, "r", encoding="utf-8") as f:
                        crda = json.load(f)
                except Exception:
                    pass

            spatial = []
            if spatial_path.exists():
                try:
                    with open(spatial_path, "r", encoding="utf-8") as f:
                        spatial = json.load(f)
                except Exception:
                    pass

            return {
                "analysis_id": f"demo-{sample_id}",
                "status": "completed",
                "is_demo": True,
                "title": "UMN Indoor Corridor Benchmark" if sample_id == "umn_indoor_clip4" else "Dense Crowd Concert Demo",
                "summary": summary,
                "timeline": timeline,
                "crda": crda,
                "spatial": spatial
            }
        return None
