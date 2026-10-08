"""
FastAPI Route Handlers for Crowd Risk Prediction Backend.
Provides endpoints for video analysis, asynchronous job tracking, telemetry streams,
precomputed demo benchmarking, and safe file exports.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
from fastapi import APIRouter, File, HTTPException, UploadFile, status
from fastapi.responses import FileResponse, JSONResponse

logger = logging.getLogger(__name__)

from .job_manager import AnalysisJobManager

router = APIRouter(prefix="/api")

# Global job manager instance
job_manager = AnalysisJobManager()


@router.get("/health", tags=["System"])
def health_check() -> Dict[str, str]:
    """
    Health check endpoint for Render deployment monitoring.
    Returns zero internal server paths or sensitive information.
    """
    return {
        "status": "ok",
        "service": "crowd-risk-api",
        "version": "1.0.0"
    }


@router.post("/analyze", tags=["Analysis"], status_code=status.HTTP_202_ACCEPTED)
async def submit_analysis(file: UploadFile = File(...)) -> Dict[str, Any]:
    """
    Submits a video file for asynchronous crowd safety and panic risk analysis.
    Returns a unique analysis_id for subsequent polling.
    """
    if not file or not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No video file provided."
        )

    try:
        job = await job_manager.create_job_from_upload(file)
        return {
            "analysis_id": job.analysis_id,
            "status": job.status,
            "filename": job.original_filename,
            "message": "Analysis job accepted and queued for processing."
        }
    except ValueError as ve:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(ve))
    except Exception as e:
        logger.error(f"Failed to initialize video processing job: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to initialize video processing job."
        )


@router.get("/analysis/{analysis_id}", tags=["Analysis"])
def get_analysis_status(analysis_id: str) -> Dict[str, Any]:
    """
    Polls the execution status and metrics summary for an analysis job.
    """
    # Check precomputed demo IDs
    if analysis_id.startswith("demo-"):
        sample_id = analysis_id.replace("demo-", "")
        demo_data = job_manager.get_demo_sample_data(sample_id)
        if demo_data:
            return demo_data
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Demo sample not found.")

    job = job_manager.get_job(analysis_id)
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Analysis job '{analysis_id}' not found."
        )

    return job.to_dict()


@router.get("/analysis/{analysis_id}/timeline", tags=["Telemetry"])
def get_analysis_timeline(analysis_id: str) -> List[Dict[str, Any]]:
    """
    Retrieves frame-by-frame risk, panic, bottleneck, and density timeline records.
    """
    if analysis_id.startswith("demo-"):
        sample_id = analysis_id.replace("demo-", "")
        data = job_manager.get_demo_sample_data(sample_id)
        return data.get("timeline", []) if data else []

    job = job_manager.get_job(analysis_id)
    if not job:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Analysis job not found.")

    if job.status != "completed":
        return []

    return job_manager.get_timeline(analysis_id)


@router.get("/analysis/{analysis_id}/spatial", tags=["Telemetry"])
def get_analysis_spatial(analysis_id: str) -> List[Dict[str, Any]]:
    """
    Retrieves 4x4 spatial grid cell feature representations and elevated risk zones.
    """
    if analysis_id.startswith("demo-"):
        sample_id = analysis_id.replace("demo-", "")
        data = job_manager.get_demo_sample_data(sample_id)
        return data.get("spatial", []) if data else []

    job = job_manager.get_job(analysis_id)
    if not job:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Analysis job not found.")

    if job.status != "completed":
        return []

    return job_manager.get_spatial_grid(analysis_id)


@router.get("/analysis/{analysis_id}/crda", tags=["Telemetry"])
def get_analysis_crda(analysis_id: str) -> List[Dict[str, Any]]:
    """
    Retrieves Spatially Localized Counterfactual Risk-Driver Attribution (CRDA) reports.
    """
    if analysis_id.startswith("demo-"):
        sample_id = analysis_id.replace("demo-", "")
        data = job_manager.get_demo_sample_data(sample_id)
        return data.get("crda", []) if data else []

    job = job_manager.get_job(analysis_id)
    if not job:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Analysis job not found.")

    if job.status != "completed":
        return []

    return job_manager.get_crda(analysis_id)


@router.get("/analysis/{analysis_id}/video", tags=["Media"])
def stream_analysis_video(analysis_id: str):
    """
    Streams the HUD-annotated output video for in-browser playback.
    """
    if analysis_id.startswith("demo-"):
        sample_id = analysis_id.replace("demo-", "")
        if sample_id == "concert_crowd":
            target = Path("results/hud_demo_concert.mp4")
        else:
            target = Path("results/hud_demo_umn_indoor.mp4")
            if not target.exists():
                target = Path("results/annotated_output.mp4")
        if target.exists():
            return FileResponse(str(target), media_type="video/mp4", filename=target.name)
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Demo video not found.")

    vid_path = job_manager.get_file_path(analysis_id, "video")
    if not vid_path or not vid_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Processed video is not ready or does not exist."
        )

    return FileResponse(str(vid_path), media_type="video/mp4", filename=f"analysis_{analysis_id}.mp4")


@router.get("/analysis/{analysis_id}/download/{file_type}", tags=["Export"])
def download_result_file(analysis_id: str, file_type: str):
    """
    Downloads raw analytical outputs (video, timeline, features, summary, crda).
    """
    file_path = job_manager.get_file_path(analysis_id, file_type)
    if not file_path or not file_path.exists():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Requested file '{file_type}' not found.")

    media_types = {
        "video": "video/mp4",
        "timeline": "text/csv",
        "features": "text/csv",
        "summary": "application/json",
        "crda": "application/json",
        "spatial": "application/json"
    }
    media_type = media_types.get(file_type, "application/octet-stream")
    return FileResponse(str(file_path), media_type=media_type, filename=file_path.name)


@router.get("/demo/samples", tags=["Demo"])
def list_demo_samples() -> List[Dict[str, Any]]:
    """
    Lists precomputed benchmark demonstrations for instant verification.
    """
    return job_manager.get_demo_manifest()


@router.get("/demo/{sample_id}", tags=["Demo"])
def get_demo_sample(sample_id: str) -> Dict[str, Any]:
    """
    Retrieves full precomputed data for a benchmark scenario.
    """
    data = job_manager.get_demo_sample_data(sample_id)
    if not data:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Demo sample '{sample_id}' not found.")
    return data
