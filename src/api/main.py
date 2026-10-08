"""
FastAPI Application Entry Point for Production Deployment on Render.
Exposes CORS-enabled crowd safety analytics, asynchronous video processing endpoints,
and health monitoring for the Vercel frontend.
"""

import logging
import os
from typing import List
from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .routes import router

logger = logging.getLogger("API")

# Determine Allowed Origins for CORS
allowed_origins_env = os.getenv("ALLOWED_ORIGINS", "")
frontend_origin_env = os.getenv("FRONTEND_ORIGIN", "")

default_origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:8000",
    "http://127.0.0.1:8000",
]

origins_set = set(default_origins)
if frontend_origin_env:
    for o in frontend_origin_env.split(","):
        cleaned = o.strip().rstrip("/")
        if cleaned:
            origins_set.add(cleaned)
if allowed_origins_env:
    for o in allowed_origins_env.split(","):
        cleaned = o.strip().rstrip("/")
        if cleaned:
            origins_set.add(cleaned)

app = FastAPI(
    title="AI-Based Crowd Panic & Risk Prediction API",
    description="Real-Time Computer Vision and Counterfactual Risk Attribution API for Evacuation Safety.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Enable CORS for Vercel deployments (*.vercel.app), custom domains, and local dev
app.add_middleware(
    CORSMiddleware,
    allow_origins=list(origins_set) if "*" not in origins_set else ["*"],
    allow_origin_regex=r"^https:\/\/([a-zA-Z0-9_-]+\.)*vercel\.app$",
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS", "HEAD", "PUT", "DELETE"],
    allow_headers=["*"],
    expose_headers=["*"],
)


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Prevents raw internal Python tracebacks from leaking to production clients."""
    logger.error(f"Unhandled error processing {request.method} {request.url.path}: {exc}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "An internal server error occurred while processing the request."}
    )


# Mount API Routes
app.include_router(router)


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("src.api.main:app", host="0.0.0.0", port=port, reload=False)
