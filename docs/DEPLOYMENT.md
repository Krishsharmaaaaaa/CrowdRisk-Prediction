# Production Deployment Guide

**Stage 10: Production Deployment**
**Architecture**: Vercel (Frontend) + Render (FastAPI Backend)

---

## Architecture Overview

```
USER BROWSER
    |
VERCEL FRONTEND  (React + Vite)
    | HTTPS REST API
RENDER FASTAPI BACKEND  (Python 3.11, uvicorn)
    |
CROWD RISK PIPELINE
  - YOLOv8n person detection
  - ByteTrack multi-object tracking
  - Optical flow & feature extraction
  - Risk fusion (Random Forest model)
  - Early warning system
  - 4x4 Spatial Cell Grid
  - CRDA counterfactual attribution engine
    |
RESULTS (JSON + MP4) --> REST API --> Vercel frontend
```

---

## Part 1 — Deploy the Backend to Render

### Prerequisites
- A [Render](https://render.com) account
- The repository pushed to GitHub/GitLab

### Steps

1. **Push the repository** to GitHub:
   ```bash
   git add .
   git commit -m "feat: add production deployment (Stage 10)"
   git push origin main
   ```

2. **Create a new Render Web Service**:
   - Go to [dashboard.render.com](https://dashboard.render.com)
   - Click **New → Web Service**
   - Connect your GitHub repository
   - Render will auto-detect `render.yaml`

3. **Settings** (if not auto-detected from `render.yaml`):

   | Setting | Value |
   |---|---|
   | Runtime | Python |
   | Build Command | `pip install -r requirements-backend.txt` |
   | Start Command | `uvicorn src.api.main:app --host 0.0.0.0 --port $PORT` |
   | Instance Type | Free (or Starter for persistent storage) |
   | Health Check Path | `/api/health` |

4. **Environment Variables** (set in Render dashboard → Environment):

   | Key | Value |
   |---|---|
   | `PYTHON_VERSION` | `3.11.0` |
   | `TORCH_DEVICE` | `cpu` |
   | `MAX_UPLOAD_SIZE_MB` | `150` |
   | `MAX_ANALYSIS_FRAMES` | `300` |
   | `FRONTEND_ORIGIN` | *(your Vercel URL, set after step 5)* |

5. **Note your Render URL**: e.g. `https://crowdrisk-api.onrender.com`

6. **Test the health endpoint**:
   ```bash
   curl https://crowdrisk-api.onrender.com/api/health
   # -> {"status":"ok","service":"crowd-risk-api","version":"1.0.0"}
   ```

> **Note**: On the Render Free tier, the service sleeps after 15 minutes of inactivity.
> The first request after a sleep takes ~30s to warm up. Use Starter plan for always-on.

---

## Part 2 — Deploy the Frontend to Vercel

### Prerequisites
- A [Vercel](https://vercel.com) account

### Steps

1. **Import the project** on Vercel:
   - Go to [vercel.com/new](https://vercel.com/new)
   - Import the GitHub repository
   - Set **Root Directory** to `frontend`
   - Framework preset: **Vite** (auto-detected)

2. **Build settings** (auto-detected from `vercel.json`):

   | Setting | Value |
   |---|---|
   | Build Command | `npm run build` |
   | Output Directory | `dist` |
   | Install Command | `npm install` |

3. **Environment Variables** (Vercel → Settings → Environment Variables):

   | Key | Value |
   |---|---|
   | `VITE_API_BASE_URL` | `https://crowdrisk-api.onrender.com` |

4. **Deploy** — Vercel auto-deploys on every push to `main`.

5. **Note your Vercel URL**: e.g. `https://crowdrisk.vercel.app`

6. **Back in Render**, set `FRONTEND_ORIGIN = https://crowdrisk.vercel.app`
   and `ALLOWED_ORIGINS = https://crowdrisk.vercel.app`

---

## Part 3 — Local Development

### Run backend locally
```bash
pip install -r requirements-backend.txt
uvicorn src.api.main:app --reload --port 8000
# API at http://localhost:8000
# Swagger UI at http://localhost:8000/docs
```

### Run frontend locally
```bash
cd frontend
cp .env.example .env.local
# Set VITE_API_BASE_URL=http://localhost:8000 in .env.local
npm install
npm run dev
# Frontend at http://localhost:5173
```

---

## Part 4 — Deployment Checklist

- [ ] `render.yaml` is at the repo root
- [ ] `requirements-backend.txt` uses `opencv-python-headless`
- [ ] `frontend/vercel.json` has SPA rewrites
- [ ] `VITE_API_BASE_URL` set in Vercel env vars
- [ ] `FRONTEND_ORIGIN` set in Render env vars
- [ ] `GET /api/health` returns `{"status":"ok"}`
- [ ] Demo benchmarks load at `/api/demo/samples`
- [ ] Video upload works end-to-end

---

## Part 5 — API Reference

| Endpoint | Method | Description |
|---|---|---|
| `/api/health` | GET | Health check for Render monitoring |
| `/api/demo/samples` | GET | List precomputed benchmark demos |
| `/api/demo/{sample_id}` | GET | Load full precomputed demo |
| `/api/analyze` | POST | Submit video file for async analysis |
| `/api/analysis/{id}` | GET | Poll job status and summary |
| `/api/analysis/{id}/timeline` | GET | Frame-by-frame risk timeline |
| `/api/analysis/{id}/spatial` | GET | 4x4 spatial grid data |
| `/api/analysis/{id}/crda` | GET | CRDA counterfactual reports |
| `/api/analysis/{id}/video` | GET | Stream HUD-annotated output video |
| `/api/analysis/{id}/download/{type}` | GET | Download result files |

Full OpenAPI schema available at: `https://your-render-url.onrender.com/docs`

---

## Part 6 — Free Tier Limitations

| Resource | Render Free | Render Starter |
|---|---|---|
| Sleep after inactivity | 15 min | No sleep |
| Disk persistence | Ephemeral | Persistent disk |
| CPU | Shared | Dedicated |
| Monthly hours | 750 | Unlimited |

For persistent job storage on Render, add a mounted disk and set `JOBS_DIR=/data/jobs`.

---

## Stage 10 Files Created

| File | Purpose |
|---|---|
| `render.yaml` | Render service definition |
| `requirements-backend.txt` | Production Python deps (headless OpenCV) |
| `server.py` | Local convenience entry point |
| `frontend/` | Complete Vite+React frontend |
| `frontend/vercel.json` | Vercel SPA routing config |
| `frontend/.env.example` | Env variable template |
| `frontend/src/api.js` | REST API client |
| `src/api/main.py` | FastAPI app (CORS-ready for Vercel) |
| `src/api/routes.py` | All API route handlers |
| `src/api/job_manager.py` | Async job pool manager |
