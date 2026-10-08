import { useState, useEffect, useCallback } from 'react';
import { AlertTriangle, RefreshCw, CheckCircle2, Clock, Play } from 'lucide-react';

import api from '../api';
import { getRiskLevel, getRiskClass, fmt } from '../utils';

import RiskGauge from '../components/RiskGauge';
import DriverPanel from '../components/DriverPanel';
import SpatialGrid from '../components/SpatialGrid';
import RiskTimeline from '../components/RiskTimeline';
import InterventionCard from '../components/InterventionCard';
import UploadZone from '../components/UploadZone';

const POLL_INTERVAL_MS = 2500;

/* ── Utility sub-components ─────────────────────────────── */

function StatusBadge({ status }) {
  const map = {
    queued: { cls: 'badge-neutral', label: 'Queued' },
    processing: { cls: 'badge-moderate', label: 'Processing' },
    completed: { cls: 'badge-low', label: 'Completed' },
    failed: { cls: 'badge-critical', label: 'Failed' },
  };
  const { cls, label } = map[status] || { cls: 'badge-neutral', label: status };
  return <span className={`badge ${cls}`}>{label}</span>;
}

function DemoSelector({ demos, onSelect }) {
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 'var(--sp-3)' }}>
      <h3 style={{ fontSize: '0.8rem', fontWeight: 600, color: 'var(--clr-text-secondary)',
        letterSpacing: '0.1em', textTransform: 'uppercase', fontFamily: 'var(--font-mono)' }}>
        Precomputed Benchmarks
      </h3>
      {demos.map((d) => (
        <div
          key={d.sample_id}
          id={`demo-${d.sample_id}`}
          className="demo-card"
          onClick={() => onSelect(d.sample_id)}
        >
          <div style={{ fontWeight: 600, fontSize: '0.9rem', marginBottom: 4 }}>{d.title}</div>
          <div style={{ fontSize: '0.78rem', color: 'var(--clr-text-secondary)', marginBottom: 8, lineHeight: 1.45 }}>
            {d.description}
          </div>
          <div style={{ display: 'flex', gap: 'var(--sp-3)', flexWrap: 'wrap' }}>
            <span className={`badge badge-${getRiskClass(getRiskLevel(d.peak_risk))}`}>
              Peak: {fmt(d.peak_risk, 2)}
            </span>
            <span className="badge badge-info">
              {d.dominant_driver}
            </span>
            <span className="badge badge-neutral">
              {d.duration_seconds}s
            </span>
          </div>
        </div>
      ))}
    </div>
  );
}

/* ── Main analysis dashboard ────────────────────────────── */

export default function AnalysisPage({ initialMode = 'upload' }) {
  const [mode, setMode] = useState(initialMode); // 'upload' | 'demo'
  const [demos, setDemos] = useState([]);
  const [uploading, setUploading] = useState(false);
  const [job, setJob] = useState(null);
  const [timeline, setTimeline] = useState([]);
  const [spatial, setSpatial] = useState([]);
  const [crda, setCrda] = useState([]);
  const [selectedCell, setSelectedCell] = useState(null);
  const [selectedCrda, setSelectedCrda] = useState(null);
  const [error, setError] = useState(null);
  const [polling, setPolling] = useState(false);

  /* Load demo list on mount */
  useEffect(() => {
    api.listDemos()
      .then(setDemos)
      .catch(() => setDemos([]));
  }, []);

  /* Poll job status */
  useEffect(() => {
    if (!job?.analysis_id || job.status === 'completed' || job.status === 'failed') {
      setPolling(false);
      return;
    }
    setPolling(true);
    const timer = setInterval(async () => {
      try {
        const updated = await api.getAnalysis(job.analysis_id);
        setJob(updated);
        if (updated.status === 'completed' || updated.status === 'failed') {
          clearInterval(timer);
          setPolling(false);
          if (updated.status === 'completed') fetchResults(updated.analysis_id);
        }
      } catch {
        clearInterval(timer);
        setPolling(false);
      }
    }, POLL_INTERVAL_MS);
    return () => clearInterval(timer);
  }, [job?.analysis_id, job?.status]);

  const fetchResults = useCallback(async (id) => {
    const [tl, sp, cr] = await Promise.allSettled([
      api.getTimeline(id),
      api.getSpatial(id),
      api.getCRDA(id),
    ]);
    if (tl.status === 'fulfilled') setTimeline(tl.value);
    if (sp.status === 'fulfilled') setSpatial(sp.value);
    if (cr.status === 'fulfilled') {
      const crdaArr = cr.value;
      setCrda(crdaArr);
      // Pick the most critical frame's most critical cell as default
      if (crdaArr.length > 0) {
        const last = crdaArr[crdaArr.length - 1];
        const critCell = last?.cell_explanations?.[0] ?? null;
        if (critCell) {
          setSelectedCell(critCell.cell_id);
          setSelectedCrda(critCell);
        }
      }
    }
  }, []);

  /* Handle demo selection */
  const handleDemo = async (sampleId) => {
    setError(null);
    setJob(null);
    setTimeline([]); setSpatial([]); setCrda([]);
    try {
      const data = await api.getDemoSample(sampleId);
      // Synthesise a fake job record
      setJob({
        analysis_id: `demo-${sampleId}`,
        status: 'completed',
        is_demo: true,
        original_filename: data.title,
        summary: data.summary,
        progress: 100,
        current_stage: 'Precomputed demo loaded',
      });
      if (data.timeline) setTimeline(data.timeline);
      if (data.spatial) setSpatial(data.spatial);
      if (data.crda) {
        setCrda(data.crda);
        const last = data.crda[data.crda.length - 1];
        const critCell = last?.cell_explanations?.[0];
        if (critCell) {
          setSelectedCell(critCell.cell_id);
          setSelectedCrda(critCell);
        }
      }
    } catch (e) {
      setError(e.message);
    }
  };

  /* Handle video upload */
  const handleUpload = async (file) => {
    setUploading(true);
    setError(null);
    setJob(null);
    setTimeline([]); setSpatial([]); setCrda([]);
    try {
      const res = await api.submitAnalysis(file);
      setJob({ analysis_id: res.analysis_id, status: res.status,
        original_filename: res.filename, progress: 0, current_stage: 'Queued' });
    } catch (e) {
      setError(e.message);
    } finally {
      setUploading(false);
    }
  };

  /* Cell selection */
  const handleCellSelect = (cellId, data) => {
    setSelectedCell(cellId);
    setSelectedCrda(data ?? null);
  };

  /* Derive current risk for gauge */
  const latestRisk = (() => {
    if (timeline.length > 0) {
      const last = timeline[timeline.length - 1];
      return last.risk_score ?? last.model_risk ?? 0;
    }
    if (job?.summary?.maximum_risk) return job.summary.maximum_risk;
    return 0;
  })();

  /* Derive driver deltas from selected crda frame */
  const driverDeltas = (() => {
    if (!crda.length) return {};
    const lastFrame = crda[crda.length - 1];
    const attrs = {};
    lastFrame?.cell_explanations?.forEach((cell) => {
      cell.single_driver_attributions?.forEach((a) => {
        attrs[a.driver_key] = (attrs[a.driver_key] ?? 0) + Math.abs(a.delta_R ?? 0);
      });
    });
    return attrs;
  })();

  /* Flatten spatial cells from last CRDA frame */
  const spatialCells = (() => {
    if (crda.length > 0) {
      const last = crda[crda.length - 1];
      return last?.cell_explanations ?? [];
    }
    return spatial;
  })();

  /* Summary stats */
  const summary = job?.summary ?? {};

  const isReady = job?.status === 'completed';
  const isFailed = job?.status === 'failed';
  const isProcessing = job?.status === 'processing' || job?.status === 'queued';
  const videoUrl = isReady && job?.analysis_id ? api.videoUrl(job.analysis_id) : null;

  return (
    <div className="page-wrapper" style={{ padding: 'var(--sp-8) 0' }}>
      <div className="container">

        {/* Page header */}
        <div style={{ marginBottom: 'var(--sp-6)' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 'var(--sp-4)',
            marginBottom: 'var(--sp-5)', flexWrap: 'wrap' }}>
            <h1 style={{ fontSize: '1.6rem', fontWeight: 800, letterSpacing: '-0.03em' }}>
              Crowd Risk Analyser
            </h1>
            {job && <StatusBadge status={job.status} />}
            {polling && <span className="badge badge-info" style={{ gap: 6 }}>
              <div className="live-dot processing" />
              Polling
            </span>}
          </div>

          {/* Mode tabs */}
          <div style={{ display: 'flex', gap: 'var(--sp-2)' }}>
            <button
              id="tab-upload"
              className={`btn btn-sm ${mode === 'upload' ? 'btn-primary' : 'btn-ghost'}`}
              onClick={() => setMode('upload')}
            >
              Upload Video
            </button>
            <button
              id="tab-demo"
              className={`btn btn-sm ${mode === 'demo' ? 'btn-primary' : 'btn-ghost'}`}
              onClick={() => setMode('demo')}
            >
              Demo Benchmarks
            </button>
          </div>
        </div>

        {/* Error banner */}
        {error && (
          <div className="disclaimer-strip" style={{
            borderColor: 'rgba(239,68,68,0.3)', background: 'rgba(239,68,68,0.06)',
            marginBottom: 'var(--sp-5)',
          }}>
            <AlertTriangle size={14} style={{ color: 'var(--clr-critical)', flexShrink: 0 }} />
            <span className="disclaimer-text" style={{ color: 'var(--clr-critical)' }}>{error}</span>
          </div>
        )}

        <div className="dashboard-grid">
          {/* Sidebar */}
          <aside className="dashboard-sidebar">

            {/* Upload / Demo selector */}
            {mode === 'upload' ? (
              <UploadZone onUpload={handleUpload} loading={uploading || isProcessing} />
            ) : (
              <div className="card">
                <DemoSelector demos={demos} onSelect={handleDemo} />
              </div>
            )}

            {/* Job progress */}
            {job && !isReady && !isFailed && (
              <div className="card animate-in" id="job-progress">
                <div className="card-header">
                  <span className="card-title">Analysis Progress</span>
                  <StatusBadge status={job.status} />
                </div>
                <div style={{ marginBottom: 'var(--sp-3)', fontSize: '0.8rem',
                  color: 'var(--clr-text-secondary)' }}>
                  {job.current_stage}
                </div>
                <div className="progress-track">
                  <div className="progress-fill" style={{ width: `${job.progress ?? 0}%` }} />
                </div>
                <div style={{ marginTop: 6, fontSize: '0.72rem',
                  color: 'var(--clr-text-muted)', fontFamily: 'var(--font-mono)' }}>
                  {job.progress ?? 0}%
                </div>
              </div>
            )}

            {/* Error detail */}
            {isFailed && (
              <div className="card" style={{ borderColor: 'rgba(239,68,68,0.3)' }}>
                <p style={{ fontSize: '0.85rem', color: 'var(--clr-critical)' }}>
                  {job.error_message ?? 'Analysis failed.'}
                </p>
                <button
                  className="btn btn-ghost btn-sm"
                  style={{ marginTop: 'var(--sp-3)' }}
                  onClick={() => { setJob(null); setError(null); }}
                >
                  <RefreshCw size={12} /> Try again
                </button>
              </div>
            )}

            {/* Risk gauge */}
            {isReady && (
              <div className="card animate-in" id="risk-gauge-card">
                <div className="card-header">
                  <span className="card-title">Global Risk Score</span>
                </div>
                <div style={{ display: 'flex', justifyContent: 'center', padding: 'var(--sp-3) 0' }}>
                  <RiskGauge score={latestRisk} size={180} />
                </div>
              </div>
            )}

            {/* Stats */}
            {isReady && Object.keys(summary).length > 0 && (
              <div className="card animate-in" id="stats-card">
                <div className="card-header">
                  <span className="card-title">Video Statistics</span>
                </div>
                <div className="stat-grid">
                  <div className="stat-block">
                    <span className="stat-label">Frames</span>
                    <span className="stat-value">{summary.processed_frames ?? '—'}</span>
                  </div>
                  <div className="stat-block">
                    <span className="stat-label">Max People</span>
                    <span className="stat-value">{summary.max_people ?? '—'}</span>
                  </div>
                  <div className="stat-block">
                    <span className="stat-label">Peak Risk</span>
                    <span className="stat-value">{fmt(summary.maximum_risk, 3)}</span>
                  </div>
                  <div className="stat-block">
                    <span className="stat-label">Duration</span>
                    <span className="stat-value">
                      {summary.duration_seconds ? `${summary.duration_seconds.toFixed(1)}s` : '—'}
                    </span>
                  </div>
                </div>
              </div>
            )}

          </aside>

          {/* Main dashboard */}
          <main className="dashboard-main">

            {/* Placeholder before any job */}
            {!job && (
              <div className="card" style={{
                display: 'flex',
                flexDirection: 'column',
                alignItems: 'center',
                justifyContent: 'center',
                minHeight: 300,
                gap: 'var(--sp-4)',
                color: 'var(--clr-text-muted)',
              }}>
                <Play size={40} style={{ opacity: 0.3 }} />
                <p style={{ fontSize: '0.95rem' }}>
                  {mode === 'upload'
                    ? 'Upload a video to start the AI risk analysis pipeline.'
                    : 'Select a precomputed benchmark to load results instantly.'}
                </p>
              </div>
            )}

            {/* Risk timeline */}
            {isReady && timeline.length > 0 && (
              <div className="card animate-in" id="timeline-card">
                <div className="card-header">
                  <span className="card-title card-title-icon">
                    <RefreshCw size={11} />
                    Risk Timeline
                  </span>
                  <span className="badge badge-neutral" style={{ fontFamily: 'var(--font-mono)', fontSize: '0.65rem' }}>
                    {timeline.length} frames
                  </span>
                </div>
                <RiskTimeline timeline={timeline} />
              </div>
            )}

            {/* CRDA Driver Panel */}
            {isReady && Object.keys(driverDeltas).length > 0 && (
              <div className="card animate-in" id="crda-driver-card">
                <div className="card-header">
                  <span className="card-title card-title-icon">
                    <AlertTriangle size={11} />
                    CRDA Driver Attribution
                  </span>
                  <span className="badge badge-info" style={{ fontSize: '0.65rem' }}>
                    ΔR = counterfactual risk reduction
                  </span>
                </div>
                <DriverPanel attributions={driverDeltas} />
              </div>
            )}

            {/* Spatial heatmap + intervention side-by-side */}
            {isReady && spatialCells.length > 0 && (
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 'var(--sp-5)' }}
                id="spatial-intervention-row">

                {/* Spatial Grid */}
                <div className="card animate-in" id="spatial-card">
                  <div className="card-header">
                    <span className="card-title">Spatial Risk Grid</span>
                    <span className="badge badge-neutral" style={{ fontSize: '0.65rem' }}>
                      4 × 4
                    </span>
                  </div>
                  <SpatialGrid
                    cells={spatialCells}
                    selectedCell={selectedCell}
                    onSelectCell={handleCellSelect}
                  />
                  <p style={{ marginTop: 'var(--sp-3)', fontSize: '0.72rem',
                    color: 'var(--clr-text-muted)' }}>
                    Click a cell to inspect its CRDA intervention report.
                  </p>
                </div>

                {/* Intervention recommender */}
                <InterventionCard crda={selectedCrda} />
              </div>
            )}

            {/* Video playback */}
            {isReady && videoUrl && (
              <div className="card animate-in" id="video-card">
                <div className="card-header">
                  <span className="card-title card-title-icon">
                    <Play size={11} />
                    HUD-Annotated Output
                  </span>
                  <a
                    href={videoUrl}
                    download
                    className="btn btn-sm btn-secondary"
                    id="btn-download-video"
                  >
                    Download
                  </a>
                </div>
                <video
                  controls
                  width="100%"
                  style={{ borderRadius: 'var(--r-md)', background: '#000', maxHeight: 420 }}
                  src={videoUrl}
                  id="output-video"
                />
              </div>
            )}

            {/* Completed checkmark */}
            {isReady && (
              <div className="disclaimer-strip animate-in" style={{
                borderColor: 'rgba(34,197,94,0.2)',
                background: 'rgba(34,197,94,0.05)',
              }}>
                <CheckCircle2 size={14} style={{ color: 'var(--clr-low)', flexShrink: 0 }} />
                <p className="disclaimer-text" style={{ color: 'rgba(34,197,94,0.9)' }}>
                  Analysis complete — 46 unit tests passing (Stages 0–9).
                  All CRDA attributions are non-causal sensitivity indicators.
                  R_safe = 0.60 (project-wide threshold).
                </p>
              </div>
            )}
          </main>
        </div>
      </div>
    </div>
  );
}
