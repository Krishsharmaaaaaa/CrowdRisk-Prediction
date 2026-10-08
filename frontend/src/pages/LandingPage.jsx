import { Shield, Zap, Brain, Map, BarChart3, AlertTriangle } from 'lucide-react';

const features = [
  {
    icon: <Shield size={22} />,
    color: '#3b82f6',
    title: 'YOLOv8 + ByteTrack Detection',
    desc: 'Frame-level person detection and multi-object tracking with sub-second latency.',
  },
  {
    icon: <Brain size={22} />,
    color: '#6366f1',
    title: 'CRDA Counterfactual Attribution',
    desc: 'Spatially localized counterfactual risk-driver attribution across all 15 driver subsets.',
  },
  {
    icon: <Map size={22} />,
    color: '#f59e0b',
    title: '4×4 Spatial Risk Heatmap',
    desc: 'Per-cell risk scoring across a 16-zone spatial grid with elevated-zone highlighting.',
  },
  {
    icon: <BarChart3 size={22} />,
    color: '#22c55e',
    title: 'Temporal Risk Timeline',
    desc: 'Frame-by-frame risk, panic, and bottleneck evolution with R_safe threshold overlay.',
  },
  {
    icon: <Zap size={22} />,
    color: '#f97316',
    title: 'Early Warning System',
    desc: 'Configurable multi-stage alert triggering for critical crowd density transitions.',
  },
  {
    icon: <AlertTriangle size={22} />,
    color: '#a855f7',
    title: 'Decision-Support HUD',
    desc: 'Operator-readable overlays with minimal intervention subsets and non-causal disclaimers.',
  },
];

export default function LandingPage({ onAnalyzeClick, onDemoClick }) {
  return (
    <div className="page-wrapper">
      {/* Hero */}
      <section className="hero">
        <div className="container">
          <div className="hero-eyebrow animate-in">
            <span style={{ width: 6, height: 6, borderRadius: '50%', background: 'currentColor' }} />
            AI-Based Crowd Safety Research System
          </div>

          <h1 className="hero-title animate-in animate-in-delay-1">
            Real-Time Crowd Risk<br />
            <span>Prediction & Attribution</span>
          </h1>

          <p className="hero-sub animate-in animate-in-delay-2">
            Upload a crowd surveillance video or run a precomputed benchmark demo.
            The pipeline runs YOLOv8 detection, ByteTrack tracking, optical flow,
            panic/bottleneck scoring, CRDA counterfactual attribution, and a
            Decision-Support HUD.
          </p>

          <div className="hero-actions animate-in animate-in-delay-3">
            <button
              id="btn-analyze"
              className="btn btn-primary btn-lg"
              onClick={onAnalyzeClick}
            >
              <Shield size={18} />
              Analyse a Video
            </button>
            <button
              id="btn-demo"
              className="btn btn-secondary btn-lg"
              onClick={onDemoClick}
            >
              <Zap size={18} />
              Try Precomputed Demo
            </button>
          </div>
        </div>
      </section>

      {/* Features */}
      <section style={{ padding: '0 0 var(--sp-12)' }}>
        <div className="container">
          <div style={{ textAlign: 'center', marginBottom: 'var(--sp-8)' }}>
            <h2 style={{ fontSize: '1.5rem', fontWeight: 700, marginBottom: 'var(--sp-3)',
              letterSpacing: '-0.02em' }}>
              Pipeline Architecture
            </h2>
            <p style={{ color: 'var(--clr-text-secondary)', fontSize: '0.95rem', maxWidth: 520, margin: '0 auto' }}>
              Stages 0–9: from baseline freeze to interactive HUD demonstration,
              46 passing unit tests.
            </p>
          </div>

          <div className="feature-grid">
            {features.map((f, i) => (
              <div
                key={i}
                className="feature-card animate-in"
                style={{ animationDelay: `${0.05 * i}s`, opacity: 0, animationFillMode: 'forwards' }}
                id={`feature-card-${i}`}
              >
                <div
                  className="feature-icon"
                  style={{
                    background: f.color + '18',
                    border: `1px solid ${f.color}30`,
                    color: f.color,
                  }}
                >
                  {f.icon}
                </div>
                <h3 className="feature-title">{f.title}</h3>
                <p className="feature-desc">{f.desc}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Scientific disclaimer */}
      <div className="container" style={{ paddingBottom: 'var(--sp-8)' }}>
        <div className="disclaimer-strip">
          <AlertTriangle size={14} style={{ color: 'rgba(245,158,11,0.9)', flexShrink: 0, marginTop: 1 }} />
          <p className="disclaimer-text">
            <strong>Scientific Disclaimer:</strong> All risk scores are model-derived,
            non-probabilistic indices. CRDA attributions are non-causal model sensitivity
            indicators and do not constitute prescriptive operational safety guidance.
            System validated on UMN Benchmark (Stages 3–5) and microscopic pedestrian
            simulation (Stage 7). For research purposes only.
          </p>
        </div>
      </div>
    </div>
  );
}
