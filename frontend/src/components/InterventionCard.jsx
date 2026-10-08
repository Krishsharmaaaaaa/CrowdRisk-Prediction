import { CheckCircle, AlertTriangle, XCircle, ArrowRight } from 'lucide-react';
import { fmt, getRiskLevel, getRiskClass } from '../utils';

/**
 * Minimal Intervention Recommender card.
 * Displays the minimal driver subset S* that drives a cell below R_safe.
 */
export default function InterventionCard({ crda = null }) {
  if (!crda) {
    return (
      <div className="card" id="intervention-card">
        <div className="card-header">
          <span className="card-title card-title-icon">
            <AlertTriangle size={12} />
            Minimal Intervention
          </span>
        </div>
        <p style={{ fontSize: '0.85rem', color: 'var(--clr-text-muted)', textAlign: 'center', padding: '1rem 0' }}>
          Select an elevated-risk cell to see CRDA intervention recommendations.
        </p>
      </div>
    );
  }

  const {
    cell_id,
    initial_risk,
    minimal_intervention_subset,
    safe_threshold_reached,
    predicted_cf_risk,
    predicted_delta_R,
  } = crda;

  const initialLevel = getRiskLevel(initial_risk ?? 0);
  const cfLevel = getRiskLevel(predicted_cf_risk ?? 0);
  const safeThold = 0.60;

  const safeReached = safe_threshold_reached ?? (predicted_cf_risk != null && predicted_cf_risk < safeThold);
  const subset = minimal_intervention_subset ?? [];

  return (
    <div className="card" id="intervention-card" style={{
      borderColor: safeReached ? 'rgba(34,197,94,0.3)' : 'rgba(249,115,22,0.3)',
    }}>
      <div className="card-header">
        <span className="card-title card-title-icon">
          {safeReached
            ? <CheckCircle size={12} style={{ color: 'var(--clr-low)' }} />
            : <XCircle size={12} style={{ color: 'var(--clr-high)' }} />}
          Minimal Intervention — {cell_id}
        </span>
        <span className={`badge badge-${getRiskClass(initialLevel)}`}>
          {initialLevel}
        </span>
      </div>

      {/* Risk transition */}
      <div style={{
        display: 'flex',
        alignItems: 'center',
        gap: 'var(--sp-3)',
        marginBottom: 'var(--sp-4)',
        background: 'rgba(255,255,255,0.03)',
        borderRadius: 'var(--r-md)',
        padding: 'var(--sp-3) var(--sp-4)',
      }}>
        <div style={{ textAlign: 'center' }}>
          <div style={{ fontFamily: 'var(--font-mono)', fontSize: '1.3rem', fontWeight: 700,
            color: 'var(--clr-high)' }}>
            {fmt(initial_risk, 4)}
          </div>
          <div style={{ fontSize: '0.65rem', color: 'var(--clr-text-muted)' }}>Initial</div>
        </div>

        <ArrowRight size={16} style={{ color: 'var(--clr-text-muted)', flexShrink: 0 }} />

        <div style={{ textAlign: 'center' }}>
          <div style={{ fontFamily: 'var(--font-mono)', fontSize: '1.3rem', fontWeight: 700,
            color: safeReached ? 'var(--clr-low)' : 'var(--clr-moderate)' }}>
            {fmt(predicted_cf_risk, 4)}
          </div>
          <div style={{ fontSize: '0.65rem', color: 'var(--clr-text-muted)' }}>CF Risk</div>
        </div>

        <div style={{ marginLeft: 'auto', textAlign: 'right' }}>
          <div style={{ fontFamily: 'var(--font-mono)', fontSize: '1rem', fontWeight: 700,
            color: 'var(--clr-accent-bright)' }}>
            ΔR = {fmt(predicted_delta_R, 4)}
          </div>
          <div style={{ fontSize: '0.65rem', color: 'var(--clr-text-muted)' }}>Reduction</div>
        </div>
      </div>

      {/* Intervention subset */}
      <div style={{ marginBottom: 'var(--sp-3)' }}>
        <div style={{ fontSize: '0.72rem', color: 'var(--clr-text-muted)', marginBottom: 'var(--sp-2)' }}>
          Minimal intervention subset S*
        </div>
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: 'var(--sp-2)' }}>
          {subset.length > 0
            ? subset.map((d) => (
              <span key={d} className="intervention-tag">
                ↓ {d}
              </span>
            ))
            : <span style={{ fontSize: '0.8rem', color: 'var(--clr-text-muted)' }}>—</span>}
        </div>
      </div>

      {/* Safe threshold status */}
      {safeReached ? (
        <div style={{
          background: 'rgba(34,197,94,0.08)',
          border: '1px solid rgba(34,197,94,0.25)',
          borderRadius: 'var(--r-sm)',
          padding: 'var(--sp-2) var(--sp-3)',
          display: 'flex',
          alignItems: 'center',
          gap: 'var(--sp-2)',
          fontSize: '0.75rem',
          color: 'var(--clr-low)',
        }}>
          <CheckCircle size={12} />
          Safe threshold reached (R_safe = {safeThold})
        </div>
      ) : (
        <div style={{
          background: 'rgba(249,115,22,0.08)',
          border: '1px solid rgba(249,115,22,0.25)',
          borderRadius: 'var(--r-sm)',
          padding: 'var(--sp-2) var(--sp-3)',
          display: 'flex',
          alignItems: 'center',
          gap: 'var(--sp-2)',
          fontSize: '0.75rem',
          color: 'var(--clr-high)',
        }}>
          <XCircle size={12} />
          NO SAFE-THRESHOLD INTERVENTION FOUND (R_safe = {safeThold})
        </div>
      )}

      {/* Disclaimer */}
      <div style={{ marginTop: 'var(--sp-3)', fontSize: '0.68rem',
        color: 'var(--clr-text-dim)', fontStyle: 'italic' }}>
        Counterfactual attributions are non-causal model sensitivity indicators.
        Not prescriptive operational guidance.
      </div>
    </div>
  );
}
