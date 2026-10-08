import { driverColor, driverName, fmt } from '../utils';

const DRIVER_KEYS = ['D', 'O', 'B', 'K'];

/**
 * CRDA Driver Attribution Panel.
 * Shows ΔR bar for each driver, ranked by attribution magnitude.
 */
export default function DriverPanel({ attributions = {} }) {
  // Build sorted array
  const rows = DRIVER_KEYS.map((k) => ({
    key: k,
    name: driverName(k),
    delta: attributions[k] ?? 0,
  })).sort((a, b) => Math.abs(b.delta) - Math.abs(a.delta));

  const maxDelta = Math.max(...rows.map((r) => Math.abs(r.delta)), 0.001);

  return (
    <div id="driver-panel" style={{ display: 'flex', flexDirection: 'column', gap: 'var(--sp-3)' }}>
      {rows.map((row, idx) => {
        const barWidth = (Math.abs(row.delta) / maxDelta) * 100;
        const color = driverColor(row.key);
        return (
          <div key={row.key} className="driver-bar-row" id={`driver-bar-${row.key}`}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 'var(--sp-2)' }}>
              <span style={{
                fontFamily: 'var(--font-mono)',
                fontSize: '0.62rem',
                fontWeight: 700,
                color: 'var(--clr-text-muted)',
                width: '1.2rem',
                textAlign: 'center',
              }}>
                #{idx + 1}
              </span>
              <span className="driver-bar-label">{row.name}</span>
            </div>

            <div className="driver-bar-track">
              <div
                className="driver-bar-fill"
                style={{
                  width: `${barWidth}%`,
                  background: color,
                  opacity: 0.85,
                }}
              />
            </div>

            <span
              className="driver-bar-value"
              style={{ color: row.delta > 0.001 ? color : 'var(--clr-text-muted)' }}
            >
              ΔR = {fmt(row.delta, 4)}
            </span>
          </div>
        );
      })}

      {rows[0] && (
        <div style={{
          marginTop: 'var(--sp-2)',
          fontSize: '0.72rem',
          color: 'var(--clr-text-secondary)',
          fontStyle: 'italic',
        }}>
          Primary driver: <strong style={{ color: driverColor(rows[0].key) }}>{rows[0].name}</strong>
          {' '}(non-causal attribution)
        </div>
      )}
    </div>
  );
}
