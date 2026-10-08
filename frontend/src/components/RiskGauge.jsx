import { ringStroke, getRiskLevel, getRiskClass, clamp } from '../utils';

/**
 * Animated SVG ring gauge for displaying model risk score (0–1).
 */
export default function RiskGauge({ score = 0, size = 180 }) {
  const r = (size - 20) / 2;
  const cx = size / 2;
  const cy = size / 2;
  const circumference = 2 * Math.PI * r;
  const safeFraction = clamp(score, 0, 1);
  const offset = circumference * (1 - safeFraction);
  const level = getRiskLevel(score);
  const levelClass = getRiskClass(level);
  const stroke = ringStroke(score);

  return (
    <div className="risk-ring-container" id="risk-gauge">
      <div style={{ position: 'relative', width: size, height: size }}>
        <svg
          width={size}
          height={size}
          className="risk-ring-svg"
          viewBox={`0 0 ${size} ${size}`}
          aria-label={`Risk gauge: ${(score * 100).toFixed(1)}%`}
        >
          {/* Background ring */}
          <circle
            className="risk-ring-bg"
            cx={cx} cy={cy} r={r}
            strokeWidth={10}
          />
          {/* Filled arc */}
          <circle
            className="risk-ring-fill"
            cx={cx} cy={cy} r={r}
            strokeWidth={10}
            stroke={stroke}
            strokeDasharray={circumference}
            strokeDashoffset={offset}
          />
        </svg>

        {/* Centre label */}
        <div style={{
          position: 'absolute',
          inset: 0,
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          justifyContent: 'center',
          gap: 2,
        }}>
          <span
            className="risk-value-text"
            style={{ color: stroke, fontSize: size > 140 ? '2rem' : '1.4rem' }}
          >
            {(score * 100).toFixed(1)}
          </span>
          <span style={{
            fontFamily: 'var(--font-mono)',
            fontSize: '0.6rem',
            fontWeight: 600,
            letterSpacing: '0.12em',
            textTransform: 'uppercase',
            color: stroke,
            opacity: 0.85,
          }}>
            {level}
          </span>
        </div>
      </div>

      <span className={`badge badge-${levelClass}`}>
        MODEL RISK INDEX
      </span>
    </div>
  );
}
