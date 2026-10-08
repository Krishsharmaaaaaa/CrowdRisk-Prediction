import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, ReferenceLine, Area, AreaChart
} from 'recharts';
import { getRiskLevel, getRiskClass } from '../utils';

const CustomTooltip = ({ active, payload, label }) => {
  if (!active || !payload?.length) return null;
  const risk = payload[0]?.value;
  const level = getRiskLevel(risk);
  const lc = getRiskClass(level);
  return (
    <div style={{
      background: 'rgba(10,15,30,0.95)',
      border: '1px solid var(--clr-border-bright)',
      borderRadius: 'var(--r-sm)',
      padding: '10px 14px',
      fontSize: '0.78rem',
    }}>
      <div style={{ color: 'var(--clr-text-secondary)', marginBottom: 4 }}>
        t = {label?.toFixed ? label.toFixed(2) : label}s
      </div>
      <div>
        <span className={`badge badge-${lc}`} style={{ marginRight: 6 }}>{level}</span>
        <span style={{ fontFamily: 'var(--font-mono)', fontWeight: 700 }}>
          {risk?.toFixed(4)}
        </span>
      </div>
    </div>
  );
};

/**
 * Temporal risk timeline chart using Recharts AreaChart.
 */
export default function RiskTimeline({ timeline = [] }) {
  const data = timeline.slice(-600).map((row) => ({
    t: row.timestamp ?? row.frame_idx,
    risk: row.risk_score ?? row.model_risk ?? 0,
  }));

  if (!data.length) {
    return (
      <div style={{
        height: 200,
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        color: 'var(--clr-text-muted)',
        fontSize: '0.875rem',
      }}>
        No timeline data available
      </div>
    );
  }

  return (
    <div className="chart-wrapper" id="risk-timeline-chart">
      <ResponsiveContainer width="100%" height="100%">
        <AreaChart data={data} margin={{ top: 8, right: 8, bottom: 0, left: -16 }}>
          <defs>
            <linearGradient id="riskGrad" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor="#3b82f6" stopOpacity={0.35} />
              <stop offset="95%" stopColor="#3b82f6" stopOpacity={0.02} />
            </linearGradient>
          </defs>
          <CartesianGrid
            strokeDasharray="3 3"
            stroke="rgba(255,255,255,0.05)"
            vertical={false}
          />
          <XAxis
            dataKey="t"
            tick={{ fill: 'var(--clr-text-muted)', fontSize: 10, fontFamily: 'var(--font-mono)' }}
            tickLine={false}
            axisLine={false}
            tickFormatter={(v) => typeof v === 'number' ? `${v.toFixed(0)}s` : v}
          />
          <YAxis
            domain={[0, 1]}
            tick={{ fill: 'var(--clr-text-muted)', fontSize: 10, fontFamily: 'var(--font-mono)' }}
            tickLine={false}
            axisLine={false}
            tickFormatter={(v) => v.toFixed(1)}
          />
          <Tooltip content={<CustomTooltip />} />

          {/* Threshold lines */}
          <ReferenceLine y={0.60} stroke="rgba(239,68,68,0.4)" strokeDasharray="4 3"
            label={{ value: 'R_safe', position: 'right', fill: 'rgba(239,68,68,0.6)', fontSize: 9 }} />
          <ReferenceLine y={0.30} stroke="rgba(245,158,11,0.3)" strokeDasharray="4 3"
            label={{ value: 'moderate', position: 'right', fill: 'rgba(245,158,11,0.5)', fontSize: 9 }} />

          <Area
            type="monotone"
            dataKey="risk"
            stroke="#3b82f6"
            strokeWidth={2}
            fill="url(#riskGrad)"
            dot={false}
            activeDot={{ r: 4, fill: '#60a5fa', strokeWidth: 0 }}
          />
        </AreaChart>
      </ResponsiveContainer>
    </div>
  );
}
