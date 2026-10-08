import { cellBgColor, cellTextColor, getRiskLevel } from '../utils';

/**
 * 4×4 Spatial Risk Heatmap Grid.
 * Renders cells C1–C16 with background colour driven by local risk score.
 */
export default function SpatialGrid({ cells = [], selectedCell = null, onSelectCell }) {
  // Build a lookup map: cellId -> data
  const cellMap = {};
  cells.forEach((c) => {
    if (c.cell_id) cellMap[c.cell_id] = c;
  });

  const rows = 4;
  const cols = 4;

  return (
    <div className="spatial-grid" id="spatial-grid" role="grid" aria-label="4x4 spatial risk grid">
      {Array.from({ length: rows * cols }, (_, idx) => {
        const cellId = `C${idx + 1}`;
        const data = cellMap[cellId];
        const risk = data?.initial_risk ?? null;
        const level = risk !== null ? getRiskLevel(risk) : null;
        const isElevated = risk !== null && risk >= 0.55;
        const isCritical = risk !== null && risk >= 0.75;
        const isSelected = selectedCell === cellId;

        return (
          <div
            key={cellId}
            id={`grid-cell-${cellId}`}
            className={`grid-cell${isElevated ? ' elevated' : ''}${isCritical ? ' critical' : ''}`}
            role="gridcell"
            aria-selected={isSelected}
            tabIndex={0}
            title={data
              ? `${cellId}: Risk ${risk?.toFixed(3)} — ${level}\nDriver: ${data.top_driver ?? '—'}`
              : `${cellId}: No data`}
            onClick={() => onSelectCell && onSelectCell(cellId, data)}
            onKeyDown={(e) => e.key === 'Enter' && onSelectCell && onSelectCell(cellId, data)}
            style={{
              background: isSelected
                ? 'rgba(59,130,246,0.2)'
                : cellBgColor(risk),
              borderColor: isSelected
                ? 'var(--clr-accent)'
                : undefined,
              outline: isSelected ? '2px solid var(--clr-accent)' : undefined,
              outlineOffset: '-1px',
            }}
          >
            <span className="grid-cell-id">{cellId}</span>
            {risk !== null ? (
              <span
                className="grid-cell-risk"
                style={{ color: cellTextColor(risk) }}
              >
                {risk.toFixed(2)}
              </span>
            ) : (
              <span className="grid-cell-risk" style={{ color: 'var(--clr-text-dim)' }}>—</span>
            )}
            {data?.top_driver && (
              <span style={{
                fontSize: '0.52rem',
                color: 'rgba(255,255,255,0.4)',
                lineHeight: 1,
                maxWidth: '100%',
                overflow: 'hidden',
                textOverflow: 'ellipsis',
                whiteSpace: 'nowrap',
              }}>
                {data.top_driver_key}
              </span>
            )}
          </div>
        );
      })}
    </div>
  );
}
