/**
 * Shared utilities for the CrowdRisk frontend.
 */

/** Map risk score (0-1) to threat level label */
export function getRiskLevel(score) {
  if (score >= 0.75) return 'CRITICAL';
  if (score >= 0.55) return 'HIGH';
  if (score >= 0.30) return 'MODERATE';
  return 'LOW';
}

/** Map threat level to CSS class suffix */
export function getRiskClass(level) {
  return level?.toLowerCase() ?? 'low';
}

/** Get CSS variable for driver key */
export function driverColor(key) {
  const map = {
    D: 'var(--clr-driver-D)',
    O: 'var(--clr-driver-O)',
    B: 'var(--clr-driver-B)',
    K: 'var(--clr-driver-K)',
  };
  return map[key] ?? '#94a3b8';
}

/** Get full driver name from key */
export function driverName(key) {
  const map = { D: 'Density', O: 'Disorder', B: 'Bottleneck', K: 'Kinematics' };
  return map[key] ?? key;
}

/** Format a number as a fixed-decimal string */
export function fmt(n, decimals = 4) {
  if (n === null || n === undefined || isNaN(n)) return '—';
  return Number(n).toFixed(decimals);
}

/** Format bytes to human readable */
export function formatBytes(bytes) {
  if (!bytes) return '0 B';
  const k = 1024;
  const sizes = ['B', 'KB', 'MB', 'GB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return `${(bytes / Math.pow(k, i)).toFixed(1)} ${sizes[i]}`;
}

/** Get grid cell background colour from risk score */
export function cellBgColor(risk) {
  if (risk === null || risk === undefined) return 'rgba(255,255,255,0.04)';
  if (risk >= 0.75) return 'rgba(239, 68, 68, 0.25)';
  if (risk >= 0.55) return 'rgba(249, 115, 22, 0.18)';
  if (risk >= 0.30) return 'rgba(245, 158, 11, 0.12)';
  return 'rgba(34, 197, 94, 0.07)';
}

/** Get grid cell text colour from risk score */
export function cellTextColor(risk) {
  if (risk === null || risk === undefined) return 'var(--clr-text-muted)';
  if (risk >= 0.75) return 'var(--clr-critical)';
  if (risk >= 0.55) return 'var(--clr-high)';
  if (risk >= 0.30) return 'var(--clr-moderate)';
  return 'var(--clr-low)';
}

/** Get ring stroke colour from risk score */
export function ringStroke(score) {
  if (score >= 0.75) return '#ef4444';
  if (score >= 0.55) return '#f97316';
  if (score >= 0.30) return '#f59e0b';
  return '#22c55e';
}

/** Clamp a number between min and max */
export function clamp(n, min, max) {
  return Math.min(Math.max(n, min), max);
}
