/**
 * CrowdRisk API Client
 * Connects Vercel frontend to the Render FastAPI backend.
 */

const API_BASE = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';

async function fetchJSON(path, options = {}) {
  const res = await fetch(`${API_BASE}${path}`, options);
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || `HTTP ${res.status}`);
  }
  return res.json();
}

export const api = {
  /** Health ping */
  health: () => fetchJSON('/api/health'),

  /** List precomputed demo samples */
  listDemos: () => fetchJSON('/api/demo/samples'),

  /** Get a demo sample by ID */
  getDemoSample: (sampleId) => fetchJSON(`/api/demo/${sampleId}`),

  /** Submit a video file for analysis */
  submitAnalysis: async (file) => {
    const form = new FormData();
    form.append('file', file);
    return fetchJSON('/api/analyze', { method: 'POST', body: form });
  },

  /** Poll analysis job status */
  getAnalysis: (id) => fetchJSON(`/api/analysis/${id}`),

  /** Get risk timeline rows */
  getTimeline: (id) => fetchJSON(`/api/analysis/${id}/timeline`),

  /** Get 4x4 spatial grid data */
  getSpatial: (id) => fetchJSON(`/api/analysis/${id}/spatial`),

  /** Get CRDA counterfactual explanations */
  getCRDA: (id) => fetchJSON(`/api/analysis/${id}/crda`),

  /** URL to stream the annotated video in-browser */
  videoUrl: (id) => `${API_BASE}/api/analysis/${id}/video`,
};

export default api;
