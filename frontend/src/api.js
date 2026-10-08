/**
 * CrowdRisk API Client
 * Connects Vercel frontend to the Render FastAPI backend.
 */

const getApiBase = () => {
  const envUrl = import.meta.env.VITE_API_BASE_URL;
  if (envUrl && typeof envUrl === 'string' && envUrl.trim().length > 0) {
    return envUrl.trim().replace(/\/+$/, '');
  }

  const isClient = typeof window !== 'undefined';
  const isLocal = isClient && (
    window.location.hostname === 'localhost' ||
    window.location.hostname === '127.0.0.1'
  );

  // Production fallback to deployed Render backend
  if (!isLocal || import.meta.env.PROD) {
    return 'https://crowdrisk-api.onrender.com';
  }

  return 'http://localhost:8000';
};

export const API_BASE = getApiBase();

async function fetchJSON(path, options = {}, retries = 2) {
  const cleanPath = path.startsWith('/') ? path : `/${path}`;
  const url = `${API_BASE}${cleanPath}`;

  for (let attempt = 0; attempt <= retries; attempt++) {
    try {
      const res = await fetch(url, options);

      if (!res.ok) {
        // Retry transient 502/503 during Render cold start wakeups
        if ((res.status === 502 || res.status === 503) && attempt < retries) {
          await new Promise(r => setTimeout(r, 1500 * (attempt + 1)));
          continue;
        }
        const err = await res.json().catch(() => ({}));
        const message = err.detail || `Request failed with HTTP ${res.status}`;
        console.error(`[API HTTP Error] ${options.method || 'GET'} ${cleanPath} -> ${res.status}:`, message);
        throw new Error(message);
      }

      return await res.json();
    } catch (err) {
      // Retry GET requests on network drop or wake-up timeout
      const isGet = !options.method || options.method.toUpperCase() === 'GET';
      if (attempt < retries && isGet) {
        await new Promise(r => setTimeout(r, 1500 * (attempt + 1)));
        continue;
      }
      console.error(`[API Network Error] ${options.method || 'GET'} ${cleanPath}:`, err);
      throw new Error('Unable to connect to the Crowd Risk API. Please check that the backend is available.');
    }
  }
}

export const api = {
  /** API Base URL */
  baseUrl: API_BASE,

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
    return fetchJSON('/api/analyze', { method: 'POST', body: form }, 0);
  },

  /** Poll analysis job status */
  getAnalysis: (id) => fetchJSON(`/api/analysis/${id}`, {}, 1),

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
