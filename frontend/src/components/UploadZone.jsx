import { useState, useRef } from 'react';
import { Upload, FileVideo, AlertCircle } from 'lucide-react';

const ALLOWED_EXTS = ['.mp4', '.avi', '.mov', '.mkv'];
const MAX_MB = 150;

/**
 * Video upload drag-and-drop zone.
 */
export default function UploadZone({ onUpload, loading = false }) {
  const [dragOver, setDragOver] = useState(false);
  const [error, setError] = useState(null);
  const inputRef = useRef(null);

  const validate = (file) => {
    if (!file) return 'No file selected.';
    const ext = '.' + file.name.split('.').pop().toLowerCase();
    if (!ALLOWED_EXTS.includes(ext)) {
      return `Unsupported format "${ext}". Allowed: ${ALLOWED_EXTS.join(', ')}`;
    }
    if (file.size > MAX_MB * 1024 * 1024) {
      return `File too large (max ${MAX_MB} MB).`;
    }
    return null;
  };

  const handleFile = (file) => {
    const err = validate(file);
    if (err) { setError(err); return; }
    setError(null);
    onUpload(file);
  };

  const onDrop = (e) => {
    e.preventDefault();
    setDragOver(false);
    const file = e.dataTransfer.files?.[0];
    handleFile(file);
  };

  const onInputChange = (e) => {
    const file = e.target.files?.[0];
    handleFile(file);
    e.target.value = '';
  };

  return (
    <div>
      <label
        id="upload-zone"
        htmlFor="video-upload-input"
        className={`upload-zone${dragOver ? ' drag-over' : ''}`}
        onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
        onDragLeave={() => setDragOver(false)}
        onDrop={onDrop}
        role="button"
        aria-label="Upload video for analysis"
        tabIndex={0}
        onKeyDown={(e) => e.key === 'Enter' && inputRef.current?.click()}
        style={{ cursor: loading ? 'not-allowed' : 'pointer', opacity: loading ? 0.6 : 1 }}
      >
        <input
          id="video-upload-input"
          ref={inputRef}
          type="file"
          accept={ALLOWED_EXTS.join(',')}
          onChange={onInputChange}
          disabled={loading}
        />

        <div className="upload-icon-wrapper">
          {loading
            ? <div className="live-dot processing" style={{ margin: 'auto' }} />
            : <Upload size={28} style={{ color: 'var(--clr-accent-bright)' }} />}
        </div>

        <div>
          <p style={{ fontWeight: 600, fontSize: '1rem', marginBottom: 4 }}>
            {loading ? 'Uploading & processing…' : 'Drop a crowd video here'}
          </p>
          <p style={{ fontSize: '0.85rem', color: 'var(--clr-text-secondary)' }}>
            {loading
              ? 'Your video is being analysed by the AI pipeline'
              : `or click to browse — ${ALLOWED_EXTS.join(', ')} · max ${MAX_MB} MB`}
          </p>
        </div>

        <div style={{ display: 'flex', gap: 'var(--sp-3)', flexWrap: 'wrap', justifyContent: 'center' }}>
          {ALLOWED_EXTS.map((e) => (
            <span key={e} className="badge badge-neutral">
              <FileVideo size={10} />{e}
            </span>
          ))}
        </div>
      </label>

      {error && (
        <div style={{
          display: 'flex',
          alignItems: 'center',
          gap: 'var(--sp-2)',
          marginTop: 'var(--sp-3)',
          padding: 'var(--sp-3) var(--sp-4)',
          background: 'var(--clr-critical-bg)',
          border: '1px solid rgba(239,68,68,0.3)',
          borderRadius: 'var(--r-sm)',
          fontSize: '0.8rem',
          color: 'var(--clr-critical)',
        }}>
          <AlertCircle size={14} />
          {error}
        </div>
      )}
    </div>
  );
}
