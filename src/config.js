/**
 * Admin UI mode (build-time):
 * - training-only: local npm start — training pipeline only
 * - full: Railway/production — users + dashboard + API keys (no training)
 */
export const ADMIN_MODE = process.env.REACT_APP_ADMIN_MODE || 'training-only';

export const isLocalTrainingOnlyAdmin = ADMIN_MODE === 'training-only';

export const isFullAdminConsole = ADMIN_MODE === 'full';

function resolveApiBaseUrl() {
  const explicit = process.env.REACT_APP_API_URL;
  if (explicit !== undefined && explicit !== '') {
    return explicit;
  }
  if (typeof window !== 'undefined') {
    const host = window.location.hostname;
    if (host !== 'localhost' && host !== '127.0.0.1') {
      return '';
    }
  }
  return 'http://localhost:8000';
}

export const API_BASE_URL = resolveApiBaseUrl();

/** Public Railway app URL (update if your domain differs). */
export const RAILWAY_APP_URL =
  process.env.REACT_APP_RAILWAY_URL || 'https://aistockgpt-production.up.railway.app';
