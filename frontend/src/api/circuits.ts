/** Circuits API client (Feature 018). */

import type { Circuit } from '../types/circuits';

const BASE = '/api/v1/circuits';

async function json<T>(res: Response): Promise<T> {
  if (!res.ok) {
    const detail = await res.json().catch(() => ({}));
    throw new Error(detail.detail || `HTTP ${res.status}`);
  }
  return res.json();
}

export const circuitsApi = {
  list: (params?: { promoted?: boolean; min_rung?: number }) => {
    const qs = new URLSearchParams();
    if (params?.promoted !== undefined) qs.set('promoted', String(params.promoted));
    if (params?.min_rung !== undefined) qs.set('min_rung', String(params.min_rung));
    const q = qs.toString();
    return fetch(`${BASE}${q ? `?${q}` : ''}`).then((r) =>
      json<{ circuits: Circuit[]; total: number }>(r));
  },
  get: (id: string) => fetch(`${BASE}/${id}`).then((r) => json<Circuit>(r)),
  promote: (id: string) =>
    fetch(`${BASE}/${id}/promote`, { method: 'POST' }).then((r) => json<Circuit>(r)),
  remove: (id: string) =>
    fetch(`${BASE}/${id}`, { method: 'DELETE' }).then((r) => json<{ deleted: string }>(r)),
  exportUrl: (id: string) => `${BASE}/${id}/export`,
  exportSlices: (id: string) =>
    fetch(`${BASE}/${id}/export-slices`, { method: 'POST' }).then((r) =>
      json<{ parent_rung: number; parent_rung_language: string; slices: unknown[] }>(r)),
};
