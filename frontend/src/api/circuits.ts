/**
 * Circuits API client (Feature 018) — through the shared fetchAPI client
 * (auth header, base URL, typed APIError; review R1 fix #1).
 *
 * List rows are SLIM summaries; full members/edges arrive on detail fetch.
 */

import { fetchAPI, buildQueryString, API_V1_BASE } from './client';
import type { Circuit, CircuitSummary } from '../types/circuits';

export const circuitsApi = {
  list: (params?: { promoted?: boolean; min_rung?: number; edge_type?: string;
                    limit?: number; offset?: number }) => {
    // buildQueryString returns WITHOUT a leading '?' (R2 B2 — a bare concat
    // produced /circuitspromoted=true and 404'd the first filtered call).
    const qs = buildQueryString(params ?? {});
    return fetchAPI<{ circuits: CircuitSummary[]; total: number }>(
      `/circuits${qs ? `?${qs}` : ''}`);
  },

  get: (id: string) => fetchAPI<Circuit>(`/circuits/${id}`),

  update: (id: string, body: { name?: string; narrative?: string }) =>
    fetchAPI<Circuit>(`/circuits/${id}`, {
      method: 'PATCH',
      body: JSON.stringify(body),
    }),

  setPromoted: (id: string, promoted: boolean) =>
    fetchAPI<Circuit>(`/circuits/${id}/promote`, {
      method: 'POST',
      body: JSON.stringify({ promoted }),
    }),

  remove: (id: string) =>
    fetchAPI<{ deleted: string }>(`/circuits/${id}`, { method: 'DELETE' }),

  /** Browser-navigable export URL (Content-Disposition download). */
  exportUrl: (id: string) => `${API_V1_BASE}/circuits/${id}/export`,

  exportSlices: (id: string) =>
    fetchAPI<{ parent_rung: number; parent_rung_language: string; slices: unknown[] }>(
      `/circuits/${id}/export-slices`, { method: 'POST' }),

  importDefinition: (definition: unknown) =>
    fetchAPI<Circuit>('/circuits/import', {
      method: 'POST',
      body: JSON.stringify(definition),
    }),
};
