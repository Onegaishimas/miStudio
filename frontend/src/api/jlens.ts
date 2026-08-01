/**
 * J-Lens readout API client (Feature 023, doc chain 022 substrate).
 *
 * One call. The response is the upstream wire format verbatim — this module
 * deliberately performs NO reshaping, because an adaptation layer here is
 * exactly what PADR IDL-45 forbids: it would let the panel drift into a
 * miStudio-only shape while still appearing to conform.
 */

import { fetchAPI } from './client';
import type {
  JLensArtifactSummary,
  JLensFitAccepted,
  JLensFitRequest,
  JLensValidationResponse,
  ReadoutAccepted,
  ReadoutRequest,
  ReadoutResult,
} from '../types/jlens';

export const jlensApi = {
  /**
   * Request a position x layer readout.
   *
   * `types` defaults server-side to LOGIT_LENS, which needs no artifact
   * (BR-005). Requesting JACOBIAN_LENS without `artifact_id` is refused by the
   * server rather than silently served as logit data under a Jacobian label
   * (BR-019).
   */
  readout: (request: ReadoutRequest) =>
    fetchAPI<ReadoutAccepted>('/jlens/readout', {
      method: 'POST',
      body: JSON.stringify(request),
    }),

  /** Poll a queued readout. Null `readout` until `status` is SUCCESS. */
  readoutResult: (taskId: string) =>
    fetchAPI<ReadoutResult>(`/jlens/readout/${encodeURIComponent(taskId)}`),

  /** Artifacts present in the mounted registry. Presence, not validity. */
  listArtifacts: () => fetchAPI<JLensArtifactSummary[]>('/jlens/artifacts'),

  /**
   * Run the validation suite. The model's dimensions are required because the
   * envelope bound must come from the model the artifact was fitted for — a
   * bound derived from the wrong model passes while missing a real
   * materialisation.
   */
  validateArtifact: (
    slug: string,
    dims: { d_model: number; n_layers: number; n_vocab: number }
  ) =>
    fetchAPI<JLensValidationResponse>(
      `/jlens/artifacts/${encodeURIComponent(slug)}/validate` +
        `?d_model=${dims.d_model}&n_layers=${dims.n_layers}&n_vocab=${dims.n_vocab}`,
      { method: 'POST' }
    ),

  /** Queue a fit. GPU-bound and long-running; poll the task id. */
  fit: (request: JLensFitRequest) =>
    fetchAPI<JLensFitAccepted>('/jlens/fit', {
      method: 'POST',
      body: JSON.stringify(request),
    }),
};
