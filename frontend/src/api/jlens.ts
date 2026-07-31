/**
 * J-Lens readout API client (Feature 023, doc chain 022 substrate).
 *
 * One call. The response is the upstream wire format verbatim — this module
 * deliberately performs NO reshaping, because an adaptation layer here is
 * exactly what PADR IDL-45 forbids: it would let the panel drift into a
 * miStudio-only shape while still appearing to conform.
 */

import { fetchAPI } from './client';
import type { ReadoutRequest, ReadoutResponse } from '../types/jlens';

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
    fetchAPI<ReadoutResponse>('/jlens/readout', {
      method: 'POST',
      body: JSON.stringify(request),
    }),
};
