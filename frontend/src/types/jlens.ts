/**
 * J-space lens wire format — the UI mirror of backend/src/schemas/jlens.py.
 *
 * THE WIRE FORMAT IS NOT OURS TO DESIGN (BR-029, PADR IDL-45). These shapes
 * mirror Neuronpedia's lens stream exactly, so the readout panel renders a
 * miStudio stream and a Neuronpedia stream with no adaptation layer. Adding a
 * miStudio-shaped field here would silently break that property — the panel
 * would keep working against our server and fail against theirs.
 *
 *   meta  = { model, types, layers_by_type, top_n, prompt_len }
 *   token = { position, token, id, is_generated, results: slice[] }
 *   slice = { type, top_tokens[layer][k], top_probs[layer][k] }
 *
 * `top_tokens` entries are DECODED STRINGS, not ids — the backend enforces
 * this because ids type-check against a looser schema and render as
 * unreadable cells.
 */

/**
 * Lens types the stream can CARRY.
 *
 * DIFF is deliberately absent: it is a client-side rendering mode over two
 * transported slices, never a transported type. Adding it here would invite a
 * request for a type the server cannot emit.
 */
export type LensType = 'JACOBIAN_LENS' | 'LOGIT_LENS';

/** What the mode tabs offer. DIFF exists only at this layer. */
export type LensMode = LensType | 'DIFF';

/**
 * Which computations are meaningful at one layer.
 *
 * Layer kind is PER-LAYER state, not a model property: a hybrid model
 * interleaves convolutional and attention layers, so "freeze Q/K" is undefined
 * on some of them. Inapplicable is `null` (absent), never `false` — a `false`
 * gets averaged by a consumer and silently understates (BR-032).
 */
export interface LayerApplicability {
  layer: number;
  has_attention: boolean;
  frozen_qk_applicable: boolean | null;
  broadcast_metrics_applicable: boolean | null;
}

/**
 * One lens type's readout for one token position, across layers.
 *
 * `top_tokens[layerIdx][k]` is indexed by POSITION IN `meta.layers_by_type[type]`,
 * not by the model's absolute layer number. Indexing it with an absolute layer
 * number reads the wrong row wherever the two differ, and produces a plausible
 * grid rather than an error.
 */
export interface LensTypeSlice {
  type: LensType;
  top_tokens: string[][];
  top_probs: number[][];
}

export interface LensMetaMessage {
  kind: 'meta';
  model: string;
  types: LensType[];
  /** Absolute layer indices per lens type. DRIVES THE LAYER AXIS — never assume
   *  a count or a spacing; models here range from 16 to 26+ layers. */
  layers_by_type: Record<string, number[]>;
  top_n: number;
  prompt_len: number;
  layer_applicability?: LayerApplicability[] | null;
}

export interface LensTokenMessage {
  kind: 'token';
  position: number;
  token: string;
  id: number;
  is_generated: boolean;
  results: LensTypeSlice[];
}

export interface ReadoutRequest {
  model_id: string;
  prompt: string;
  types?: LensType[];
  layers?: number[] | null;
  top_n?: number;
  /** Required by the server when `types` includes JACOBIAN_LENS. The logit lens
   *  needs no artifact (BR-005). */
  artifact_id?: string | null;
}

/** Non-streaming envelope. CONTAINS a meta message rather than being one. */
export interface ReadoutResponse {
  meta: LensMetaMessage;
  tokens: LensTokenMessage[];
}

/**
 * Sensory / workspace / motor boundaries for one model.
 *
 * THERE IS DELIBERATELY NO DEFAULT VALUE AND NO CONSTANT ANYWHERE IN THIS
 * FEATURE. The reference implementation's L40/L90 are the source paper's
 * Sonnet-4.5 figures; BR-002 forbids porting them to another model and requires
 * the product make porting impossible by construction. Bands render only from a
 * report computed for the selected model, and are absent otherwise — a
 * fallback object would be those figures under another name.
 */
export interface BandReport {
  model: string;
  workspace_start: number;
  motor_start: number;
  /** How the boundaries were derived, surfaced next to the shading. */
  derivation: string;
}

/** Provenance behind the current readout (BR-007). */
export interface ReadoutProvenance {
  /** Absent for the logit lens, which involves no artifact at all. */
  artifact_id: string | null;
  target_layer?: string;
  attention_gradients?: string;
  target_position_scope?: string;
  aggregation?: string;
  corpus?: string;
  n_prompts?: number;
  seq_len?: number;
  dtype?: string;
}
