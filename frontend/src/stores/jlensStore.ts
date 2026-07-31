/**
 * J-Lens panel store (Feature 023).
 *
 * Holds the readout, the selection, and the pinned tokens. Three properties are
 * load-bearing and each has a test pinning it:
 *
 *  1. NO FIXTURE SEED. The reference implementation ships `FIXTURES` /
 *     `buildFixture` / `scoreAt`; none of it is here or anywhere under src/.
 *     Synthetic readouts are indistinguishable from real ones once rendered.
 *  2. A REFETCH NEVER BLANKS THE READOUT. `isLoading` is set without clearing
 *     `meta`/`tokens`, so a background refresh cannot unmount the grid and drop
 *     the user's pins — the house regression fixed in ExtractionsPanel.
 *  3. THE LAYER AXIS COMES FROM `meta.layers_by_type`. Selection is stored as
 *     an INDEX into that axis, and clamped whenever a new readout arrives with
 *     a different layer count. Storing an absolute layer number would read the
 *     wrong slice row on any model whose axis is not 0..n-1.
 */

import { create } from 'zustand';
import { devtools } from 'zustand/middleware';
import { jlensApi } from '../api/jlens';
import type {
  BandReport,
  LensMetaMessage,
  LensMode,
  LensTokenMessage,
  LensType,
  LensTypeSlice,
  ReadoutProvenance,
} from '../types/jlens';

/** Mirrors the server cap; the request is rejected above this anyway. */
export const MAX_PINNED = 6;

/**
 * Matches `ReadoutRequest.prompt`'s `max_length` in backend/src/schemas/jlens.py.
 *
 * The bound exists there because readout cost is O(positions x layers x top_n)
 * and every position holds a d_model residual. Duplicating it here turns a 422
 * the user cannot act on into a field they can see filling up.
 */
export const MAX_PROMPT_CHARS = 8000;

interface JLensState {
  modelId: string;
  prompt: string;

  meta: LensMetaMessage | null;
  tokens: LensTokenMessage[];

  /**
   * Bands are DATA, not a default. Null means no band report exists for this
   * model, and the grid renders unshaded with a stated reason (BR-002).
   * Nothing in this feature may substitute a constant here.
   */
  bandReport: BandReport | null;
  provenance: ReadoutProvenance | null;

  lensMode: LensMode;
  selPos: number;
  selLayerIdx: number;
  pinned: string[];
  hover: { pos: number; layerIdx: number } | null;

  isLoading: boolean;
  error: string | null;

  setModelId: (id: string) => void;
  setPrompt: (p: string) => void;
  setLensMode: (m: LensMode) => void;
  setSelPos: (p: number) => void;
  setSelLayerIdx: (i: number) => void;
  setHover: (h: { pos: number; layerIdx: number } | null) => void;
  togglePin: (token: string) => void;
  clearPins: () => void;
  fetchReadout: () => Promise<void>;
  reset: () => void;
}

const INITIAL = {
  modelId: '',
  prompt: '',
  meta: null,
  tokens: [],
  bandReport: null,
  provenance: null,
  lensMode: 'LOGIT_LENS' as LensMode,
  selPos: 0,
  selLayerIdx: 0,
  pinned: [] as string[],
  hover: null,
  isLoading: false,
  error: null,
};

export const useJLensStore = create<JLensState>()(
  devtools(
    (set, get) => ({
      ...INITIAL,

      setModelId: (modelId) =>
        set((state) =>
          state.modelId === modelId
            ? { modelId }
            : {
                // Pins are token strings from ANOTHER model's vocabulary.
                // Carrying them across draws empty trajectory lines that look
                // like "this concept is absent" rather than "this pin does not
                // apply here".
                modelId,
                pinned: [],
                meta: null,
                tokens: [],
                provenance: null,
                hover: null,
              }
        ),
      setPrompt: (prompt) => set({ prompt }),
      setLensMode: (lensMode) => set({ lensMode }),
      setSelPos: (selPos) => set({ selPos }),
      setSelLayerIdx: (selLayerIdx) => set({ selLayerIdx }),
      setHover: (hover) => set({ hover }),

      togglePin: (token) =>
        set((state) => ({
          pinned: state.pinned.includes(token)
            ? state.pinned.filter((t) => t !== token)
            : [...state.pinned, token].slice(0, MAX_PINNED),
        })),

      clearPins: () => set({ pinned: [] }),

      fetchReadout: async () => {
        const { modelId, prompt } = get();
        if (!modelId || !prompt.trim()) {
          set({ error: 'Select a model and enter a prompt.' });
          return;
        }
        if (prompt.length > MAX_PROMPT_CHARS) {
          set({
            error: `Prompt is ${prompt.length} characters; the readout accepts at most ${MAX_PROMPT_CHARS}.`,
          });
          return;
        }

        // Deliberately does NOT clear meta/tokens: a refetch must leave the
        // current readout on screen so the grid does not unmount and drop pins.
        //
        // The sequence guard makes a SLOW EARLIER request unable to overwrite a
        // FAST LATER one. Without it, two rapid submits resolve in completion
        // order rather than request order, and the grid can settle showing the
        // prompt the user already replaced.
        const seq = ++requestSeq;
        set({ isLoading: true, error: null });

        try {
          // LOGIT_LENS only for now. A Jacobian request needs an artifact_id and
          // the server refuses it without one; asking for a type we cannot
          // supply an artifact for would surface as a 422 the user cannot act on.
          const response = await jlensApi.readout({
            model_id: modelId,
            prompt,
            types: ['LOGIT_LENS'],
          });

          if (seq !== requestSeq) return;

          set((state) => {
            // A mode the new readout cannot serve must not stay SELECTED. It
            // renders empty, which reads as "this lens found nothing" rather
            // than "this lens is not in this readout" — the disabled tab says
            // the latter, and the two must not contradict each other.
            const lensMode = modeAvailability(response.meta, state.lensMode).enabled
              ? state.lensMode
              : ('LOGIT_LENS' as const);
            // Clamp against the axis the panel will actually READ, not against
            // a fixed type: two lens types may carry different layer counts,
            // and clamping to the wrong one leaves an out-of-range row index.
            const axis = axisFor(response.meta, readTypeFor(lensMode));

            return {
              meta: response.meta,
              tokens: response.tokens,
              // Logit lens involves no artifact at all — say so rather than
              // leaving the provenance strip blank (BR-007).
              provenance: { artifact_id: null },
              isLoading: false,
              error: null,
              // Clamp both selections into the NEW readout's extents. A stale
              // index survives a model change and indexes a shorter array.
              selPos: clampPosition(state.selPos, response.tokens),
              selLayerIdx: clamp(state.selLayerIdx, axis.length),
              lensMode,
              hover: null,
            };
          });
        } catch (err) {
          if (seq !== requestSeq) return;
          set({
            isLoading: false,
            error: err instanceof Error ? err.message : 'Readout failed.',
          });
        }
      },

      // Bumping the sequence is part of the reset: without it an in-flight
      // request lands afterwards and repopulates a store the user just cleared.
      reset: () => {
        requestSeq += 1;
        set({ ...INITIAL });
      },
    }),
    { name: 'jlens-store' }
  )
);

/** Monotonic request counter backing the stale-response guard. */
let requestSeq = 0;

function clamp(value: number, length: number): number {
  if (length <= 0) return 0;
  return Math.min(Math.max(value, 0), length - 1);
}

/**
 * Keep the selected POSITION valid, by position value rather than array index.
 *
 * `position` is the token's index in the PROMPT, which the wire format does not
 * require to be the array index — a readout over a subset of positions is
 * legal. Falling back to array arithmetic works against our own server and
 * fails against a conformant one.
 */
function clampPosition(selPos: number, tokens: LensTokenMessage[]): number {
  if (tokens.length === 0) return 0;
  return tokens.some((t) => t.position === selPos)
    ? selPos
    : tokens[tokens.length - 1].position;
}

/** The token at a POSITION, not at an array index. */
export function tokenAtPosition(
  tokens: LensTokenMessage[],
  position: number
): LensTokenMessage | undefined {
  return tokens.find((t) => t.position === position);
}

/**
 * The layer axis for one lens type, straight from the stream.
 *
 * Every layer-indexed lookup in the panel goes through here. There is no
 * fallback axis: an absent type yields an empty axis, which renders as "no
 * data" rather than as a plausible grid over invented layers.
 */
export function axisFor(meta: LensMetaMessage | null, type: LensType): number[] {
  if (!meta) return [];
  return meta.layers_by_type[type] ?? [];
}

/** The slice of one lens type at one token, or undefined if not transported. */
export function sliceFor(
  token: LensTokenMessage | undefined,
  type: LensType
): LensTypeSlice | undefined {
  return token?.results.find((r) => r.type === type);
}

/**
 * Which lens type a mode READS.
 *
 * DIFF renders a comparison and reads the Jacobian slice as its primary; it is
 * only reachable when both types are present, which `modeAvailability` enforces.
 */
export function readTypeFor(mode: LensMode): LensType {
  return mode === 'DIFF' ? 'JACOBIAN_LENS' : mode;
}

export interface ModeAvailability {
  enabled: boolean;
  /** Why a mode is unavailable. A disabled control with no reason is a defect. */
  reason: string | null;
}

/**
 * Mode enablement derived from what the stream ACTUALLY CARRIES (BR-019).
 *
 * Not a feature flag and not a guess. Enabling Jacobian while the stream
 * carries only logit data would render logit readouts under a Jacobian label —
 * a lower rung in a higher rung's clothing, and invisible to the user.
 */
export function modeAvailability(
  meta: LensMetaMessage | null,
  mode: LensMode
): ModeAvailability {
  if (!meta) {
    return { enabled: false, reason: 'No readout yet.' };
  }
  const available = new Set<string>(meta.types);

  if (mode === 'LOGIT_LENS') {
    return available.has('LOGIT_LENS')
      ? { enabled: true, reason: null }
      : { enabled: false, reason: 'This readout carries no logit-lens data.' };
  }

  if (mode === 'JACOBIAN_LENS') {
    return available.has('JACOBIAN_LENS')
      ? { enabled: true, reason: null }
      : {
          enabled: false,
          reason:
            'No validated J-lens artifact for this model. Fit and validate one to enable this lens.',
        };
  }

  // DIFF compares two transported slices; it needs both.
  if (available.has('JACOBIAN_LENS') && available.has('LOGIT_LENS')) {
    return { enabled: true, reason: null };
  }
  return {
    enabled: false,
    reason: 'Diff compares two lenses; only one is present in this readout.',
  };
}

/** Rank of a token at one layer index, 1-based; null when outside the top-k. */
export function rankOf(
  slice: LensTypeSlice | undefined,
  layerIdx: number,
  token: string
): number | null {
  const row = slice?.top_tokens[layerIdx];
  if (!row) return null;
  const i = row.indexOf(token);
  return i === -1 ? null : i + 1;
}
