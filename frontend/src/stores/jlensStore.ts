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
import { devtools, persist } from 'zustand/middleware';
import { jlensApi } from '../api/jlens';
import type {
  BandReport,
  ReadoutResponse,
  JLensArtifactSummary,
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

  /** Artifacts present in the mounted registry. Presence, NOT validity. */
  artifacts: JLensArtifactSummary[];
  /** Repo id of the selected model, used to derive its artifact slug. */
  modelRepoId: string;

  isLoading: boolean;
  /** What the queued readout is doing right now, e.g. 'loading_model'. */
  stage: string | null;
  error: string | null;

  setModelId: (id: string, repoId?: string) => void;
  fetchArtifacts: () => Promise<void>;
  setPrompt: (p: string) => void;
  setLensMode: (m: LensMode) => void;
  setSelPos: (p: number) => void;
  setSelLayerIdx: (i: number) => void;
  setHover: (h: { pos: number; layerIdx: number } | null) => void;
  togglePin: (token: string) => void;
  clearPins: () => void;
  fetchReadout: () => Promise<void>;
  reset: () => void;
  /**
   * Forget the persisted setup: model, prompt, lens mode and pins.
   *
   * Distinct from `reset`, which also drops the readout. This is the control
   * for "stop restoring my last session", and the readout goes with it because
   * a readout kept beside a cleared prompt describes text no longer on screen.
   */
  clearConfig: () => void;
}

const INITIAL = {
  modelId: '',
  modelRepoId: '',
  artifacts: [] as JLensArtifactSummary[],
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
  stage: null,
  error: null,
};

export const useJLensStore = create<JLensState>()(
  devtools(
    persist(
      (set, get) => ({
      ...INITIAL,

      setModelId: (modelId, repoId) =>
        set((state) =>
          state.modelId === modelId
            ? { modelId, modelRepoId: repoId ?? state.modelRepoId }
            : {
                modelRepoId: repoId ?? '',
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
      fetchArtifacts: async () => {
        try {
          set({ artifacts: await jlensApi.listArtifacts() });
        } catch {
          // A registry that cannot be listed is not an error the readout path
          // needs to surface: the logit lens works regardless, and the Jacobian
          // tab already states why it is unavailable.
          set({ artifacts: [] });
        }
      },

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
        set({ isLoading: true, error: null, stage: 'queued' });

        try {
          // Ask for the Jacobian lens ONLY when this model has an artifact.
          // Requesting it otherwise is refused by the schema (it needs an
          // artifact_id) and surfaces as a 422 the user cannot act on.
          const slug = artifactSlugFor(get().modelRepoId);
          const artifact = slug
            ? get().artifacts.find((a) => a.slug === slug)
            : undefined;

          const accepted = await jlensApi.readout({
            model_id: modelId,
            prompt,
            types: artifact ? ['JACOBIAN_LENS', 'LOGIT_LENS'] : ['LOGIT_LENS'],
            ...(artifact ? { artifact_id: artifact.slug } : {}),
          });

          // POLL. The readout is queued because it needs the whole model
          // resident, and a synchronous request 502'd at the ingress on a real
          // model. `stage` is surfaced so a minute-long first load reads as
          // "loading the model", not as a hung button.
          const response = await pollReadout(accepted.task_id, (stage) => {
            if (seq === requestSeq) set({ stage });
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
            // Clamp against the axis the panel will actually READ: two lens
            // types may carry different layer counts, and clamping to the
            // wrong one leaves an out-of-range row index.
            const axis = axisFor(response.meta, readTypeFor(lensMode));

            return {
              meta: response.meta,
              tokens: response.tokens,
              // Logit lens involves no artifact at all — say so rather than
              // leaving the provenance strip blank (BR-007).
              provenance: { artifact_id: artifact ? artifact.slug : null },
              isLoading: false,
              stage: null,
              error: null,
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
            stage: null,
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

      // Same bump as `reset`, for the same reason: an in-flight readout that
      // lands after a clear would repopulate the store the user just emptied,
      // and the prompt it describes is gone from the screen.
      clearConfig: () => {
        requestSeq += 1;
        set({ ...INITIAL });
      },
      }),
      {
        name: 'miStudio-jlens',
        /**
         * SETUP PERSISTS; RESULTS DO NOT.
         *
         * `meta`, `tokens` and `provenance` are deliberately absent. A readout
         * restored from a previous session describes a prompt the user may
         * have edited since, and a grid full of stale content is
         * indistinguishable from one describing what is currently on screen —
         * the same confusion the fixture ban exists to prevent.
         *
         * `artifacts` is absent for a different reason: it is the mounted
         * registry's contents, which change on the server. A restored copy
         * would decide which lenses to offer from a list that may no longer be
         * true.
         */
        partialize: (state) => ({
          modelId: state.modelId,
          modelRepoId: state.modelRepoId,
          prompt: state.prompt,
          lensMode: state.lensMode,
          pinned: state.pinned,
        }),
      }
    ),
    { name: 'jlens-store' }
  )
);

/**
 * The artifact slug a repo id would produce — mirrors the server's `slug_for`.
 *
 * The slug is how weight identity is checked: a base model and its
 * instruction-tuned variant differ by a suffix and produce different slugs, so
 * a lens fitted for one is never matched to the other. Deriving it here only
 * decides which lens to REQUEST; the server re-derives and refuses a mismatch,
 * so a drift in this function costs a 409 rather than a wrong readout.
 */
export function artifactSlugFor(repoId: string): string {
  if (!repoId) return '';
  const tail = repoId.split('/').pop() ?? '';
  return tail
    .toLowerCase()
    .replace(/[^a-z0-9._-]+/g, '-')
    .replace(/^-+|-+$/g, '');
}

/**
 * Poll a queued readout until it succeeds or fails.
 *
 * A terminal FAILURE is raised with its reason rather than resolving to an
 * empty readout — an empty readout is indistinguishable from a real one with
 * no content, which is the failure this whole feature is built to avoid.
 */
export async function pollReadout(
  taskId: string,
  onStage: (stage: string | null) => void,
  intervalMs = 1500,
  timeoutMs = 600_000
): Promise<ReadoutResponse> {
  const deadline = Date.now() + timeoutMs;

  for (;;) {
    const result = await jlensApi.readoutResult(taskId);

    if (result.status === 'SUCCESS' && result.readout) {
      return result.readout;
    }
    if (result.status === 'FAILURE') {
      throw new Error(result.error || 'Readout failed.');
    }
    if (Date.now() > deadline) {
      // Reported as a timeout on OUR side, naming the task, so the job is
      // still findable rather than silently abandoned.
      throw new Error(
        `Readout ${taskId} did not finish within ${Math.round(timeoutMs / 1000)}s. ` +
          'It may still be running — the first readout for a model loads the ' +
          'whole model.'
      );
    }
    onStage(result.stage ?? null);
    await new Promise((r) => setTimeout(r, intervalMs));
  }
}

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
  mode: LensMode,
  hasArtifact = false
): ModeAvailability {
  if (!meta) {
    // BEFORE the first readout we know one thing for certain: the logit lens
    // needs no artifact and works on any loaded model (BR-005). Reporting it
    // as unavailable alongside the two that genuinely require one told the
    // user nothing works, when the default path always does.
    if (mode === 'LOGIT_LENS') {
      return { enabled: true, reason: null };
    }
    // AN ARTIFACT IS ENOUGH TO SELECT THE LENS. `fetchReadout` requests BOTH
    // lens types whenever this model has an artifact, so a Jacobian selection
    // made before the first readout is honoured by the very next one — the tab
    // was disabled on the absence of a stream that the click itself produces.
    //
    // This is not a claim that the lens WORKED: if the readout comes back
    // without JACOBIAN_LENS the branch below disables the tab again and
    // `fetchReadout` falls the selection back to logit. Enabling here promises
    // the request, not the result.
    if (hasArtifact) {
      return { enabled: true, reason: null };
    }
    return {
      enabled: false,
      reason:
        mode === 'JACOBIAN_LENS'
          ? 'Needs a validated J-lens artifact for this model. Fit one to enable it.'
          : 'Diff compares two lenses; it needs a fitted J-lens artifact.',
    };
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
