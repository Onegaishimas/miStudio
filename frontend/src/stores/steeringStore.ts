/**
 * Zustand store for Steering operations.
 *
 * This store manages the global state for feature steering, including:
 * - Selected SAE for steering
 * - Selected features (up to 4) with colors
 * - Generation parameters
 * - Comparison requests and responses
 * - Experiment management (save/load)
 * - Real-time progress updates
 *
 * Connects to REAL backend API at /api/v1/steering
 */

import { create } from 'zustand';
import { devtools, persist } from 'zustand/middleware';
import {
  SelectedFeature,
  GenerationParams,
  AdvancedGenerationParams,
  DEFAULT_GENERATION_PARAMS,
  SteeringComparisonRequest,
  SteeringComparisonResponse,
  SteeringStrengthSweepRequest,
  StrengthSweepResponse,
  SteeringExperiment,
  SteeringProgressUpdate,
  FeatureColor,
  FEATURE_COLOR_ORDER,
  BatchPromptResult,
  BatchState,
  CombinedSteeringRequest,
  CombinedSteeringResponse,
  ClusterContext,
  ClusterBudget,
} from '../types/steering';
import { SAE } from '../types/sae';
import * as steeringApi from '../api/steering';
import type { SteeringTaskResponse } from '../api/steering';
import { computeBaselineStrength } from '../utils/steeringStrength';

// Maximum number of features that can be selected (Feature 011: raised 4 → 20)
export const MAX_SELECTED_FEATURES = 20;

// Steering dispatch (Feature 011): the store keeps the existing `combinedMode`
// boolean (true = Blended, all features summed via /combined; false = Compare,
// each feature its own output via /compare). The UI renders this as a two-way
// Blended | Compare segmented toggle instead of the old checkbox.

// Input to addFeature. `strength` is optional (Feature 011): when omitted, the
// store auto-computes a baseline from `activation_frequency`. The stat fields
// let the baseline calculation and the tile display use per-feature data.
export type AddFeatureInput = Omit<
  SelectedFeature,
  'color' | 'instance_id' | 'strength' | 'strengthSource'
> & { strength?: number };

// Batch coordination state for async promise resolution
// This coordinates between generateBatchComparison (creates promise) and
// handleAsyncCompleted/handleAsyncFailed (resolves/rejects promise)
interface BatchResolver {
  resolve: (result: SteeringComparisonResponse) => void;
  reject: (error: Error) => void;
  taskId: string;  // Track which task this resolver is for
  timeoutId: ReturnType<typeof setTimeout>;  // Cleanup timeout reference
  createdAt: number;  // For debugging stale resolvers
}

let pendingBatchResolver: BatchResolver | null = null;

/**
 * Cleanup the pending batch resolver safely.
 * Clears the timeout, rejects the outgoing promise (so a superseded operation
 * fails cleanly instead of hanging forever), and nulls the resolver.
 */
function cleanupBatchResolver(reason?: string): void {
  if (pendingBatchResolver) {
    clearTimeout(pendingBatchResolver.timeoutId);
    const outgoing = pendingBatchResolver;
    pendingBatchResolver = null;
    if (reason) {
      outgoing.reject(new Error(reason));
    }
  }
}

/**
 * Create a batch resolver for a specific task.
 * Includes automatic timeout cleanup to prevent memory leaks.
 */
function createBatchResolver(
  taskId: string,
  timeoutMs: number,
  onTimeout: () => void
): Promise<SteeringComparisonResponse> {
  // Clean up any existing resolver first, rejecting its promise so the
  // superseded operation doesn't hang indefinitely.
  cleanupBatchResolver('Superseded by a newer comparison request');

  return new Promise<SteeringComparisonResponse>((resolve, reject) => {
    const timeoutId = setTimeout(() => {
      if (pendingBatchResolver?.taskId === taskId) {
        console.log(`[SteeringStore] Batch resolver timeout for task ${taskId}`);
        pendingBatchResolver = null;
        onTimeout();
        reject(new Error(`Timeout: generation took longer than ${timeoutMs / 1000}s`));
      }
    }, timeoutMs);

    pendingBatchResolver = {
      resolve,
      reject,
      taskId,
      timeoutId,
      createdAt: Date.now(),
    };
  });
}

// Sweep resolver for WebSocket-based sweep coordination
// Similar pattern to batch resolver but for StrengthSweepResponse
interface SweepResolver {
  resolve: (result: StrengthSweepResponse) => void;
  reject: (error: Error) => void;
  taskId: string;
  timeoutId: ReturnType<typeof setTimeout>;
  createdAt: number;
}

let pendingSweepResolver: SweepResolver | null = null;

/**
 * Cleanup the pending sweep resolver safely.
 * Rejects the outgoing promise if a reason is given (superseded request).
 */
function cleanupSweepResolver(reason?: string): void {
  if (pendingSweepResolver) {
    clearTimeout(pendingSweepResolver.timeoutId);
    const outgoing = pendingSweepResolver;
    pendingSweepResolver = null;
    if (reason) {
      outgoing.reject(new Error(reason));
    }
  }
}

/**
 * Create a sweep resolver for a specific task.
 * Includes automatic timeout cleanup to prevent memory leaks.
 */
function createSweepResolver(
  taskId: string,
  timeoutMs: number,
  onTimeout: () => void
): Promise<StrengthSweepResponse> {
  // Clean up any existing resolver first, rejecting its promise so the
  // superseded operation doesn't hang indefinitely.
  cleanupSweepResolver('Superseded by a newer sweep request');

  return new Promise<StrengthSweepResponse>((resolve, reject) => {
    const timeoutId = setTimeout(() => {
      if (pendingSweepResolver?.taskId === taskId) {
        console.log(`[SteeringStore] Sweep resolver timeout for task ${taskId}`);
        pendingSweepResolver = null;
        onTimeout();
        reject(new Error(`Timeout: sweep took longer than ${timeoutMs / 1000}s`));
      }
    }, timeoutMs);

    pendingSweepResolver = {
      resolve,
      reject,
      taskId,
      timeoutId,
      createdAt: Date.now(),
    };
  });
}

// Combined resolver for WebSocket-based combined steering coordination
interface CombinedResolver {
  resolve: (result: CombinedSteeringResponse) => void;
  reject: (error: Error) => void;
  taskId: string;
  timeoutId: ReturnType<typeof setTimeout>;
  createdAt: number;
}

let pendingCombinedResolver: CombinedResolver | null = null;

/**
 * Cleanup the pending combined resolver safely.
 * Rejects the outgoing promise if a reason is given (superseded request).
 */
function cleanupCombinedResolver(reason?: string): void {
  if (pendingCombinedResolver) {
    clearTimeout(pendingCombinedResolver.timeoutId);
    const outgoing = pendingCombinedResolver;
    pendingCombinedResolver = null;
    if (reason) {
      outgoing.reject(new Error(reason));
    }
  }
}

/**
 * Create a combined resolver for a specific task.
 * Includes automatic timeout cleanup to prevent memory leaks.
 */
function createCombinedResolver(
  taskId: string,
  timeoutMs: number,
  onTimeout: () => void
): Promise<CombinedSteeringResponse> {
  // Clean up any existing resolver first, rejecting its promise so the
  // superseded operation doesn't hang indefinitely.
  cleanupCombinedResolver('Superseded by a newer combined generation request');

  return new Promise<CombinedSteeringResponse>((resolve, reject) => {
    const timeoutId = setTimeout(() => {
      if (pendingCombinedResolver?.taskId === taskId) {
        console.log(`[SteeringStore] Combined resolver timeout for task ${taskId}`);
        pendingCombinedResolver = null;
        onTimeout();
        reject(new Error(`Timeout: combined generation took longer than ${timeoutMs / 1000}s`));
      }
    }, timeoutMs);

    pendingCombinedResolver = {
      resolve,
      reject,
      taskId,
      timeoutId,
      createdAt: Date.now(),
    };
  });
}

/**
 * Adapt a CombinedSteeringResponse into a single-variation
 * SteeringComparisonResponse so the existing batch-results UI can render a
 * blended output without special-casing. The one "steered" entry is the summed
 * (blended) generation; unsteered is the baseline when present.
 *
 * Feature 012: also carries `applied_features` (server truth) and titles the
 * blend via blendedTitle.
 */

/**
 * Blended result title chain (Feature 012): cluster display token → generic,
 * with the single-feature case folded in (a 1-feature blend is honestly titled
 * by the member itself; the cluster token prefixes it when provenance exists).
 * Feature 014 will prepend an authored-profile-name tier here — one function
 * owns the whole chain so future tiers land in exactly one place.
 */
export function blendedTitle(
  ctx: ClusterContext | null,
  n: number,
  loneLabel?: string | null,
): string {
  if (n === 1) {
    const label = loneLabel || 'Blended (1 feature)';
    return ctx ? `${ctx.display_token} — ${label}` : label;
  }
  const blend = `Blended (${n} features)`;
  return ctx ? `${ctx.display_token} — ${blend}` : blend;
}

/**
 * Apply the cluster-intensity dial λ ONCE, at request-build time (Feature 013
 * step 9). Tiles/pins always show pre-λ strengths; only the outgoing request
 * is scaled, and only while a cluster budget governs the selection.
 */
function applyIntensity(
  features: SelectedFeature[],
  intensity: number,
  active: boolean,
): SelectedFeature[] {
  if (!active || intensity === 1) return features;
  return features.map((f) => ({
    ...f,
    strength: Math.round(f.strength * intensity * 10) / 10,
  }));
}

function combinedToComparison(
  c: CombinedSteeringResponse,
  ctx: ClusterContext | null = null,
): SteeringComparisonResponse {
  const fa = c.features_applied;
  const blendedLabel = blendedTitle(
    ctx,
    fa.length,
    fa.length === 1 ? fa[0].label || `Feature #${fa[0].feature_idx}` : undefined,
  );
  return {
    comparison_id: c.combined_id,
    sae_id: c.sae_id,
    model_id: c.model_id,
    prompt: c.prompt,
    unsteered: c.baseline_output != null
      ? { text: c.baseline_output, metrics: c.baseline_metrics }
      : null,
    steered: [
      {
        text: c.combined_output,
        // Synthetic feature_config representing the whole blend; carries the
        // first feature's identity plus a descriptive label.
        feature_config: {
          ...(c.features_applied[0] as unknown as SelectedFeature),
          label: blendedLabel,
        },
        metrics: c.combined_metrics,
      },
    ],
    steered_multi: null,
    metrics_summary: null,
    // Feature 012: full applied-features summary rides the adapted result so
    // the UI can prove every member contributed (server-truth, not request state).
    applied_features: c.features_applied,
    total_time_ms: c.total_time_ms,
    created_at: c.created_at,
  };
}

interface SteeringState {
  // Selected SAE
  selectedSAE: SAE | null;

  // Selected features (up to 4)
  selectedFeatures: SelectedFeature[];

  // Cluster provenance (Feature 012): non-null only while the selection is
  // exactly the set handed off from one cluster. Any selection mutation clears
  // it (dropping to the generic label is always honest; a stale cluster label
  // never is). Deliberately NOT in the persist partialize.
  clusterContext: ClusterContext | null;

  // Prompts (batch support)
  prompts: string[];

  // Generation parameters
  generationParams: GenerationParams;
  advancedParams: AdvancedGenerationParams | null;
  showAdvancedParams: boolean;

  // Comparison state (single prompt)
  isGenerating: boolean;
  comparisonId: string | null;
  taskId: string | null;  // Celery task ID for async steering
  progress: number;
  progressMessage: string | null;
  currentComparison: SteeringComparisonResponse | null;

  // Recent comparisons (persisted for reload)
  recentComparisons: Array<{
    id: string;
    prompt: string;
    timestamp: string;
    result: SteeringComparisonResponse;
  }>;

  // Batch processing state
  batchState: BatchState | null;

  // Strength sweep state
  isSweeping: boolean;
  sweepResults: StrengthSweepResponse | null;

  // Combined mode state
  combinedMode: boolean;
  isCombinedGenerating: boolean;
  combinedResults: CombinedSteeringResponse | null;

  // Experiment management
  experiments: SteeringExperiment[];
  experimentsLoading: boolean;
  experimentsPagination: {
    skip: number;
    limit: number;
    total: number;
    hasMore: boolean;
  };

  // Error state
  error: string | null;

  // Actions - SAE Selection
  selectSAE: (sae: SAE | null) => void;

  // Cluster context (Feature 012)
  setClusterContext: (ctx: ClusterContext | null) => void;

  // Feature 013: allocation + rebalance + intensity
  requestClusterAllocation: (groupCohesion?: number | null) => Promise<void>;
  rebalanceStrength: (instanceId: string, newStrength: number) => void;
  togglePin: (instanceId: string) => void;
  setIntensity: (value: number) => void;

  // Feature 014: profile hydration. Replaces the selection with the profile's
  // explicit tuned strengths (auto-baselines bypassed), restores the budget
  // snapshot + λ, and sets activeProfile LAST. Returns false when the profile
  // cannot load (unbound, SAE mismatch, no members).
  loadProfileIntoSteering: (profile: import('../types/clusterProfile').ClusterProfile) => boolean;

  // Active profile identity (label tier 1); cleared with clusterContext on any
  // selection mutation — a stale profile title is the mislabeling 012 removed.
  activeProfile: { id: string; name: string } | null;
  // Transient hydration guard: requestClusterAllocation must not fire over a
  // profile's explicit strengths (they ARE the tuned allocation).
  isHydratingProfile: boolean;

  // Title baked for the CURRENT combinedResults at completion time (Feature
  // 012). Live-deriving from clusterContext retitles old results when context
  // changes — the exact mislabeling this feature removes.
  combinedResultsTitle: string | null;

  // Feature 013 (IDL-29): cluster strength budget state. Non-null only while a
  // server-computed allocation governs the current selection. The formula is
  // server-side; the frontend owns only budget-preserving rebalance.
  clusterBudget: ClusterBudget | null;
  // One-line notice shown when the budget model deliberately did NOT engage
  // (e.g. low-cohesion gate). Cleared alongside clusterBudget.
  clusterNotice: string | null;
  // Master cluster-intensity dial λ ∈ [0,2]; applied ONCE, in the request
  // builders (tiles show pre-λ strengths). Not persisted.
  intensity: number;

  // Actions - Feature Selection
  addFeature: (feature: AddFeatureInput) => boolean;
  removeFeature: (instanceId: string) => void;
  duplicateFeature: (instanceId: string) => boolean;
  updateFeatureStrength: (instanceId: string, strength: number) => void;
  setAdditionalStrengths: (instanceId: string, strengths: number[]) => void;
  updateAdditionalStrength: (instanceId: string, strengthIndex: number, newStrength: number) => void;
  removeAdditionalStrength: (instanceId: string, strengthIndex: number) => void;
  applyStrengthPreset: (strength: number) => void;
  applyAutoBaseline: () => void;
  clearFeatures: () => void;
  reorderFeatures: (fromIndex: number, toIndex: number) => void;

  // Actions - Prompts (batch support)
  addPrompt: () => void;
  removePrompt: (index: number) => void;
  updatePrompt: (index: number, value: string) => void;
  clearPrompts: () => void;
  replacePromptWithMultiple: (index: number, newPrompts: string[]) => void;
  setPrompts: (prompts: string[]) => void;

  // Actions - Generation Parameters
  setGenerationParams: (params: Partial<GenerationParams>) => void;
  setAdvancedParams: (params: Partial<AdvancedGenerationParams> | null) => void;
  toggleAdvancedParams: () => void;
  resetParams: () => void;

  // Actions - Comparison (single prompt)
  generateComparison: (includeUnsteered?: boolean, computeMetrics?: boolean) => Promise<SteeringTaskResponse>;
  abortComparison: () => Promise<void>;
  clearComparison: () => void;
  loadRecentComparison: (id: string) => void;
  recoverTaskResult: (taskId: string) => Promise<SteeringComparisonResponse>;
  clearRecentComparisons: () => void;

  // Actions - Async Task Handling (WebSocket callbacks)
  handleAsyncProgress: (percent: number, message: string, currentFeature?: number, currentStrength?: number) => void;
  handleAsyncCompleted: (result: SteeringComparisonResponse | StrengthSweepResponse | CombinedSteeringResponse) => void;
  handleAsyncFailed: (error: string) => void;

  // Actions - Batch Processing
  generateBatchComparison: (includeUnsteered?: boolean, computeMetrics?: boolean) => Promise<void>;
  abortBatch: () => Promise<void>;
  clearBatchResults: () => void;

  // Actions - Strength Sweep
  runStrengthSweep: (featureIdx: number, layer: number, strengthValues: number[]) => Promise<StrengthSweepResponse>;
  clearSweepResults: () => void;

  // Actions - Combined Mode
  setCombinedMode: (enabled: boolean) => void;
  generateCombined: (includeBaseline?: boolean, computeMetrics?: boolean) => Promise<CombinedSteeringResponse>;
  clearCombinedResults: () => void;

  // Actions - Progress Updates (WebSocket)
  updateProgress: (update: SteeringProgressUpdate) => void;

  // Actions - Experiments
  fetchExperiments: (params?: { skip?: number; limit?: number; search?: string; sae_id?: string }) => Promise<void>;
  saveExperiment: (name: string, description?: string, tags?: string[]) => Promise<SteeringExperiment>;
  loadExperiment: (experiment: SteeringExperiment) => void;
  deleteExperiment: (id: string) => Promise<void>;
  deleteExperimentsBatch: (ids: string[]) => Promise<void>;

  // Actions - Error Handling
  setError: (error: string | null) => void;
  clearError: () => void;

  // Actions - Full Reset
  resetSession: () => void;

  // Actions - State Recovery (after page refresh)
  recoverActiveTask: () => Promise<void>;
  _hasHydrated: boolean;
  setHasHydrated: (hydrated: boolean) => void;
}

export const useSteeringStore = create<SteeringState>()(
  devtools(
    persist(
      (set, get) => ({
      // Initial state
      selectedSAE: null,
      selectedFeatures: [],
      clusterContext: null,
      combinedResultsTitle: null,
      clusterBudget: null,
      clusterNotice: null,
      activeProfile: null,
      isHydratingProfile: false,
      intensity: 1,
      prompts: [''],
      generationParams: { ...DEFAULT_GENERATION_PARAMS },
      advancedParams: null,
      showAdvancedParams: false,
      isGenerating: false,
      comparisonId: null,
      taskId: null,
      progress: 0,
      progressMessage: null,
      currentComparison: null,
      recentComparisons: [],
      batchState: null,
      isSweeping: false,
      sweepResults: null,
      combinedMode: false,
      isCombinedGenerating: false,
      combinedResults: null,
      experiments: [],
      experimentsLoading: false,
      experimentsPagination: {
        skip: 0,
        limit: 20,
        total: 0,
        hasMore: false,
      },
      error: null,
      _hasHydrated: false,

      // Select an SAE for steering
      selectSAE: (sae: SAE | null) => {
        set({
          selectedSAE: sae,
          selectedFeatures: [], // Clear features when SAE changes
          clusterContext: null,
          activeProfile: null,
          clusterBudget: null,
          clusterNotice: null,
          intensity: 1,
          currentComparison: null,
          sweepResults: null,
          combinedResults: null, // stale results from another SAE must not linger
          combinedResultsTitle: null,
        });
      },

      // Feature 012: cluster provenance. Callers set it ONLY after a clean
      // single-cluster hand-off; every selection mutation clears it.
      setClusterContext: (ctx: ClusterContext | null) => {
        set({ clusterContext: ctx });
      },

      // Feature 013: request the server-computed allocation for the CURRENT
      // selection (fires after a cluster hand-off). Progressive: the Feature
      // 011 solo baselines already applied stay until the response lands; a
      // stale response (selection changed meanwhile) is dropped.
      requestClusterAllocation: async (groupCohesion: number | null = null) => {
        const { selectedSAE, selectedFeatures } = get();
        if (!selectedSAE || selectedFeatures.length === 0) return;
        // Profile hydration carries EXPLICIT tuned strengths — never overwrite
        // them with a fresh allocation (Feature 014 guard).
        if (get().isHydratingProfile || get().activeProfile) return;
        // N=1 routes through the solo path verbatim (IDL-29 step 10).
        if (selectedFeatures.length === 1) return;

        // Duplicate feature indices would double-count one direction in the
        // formula — refuse and keep the solo baselines (v1 single-membership).
        const idxs = selectedFeatures.map((f) => f.feature_idx);
        if (new Set(idxs).size !== idxs.length) {
          console.warn('[SteeringStore] Duplicate feature indices — skipping cluster allocation');
          return;
        }

        // Snapshot the request order: response arrays are index-aligned with
        // THIS order, and instance ids make the apply step order-safe even if
        // the user reorders tiles while the request is in flight.
        const requestMembers = selectedFeatures.map((f) => ({
          instance_id: f.instance_id,
          feature_idx: f.feature_idx,
        }));
        const requestKey = requestMembers.map((m) => m.instance_id).sort().join(',');
        try {
          const allocation = await steeringApi.computeClusterAllocation({
            sae_id: selectedSAE.id,
            members: selectedFeatures.map((f) => ({
              feature_idx: f.feature_idx,
              layer: f.layer,
              similarity: f.similarity ?? null,
              activation_frequency: f.activation_frequency ?? null,
              sign: f.strength < 0 ? -1 : 1,
            })),
            group_cohesion: groupCohesion,
          });

          // Stale guard: same instances, same SAE.
          const current = get().selectedFeatures;
          const currentKey = current.map((f) => f.instance_id).sort().join(',');
          if (currentKey !== requestKey || get().selectedSAE?.id !== selectedSAE.id) {
            console.log('[SteeringStore] Dropping stale cluster allocation');
            return;
          }

          // Low-cohesion gate: keep solo baselines AND no governing budget —
          // rebalance, λ, and the budget bar must not engage on a gated cluster.
          if (allocation.flags.includes('low_cohesion')) {
            set({
              clusterBudget: null,
              clusterNotice:
                'Low cluster cohesion — kept per-feature baselines (budget model gated)',
            });
            return;
          }

          // Per-instance allocation map, built from the REQUEST order (the
          // only order the response arrays are aligned with). Server strengths
          // already carry the member sign.
          const allocByInstance: Record<string, { weight: number; strength: number }> = {};
          requestMembers.forEach((m, i) => {
            allocByInstance[m.instance_id] = {
              weight: allocation.weights[i] ?? 0,
              strength: allocation.strengths[i] ?? 0,
            };
          });
          const weightsByInstance: Record<string, number> = {};
          for (const [id, a] of Object.entries(allocByInstance)) weightsByInstance[id] = a.weight;

          set({
            clusterBudget: {
              B: allocation.B,
              B_dir: allocation.B_dir,
              G: allocation.G,
              flags: allocation.flags,
              approximate: allocation.approximate,
              weightsByInstance,
              // Provenance for saved profiles (self-describing budgets, 014).
              formula_id: allocation.formula_id,
              constants: allocation.constants_used,
              f_eff: allocation.f_eff,
            },
            clusterNotice: null,
            selectedFeatures: current.map((f) => {
              const a = allocByInstance[f.instance_id];
              return a
                ? { ...f, strength: a.strength, strengthSource: 'cluster' as const, pinned: false }
                : f;
            }),
          });
        } catch (error) {
          // Allocation failure is non-fatal: solo baselines remain in effect.
          console.warn('[SteeringStore] Cluster allocation failed; keeping solo baselines', error);
        }
      },

      // Feature 013: budget-preserving strength edit (cluster mode only).
      // Pins the edited member; redistributes R = B − Σ|pinned| across
      // unpinned members by renormalized allocation weights. B and G are never
      // recomputed on strength edits.
      rebalanceStrength: (instanceId: string, newStrength: number) => {
        const { clusterBudget, selectedFeatures } = get();
        if (!clusterBudget) {
          // Not in cluster mode — plain edit.
          get().updateFeatureStrength(instanceId, newStrength);
          return;
        }
        const features = selectedFeatures.map((f) =>
          f.instance_id === instanceId
            ? { ...f, strength: newStrength, pinned: true, strengthSource: 'manual' as const }
            : { ...f },
        );
        const pinnedTotal = features.filter((f) => f.pinned).reduce((s, f) => s + Math.abs(f.strength), 0);
        const remaining = clusterBudget.B - pinnedTotal;
        const unpinned = features.filter((f) => !f.pinned);
        if (unpinned.length > 0) {
          if (remaining < 0) {
            // Over budget: unpinned drop to 0; pins are never rescaled.
            for (const f of features) if (!f.pinned) f.strength = 0;
          } else {
            const wsum = unpinned.reduce(
              (s, f) => s + (clusterBudget.weightsByInstance[f.instance_id] ?? 0), 0);
            for (const f of features) {
              if (f.pinned) continue;
              const w = clusterBudget.weightsByInstance[f.instance_id] ?? 0;
              const share = wsum > 0 ? w / wsum : 1 / unpinned.length;
              const sign = f.strength < 0 ? -1 : 1;
              f.strength = Math.round(sign * remaining * share * 10) / 10;
            }
          }
        }
        set({ selectedFeatures: features });
      },

      togglePin: (instanceId: string) => {
        set((state) => ({
          selectedFeatures: state.selectedFeatures.map((f) =>
            f.instance_id === instanceId ? { ...f, pinned: !f.pinned } : f,
          ),
        }));
      },

      setIntensity: (value: number) => {
        set({ intensity: Math.max(0, Math.min(2, value)) });
      },

      // Feature 014: hydrate a saved profile into the steering selection.
      loadProfileIntoSteering: (profile) => {
        const { selectedSAE } = get();
        if (!profile.members || profile.members.length === 0) return false;
        // Unbound profiles can't steer; bound profiles must match the panel's SAE
        // (ProfilesMenu lists per-SAE, so this only trips on stale UI state).
        if (!profile.sae_id || !selectedSAE || profile.sae_id !== selectedSAE.id) return false;
        if (profile.members.length > MAX_SELECTED_FEATURES) return false;

        set({ isHydratingProfile: true });
        try {
          const layer = selectedSAE.layer ?? 0;
          const features: SelectedFeature[] = profile.members.map((m, i) => ({
            instance_id: `${m.feature_idx}-${layer}-${Date.now()}-${i}-${Math.random().toString(36).substring(2, 9)}`,
            feature_idx: m.feature_idx,
            feature_id: null, // DB feature id unknown from a portable profile
            layer,
            // Profile strengths are EXPLICIT tuned values — auto-baselines bypassed.
            strength: m.strength,
            strengthSource: 'manual' as const,
            pinned: m.pinned ?? false,
            label: m.label ?? null,
            color: FEATURE_COLOR_ORDER[i % FEATURE_COLOR_ORDER.length],
            max_activation: m.max_activation ?? null,
            activation_frequency: m.activation_frequency ?? null,
            similarity: m.similarity ?? null,
          }));

          // Budget snapshot (013): weights reconstructed from |strength| shares —
          // exact for untouched allocations, a fair prior after manual edits.
          const b = profile.budget;
          let clusterBudget: ClusterBudget | null = null;
          if (b && b.B != null) {
            const totalAbs = features.reduce((s, f) => s + Math.abs(f.strength), 0);
            const weightsByInstance: Record<string, number> = {};
            features.forEach((f) => {
              weightsByInstance[f.instance_id] =
                totalAbs > 0 ? Math.abs(f.strength) / totalAbs : 1 / features.length;
            });
            clusterBudget = {
              B: b.B,
              B_dir: b.B_dir ?? b.B,
              G: b.G ?? 1,
              flags: [],
              approximate: false,
              weightsByInstance,
              // Carry the saved formula provenance back (self-describing).
              formula_id: b.formula_id ?? undefined,
              constants: b.constants ?? undefined,
              f_eff: b.f_eff ?? null,
            };
          }

          set({
            selectedFeatures: features,
            clusterBudget,
            clusterNotice: null,
            intensity: Math.max(0, Math.min(2, b?.intensity ?? 1)),
            // Label tier 1: the authored profile name titles blended results
            // through the existing clusterContext baking path.
            clusterContext: {
              group_id: profile.source_group_id ?? profile.id,
              display_token: profile.name,
            },
            // Set LAST (FTID rule): consumers may react to activeProfile by
            // reading the rest of the hydrated state.
            activeProfile: { id: profile.id, name: profile.name },
          });
          return true;
        } finally {
          set({ isHydratingProfile: false });
        }
      },

      // Add a feature to selection (returns false if max reached)
      // Note: Duplicates of the same feature_idx/layer are now allowed (each gets unique instance_id)
      addFeature: (feature: AddFeatureInput) => {
        const { selectedFeatures } = get();

        // Check if max features reached
        if (selectedFeatures.length >= MAX_SELECTED_FEATURES) {
          return false;
        }

        // Next color: first unused from the 20-order, wrapping when all are used.
        const usedColors = selectedFeatures.map((f) => f.color);
        const nextColor =
          FEATURE_COLOR_ORDER.find((c) => !usedColors.includes(c)) ||
          FEATURE_COLOR_ORDER[selectedFeatures.length % FEATURE_COLOR_ORDER.length];

        // Generate unique instance_id
        const instanceId = `${feature.feature_idx}-${feature.layer}-${Date.now()}-${Math.random().toString(36).substring(2, 9)}`;

        // Feature 011: if the caller supplied an explicit strength, honor it
        // (manual); otherwise auto-compute the baseline from activation frequency,
        // falling back to the default when frequency is unavailable.
        let strength: number;
        let strengthSource: SelectedFeature['strengthSource'];
        if (feature.strength != null) {
          strength = feature.strength;
          strengthSource = 'manual';
        } else {
          const baseline = computeBaselineStrength(feature.activation_frequency);
          strength = baseline.value;
          strengthSource = baseline.source;
        }

        const newFeature: SelectedFeature = {
          instance_id: instanceId,
          feature_idx: feature.feature_idx,
          layer: feature.layer,
          strength,
          strengthSource,
          label: feature.label,
          color: nextColor,
          feature_id: feature.feature_id,
          max_activation: feature.max_activation ?? null,
          activation_frequency: feature.activation_frequency ?? null,
        };

        // Any selection mutation invalidates cluster provenance (Feature 012)
        // and the cluster budget computed for it (Feature 013).
        set({ selectedFeatures: [...selectedFeatures, newFeature], clusterContext: null, activeProfile: null, clusterBudget: null, clusterNotice: null });
        return true;
      },

      // Remove a feature from selection by instance_id
      removeFeature: (instanceId: string) => {
        set((state) => ({
          selectedFeatures: state.selectedFeatures.filter(
            (f) => f.instance_id !== instanceId
          ),
          clusterContext: null, // selection mutated (Feature 012)
          activeProfile: null,
          clusterBudget: null,
          clusterNotice: null,
        }));
      },

      // Duplicate a feature with negated strength (for A/B comparison)
      duplicateFeature: (instanceId: string) => {
        const { selectedFeatures } = get();

        // Check if max features reached
        if (selectedFeatures.length >= MAX_SELECTED_FEATURES) {
          return false;
        }

        // Find the original feature by instance_id
        const original = selectedFeatures.find(
          (f) => f.instance_id === instanceId
        );
        if (!original) {
          return false;
        }

        // Next color: first unused, wrapping when all 20 are taken.
        const usedColors = selectedFeatures.map((f) => f.color);
        const nextColor =
          FEATURE_COLOR_ORDER.find((c) => !usedColors.includes(c)) ||
          FEATURE_COLOR_ORDER[selectedFeatures.length % FEATURE_COLOR_ORDER.length];

        // Generate unique instance_id for the duplicate
        const newInstanceId = `${original.feature_idx}-${original.layer}-${Date.now()}-${Math.random().toString(36).substring(2, 9)}`;

        // Copy and negate additional_strengths if they exist
        const negatedAdditionalStrengths = original.additional_strengths
          ? original.additional_strengths.map((s) => -s)
          : undefined;

        // Create duplicate with negated strength (for testing opposite direction)
        const duplicatedFeature: SelectedFeature = {
          instance_id: newInstanceId,
          feature_idx: original.feature_idx,
          layer: original.layer,
          strength: -original.strength, // Negate the strength
          strengthSource: 'manual',
          additional_strengths: negatedAdditionalStrengths, // Copy and negate additional strengths
          label: original.label ? `${original.label} (copy)` : null,
          color: nextColor,
          feature_id: original.feature_id,
          max_activation: original.max_activation ?? null,
          activation_frequency: original.activation_frequency ?? null,
        };

        set({ selectedFeatures: [...selectedFeatures, duplicatedFeature], clusterContext: null, activeProfile: null, clusterBudget: null, clusterNotice: null });
        return true;
      },

      // Update feature strength by instance_id
      updateFeatureStrength: (instanceId: string, strength: number) => {
        set((state) => ({
          selectedFeatures: state.selectedFeatures.map((f) =>
            f.instance_id === instanceId
              ? { ...f, strength }
              : f
          ),
        }));
      },

      // Set all additional strengths for a feature (replaces existing) by instance_id
      setAdditionalStrengths: (instanceId: string, strengths: number[]) => {
        // Max 3 additional strengths
        const clampedStrengths = strengths.slice(0, 3);
        set((state) => ({
          selectedFeatures: state.selectedFeatures.map((f) =>
            f.instance_id === instanceId
              ? { ...f, additional_strengths: clampedStrengths.length > 0 ? clampedStrengths : undefined }
              : f
          ),
        }));
      },

      // Update a specific additional strength by index using instance_id
      updateAdditionalStrength: (instanceId: string, strengthIndex: number, newStrength: number) => {
        set((state) => ({
          selectedFeatures: state.selectedFeatures.map((f) => {
            if (f.instance_id === instanceId && f.additional_strengths) {
              const newAdditional = [...f.additional_strengths];
              if (strengthIndex >= 0 && strengthIndex < newAdditional.length) {
                newAdditional[strengthIndex] = newStrength;
              }
              return { ...f, additional_strengths: newAdditional };
            }
            return f;
          }),
        }));
      },

      // Remove a specific additional strength by index using instance_id
      removeAdditionalStrength: (instanceId: string, strengthIndex: number) => {
        set((state) => ({
          selectedFeatures: state.selectedFeatures.map((f) => {
            if (f.instance_id === instanceId && f.additional_strengths) {
              const newAdditional = f.additional_strengths.filter((_, i) => i !== strengthIndex);
              return { ...f, additional_strengths: newAdditional.length > 0 ? newAdditional : undefined };
            }
            return f;
          }),
        }));
      },

      // Apply strength preset to all selected features
      applyStrengthPreset: (strength: number) => {
        // Uniform presets are incompatible with a computed cluster budget —
        // applying one exits cluster mode (Feature 013, documented behavior).
        set((state) => ({
          clusterBudget: null,
          clusterNotice: null,
          selectedFeatures: state.selectedFeatures.map((f) => ({
            ...f,
            strength,
            strengthSource: 'manual' as const,
            pinned: false,
          })),
        }));
      },

      // Feature 011: recompute each feature's baseline from its activation
      // frequency (default fallback where frequency is unavailable).
      applyAutoBaseline: () => {
        set((state) => ({
          selectedFeatures: state.selectedFeatures.map((f) => {
            const baseline = computeBaselineStrength(f.activation_frequency);
            return { ...f, strength: baseline.value, strengthSource: baseline.source };
          }),
        }));
      },

      // Clear all selected features
      clearFeatures: () => {
        set({ selectedFeatures: [], clusterContext: null, activeProfile: null, clusterBudget: null, clusterNotice: null, currentComparison: null });
      },

      // Reorder features (drag and drop)
      reorderFeatures: (fromIndex: number, toIndex: number) => {
        set((state) => {
          const features = [...state.selectedFeatures];
          const [removed] = features.splice(fromIndex, 1);
          features.splice(toIndex, 0, removed);
          return { selectedFeatures: features };
        });
      },

      // Add a new empty prompt to the list
      addPrompt: () => {
        set((state) => ({
          prompts: [...state.prompts, ''],
        }));
      },

      // Remove a prompt by index (minimum 1 prompt required)
      removePrompt: (index: number) => {
        set((state) => {
          if (state.prompts.length <= 1) {
            return state; // Keep at least one prompt
          }
          const newPrompts = state.prompts.filter((_, i) => i !== index);
          return { prompts: newPrompts };
        });
      },

      // Update a specific prompt by index
      updatePrompt: (index: number, value: string) => {
        set((state) => {
          const newPrompts = [...state.prompts];
          if (index >= 0 && index < newPrompts.length) {
            newPrompts[index] = value;
          }
          return { prompts: newPrompts };
        });
      },

      // Clear all prompts (reset to single empty prompt)
      clearPrompts: () => {
        set({ prompts: [''] });
      },

      // Replace a single prompt with multiple prompts (used for multi-line paste parsing)
      replacePromptWithMultiple: (index: number, newPrompts: string[]) => {
        set((state) => {
          if (index < 0 || index >= state.prompts.length || newPrompts.length === 0) {
            return state;
          }
          // Splice: keep prompts before index, insert newPrompts, keep prompts after index
          const before = state.prompts.slice(0, index);
          const after = state.prompts.slice(index + 1);
          return { prompts: [...before, ...newPrompts, ...after] };
        });
      },

      // Set all prompts at once (used for loading from template)
      setPrompts: (prompts: string[]) => {
        set({ prompts: prompts.length > 0 ? prompts : [''] });
      },

      // Set generation parameters
      setGenerationParams: (params: Partial<GenerationParams>) => {
        set((state) => ({
          generationParams: { ...state.generationParams, ...params },
        }));
      },

      // Set advanced parameters
      setAdvancedParams: (params: Partial<AdvancedGenerationParams> | null) => {
        if (params === null) {
          set({ advancedParams: null });
        } else {
          set((state) => ({
            advancedParams: state.advancedParams
              ? { ...state.advancedParams, ...params }
              : {
                  repetition_penalty: 1.15,
                  presence_penalty: 0.0,
                  frequency_penalty: 0.0,
                  do_sample: true,
                  stop_sequences: [],
                  ...params,
                },
          }));
        }
      },

      // Toggle advanced parameters visibility
      toggleAdvancedParams: () => {
        set((state) => ({ showAdvancedParams: !state.showAdvancedParams }));
      },

      // Reset parameters to defaults
      resetParams: () => {
        set({
          generationParams: { ...DEFAULT_GENERATION_PARAMS },
          advancedParams: null,
        });
      },

      // Generate comparison (uses first prompt for single-prompt mode)
      // Now uses async Celery-based API with WebSocket progress updates
      generateComparison: async (includeUnsteered = true, computeMetrics = false) => {
        const { selectedSAE, selectedFeatures, prompts, generationParams, advancedParams, isGenerating } = get();

        // Guard against double submission (can happen with fast double-click before state updates)
        if (isGenerating) {
          console.log('[SteeringStore] Already generating, ignoring duplicate call');
          return { task_id: '', task_type: 'compare', status: 'ignored', websocket_channel: '', message: 'Already generating' } as SteeringTaskResponse;
        }

        const prompt = prompts[0] || '';

        if (!selectedSAE) {
          throw new Error('No SAE selected');
        }
        if (selectedFeatures.length === 0) {
          throw new Error('No features selected');
        }
        if (!prompt.trim()) {
          throw new Error('Prompt is required');
        }

        set({
          isGenerating: true,
          taskId: null,
          comparisonId: null,
          progress: 0,
          progressMessage: 'Submitting task...',
          error: null,
          batchState: null, // Clear any batch state for single prompt mode
        });

        try {
          const request: SteeringComparisonRequest = {
            sae_id: selectedSAE.id,
            prompt,
            // λ applies in Compare too — the dial governs the selection, not
            // just the Blended request path (013 review finding).
            selected_features: applyIntensity(selectedFeatures, get().intensity, !!get().clusterBudget),
            generation_params: generationParams,
            include_unsteered: includeUnsteered,
            compute_metrics: computeMetrics,
          };

          if (advancedParams) {
            request.advanced_params = advancedParams;
          }

          // Submit async task - returns immediately with task_id
          const taskResponse = await steeringApi.submitAsyncComparison(request);

          console.log('[SteeringStore] Async task submitted:', taskResponse.task_id);

          // Store task ID for tracking
          set({
            taskId: taskResponse.task_id,
            progress: 5,
            progressMessage: 'Task queued, waiting for worker...',
          });

          // WebSocket hook will handle progress/completed/failed events
          // via handleAsyncProgress, handleAsyncCompleted, handleAsyncFailed

          return taskResponse;
        } catch (error) {
          const errorMessage = error instanceof Error ? error.message : 'Failed to submit task';
          set({
            error: errorMessage,
            isGenerating: false,
            taskId: null,
            progress: 0,
            progressMessage: null,
          });
          throw error;
        }
      },

      // Abort an in-progress comparison (cancels Celery task)
      abortComparison: async () => {
        const { taskId } = get();
        if (!taskId) return;

        try {
          const result = await steeringApi.cancelTask(taskId);
          console.log('[SteeringStore] Task cancelled:', result);
          set({
            isGenerating: false,
            taskId: null,
            progress: 0,
            progressMessage: 'Comparison cancelled',
          });
        } catch (error) {
          const errorMessage = error instanceof Error ? error.message : 'Failed to cancel task';
          set({ error: errorMessage });
        }
      },

      // Clear comparison results
      clearComparison: () => {
        set({
          currentComparison: null,
          comparisonId: null,
          taskId: null,
          progress: 0,
          progressMessage: null,
        });
      },

      // Load a recent comparison by ID
      loadRecentComparison: (id: string) => {
        const { recentComparisons } = get();
        const recent = recentComparisons.find(r => r.id === id);
        if (recent) {
          set({
            currentComparison: recent.result,
            comparisonId: recent.id,
            isGenerating: false,
            taskId: null,
            progress: 100,
            progressMessage: 'Loaded from recent',
            batchState: null,
          });
        }
      },

      // Recover a comparison result from Redis by task ID and add to recent
      recoverTaskResult: async (taskId: string) => {
        try {
          console.log('[SteeringStore] Recovering task result:', taskId);
          const response = await steeringApi.getTaskResult(taskId);

          if (response.status.status === 'success' && response.result) {
            const result = response.result as SteeringComparisonResponse;
            const { recentComparisons } = get();

            // Add to recent comparisons
            const newRecent = {
              id: result.comparison_id,
              prompt: result.prompt,
              timestamp: result.created_at || new Date().toISOString(),
              result,
            };
            const updatedRecent = [newRecent, ...recentComparisons.filter(r => r.id !== result.comparison_id)].slice(0, 10);

            set({
              currentComparison: result,
              comparisonId: result.comparison_id,
              recentComparisons: updatedRecent,
              isGenerating: false,
              taskId: null,
              progress: 100,
              progressMessage: 'Recovered from task',
            });

            console.log('[SteeringStore] Task result recovered and added to recent');
            return result;
          } else {
            throw new Error(response.status.error || 'Task not completed or no result');
          }
        } catch (error) {
          const errorMessage = error instanceof Error ? error.message : 'Failed to recover task';
          console.error('[SteeringStore] Failed to recover task:', errorMessage);
          set({ error: errorMessage });
          throw error;
        }
      },

      // Clear all recent comparisons
      clearRecentComparisons: () => {
        set({ recentComparisons: [] });
      },

      // Handle async progress updates from WebSocket
      handleAsyncProgress: (percent: number, message: string, _currentFeature?: number, _currentStrength?: number) => {
        console.log('[SteeringStore] Async progress:', percent, message);
        set({
          progress: percent,
          progressMessage: message,
        });
      },

      // Handle async completion from WebSocket
      // This handles comparison, sweep, and combined results
      handleAsyncCompleted: (result: SteeringComparisonResponse | StrengthSweepResponse | CombinedSteeringResponse) => {
        const { batchState, recentComparisons, taskId, isSweeping, isCombinedGenerating } = get();
        console.log('[SteeringStore] Async completed:', {
          resultTaskId: 'comparison_id' in result ? result.comparison_id : ('sweep_id' in result ? result.sweep_id : ('combined_id' in result ? result.combined_id : 'unknown')),
          storeTaskId: taskId,
          batchIsRunning: batchState?.isRunning,
          hasBatchResolver: !!pendingBatchResolver,
          batchResolverTaskId: pendingBatchResolver?.taskId,
          isSweeping,
          isCombinedGenerating,
        });

        // Check result type
        const isSweepResult = 'sweep_id' in result;
        const isCombinedResult = 'combined_id' in result;

        // If in sweep mode with a pending resolver for this task, resolve it
        if (isSweeping && pendingSweepResolver && pendingSweepResolver.taskId === taskId && isSweepResult) {
          console.log(`[SteeringStore] Resolving sweep promise for task ${taskId}`);
          const resolver = pendingSweepResolver;
          cleanupSweepResolver();
          resolver.resolve(result as StrengthSweepResponse);
          return;
        }

        // Resolve a pending combined resolver whenever this task produced a
        // combined result. The pending resolver + matching taskId are the
        // correct guard: this covers single-prompt Blended (isCombinedGenerating)
        // AND batch Blended (where the batch loop, not isCombinedGenerating, owns
        // the combined resolver per prompt).
        if (pendingCombinedResolver && pendingCombinedResolver.taskId === taskId && isCombinedResult) {
          console.log(`[SteeringStore] Resolving combined promise for task ${taskId}`);
          const resolver = pendingCombinedResolver;
          cleanupCombinedResolver();
          resolver.resolve(result as CombinedSteeringResponse);
          return;
        }

        // Cast to comparison result for remaining logic
        const comparisonResult = result as SteeringComparisonResponse;

        // Save to recent comparisons (keep last 10) - only for comparison results
        if (!isSweepResult && !isCombinedResult) {
          const newRecent = {
            id: comparisonResult.comparison_id,
            prompt: comparisonResult.prompt,
            timestamp: comparisonResult.created_at || new Date().toISOString(),
            result: comparisonResult,
          };
          const updatedRecent = [newRecent, ...recentComparisons.filter(r => r.id !== comparisonResult.comparison_id)].slice(0, 10);

          // If in batch mode with a pending resolver for this task, resolve the promise
          if (batchState?.isRunning && pendingBatchResolver && pendingBatchResolver.taskId === taskId) {
            console.log(`[SteeringStore] Resolving batch promise for task ${taskId}`);
            const resolver = pendingBatchResolver;
            cleanupBatchResolver();  // Clear timeout and null resolver before resolving
            resolver.resolve(comparisonResult);
            // Still save to recent even in batch mode
            set({ recentComparisons: updatedRecent });
            return;
          }

          // Single prompt mode - update state directly
          set({
            isGenerating: false,
            comparisonId: comparisonResult.comparison_id,
            currentComparison: comparisonResult,
            progress: 100,
            progressMessage: 'Comparison complete',
            recentComparisons: updatedRecent,
          });
        }
      },

      // Handle async failure from WebSocket
      // This handles comparison, sweep, and combined failures
      handleAsyncFailed: (error: string) => {
        console.log('[SteeringStore] Async failed:', error);
        const { batchState, taskId, isSweeping } = get();

        // If in sweep mode with a pending resolver for this task, reject it
        if (isSweeping && pendingSweepResolver && pendingSweepResolver.taskId === taskId) {
          console.log(`[SteeringStore] Rejecting sweep promise for task ${taskId}:`, error);
          const resolver = pendingSweepResolver;
          cleanupSweepResolver();
          resolver.reject(new Error(error));
          return;
        }

        // Reject a pending combined resolver for this task (covers single-prompt
        // Blended and batch Blended — see the matching resolve path).
        if (pendingCombinedResolver && pendingCombinedResolver.taskId === taskId) {
          console.log(`[SteeringStore] Rejecting combined promise for task ${taskId}:`, error);
          const resolver = pendingCombinedResolver;
          cleanupCombinedResolver();
          resolver.reject(new Error(error));
          return;
        }

        // If in batch mode with a pending resolver for this task, reject the promise
        if (batchState?.isRunning && pendingBatchResolver && pendingBatchResolver.taskId === taskId) {
          console.log(`[SteeringStore] Rejecting batch promise for task ${taskId}:`, error);
          const resolver = pendingBatchResolver;
          cleanupBatchResolver();  // Clear timeout and null resolver before rejecting
          resolver.reject(new Error(error));
          return;
        }

        // Single prompt mode - update state directly
        set({
          isGenerating: false,
          isCombinedGenerating: false,
          error: error,
          progress: 0,
          progressMessage: null,
        });
      },

      // Generate batch comparison (iterates through all prompts).
      // Honors combinedMode: Blended runs the summed (combined) generation per
      // prompt; Compare runs the per-feature comparison per prompt.
      generateBatchComparison: async (includeUnsteered = true, computeMetrics = false) => {
        const { selectedSAE, selectedFeatures, prompts, generationParams, advancedParams, isGenerating, batchState, combinedMode } = get();
        // Provenance snapshot for the whole batch: results are titled by the
        // context they were generated under, immune to mid-batch mutations.
        const clusterContextAtStart = get().clusterContext;

        // Guard against double submission
        if (isGenerating || batchState?.isRunning) {
          console.log('[SteeringStore] Already generating batch, ignoring duplicate call');
          return;
        }

        // Filter to only non-empty prompts
        const validPrompts = prompts.filter((p) => p.trim().length > 0);

        if (!selectedSAE) {
          throw new Error('No SAE selected');
        }
        if (selectedFeatures.length === 0) {
          throw new Error('No features selected');
        }
        if (validPrompts.length === 0) {
          throw new Error('At least one prompt is required');
        }

        // Initialize batch state
        const initialResults: BatchPromptResult[] = validPrompts.map((prompt, index) => ({
          prompt,
          promptIndex: index,
          status: 'pending',
          comparison: null,
          error: null,
        }));

        set({
          isGenerating: true,
          currentComparison: null, // Clear previous single comparison
          error: null,
          batchState: {
            isRunning: true,
            currentIndex: 0,
            totalPrompts: validPrompts.length,
            results: initialResults,
            aborted: false,
          },
        });

        // Process each prompt sequentially
        console.log(`[SteeringStore] Starting batch with ${validPrompts.length} prompts`);
        for (let i = 0; i < validPrompts.length; i++) {
          console.log(`[SteeringStore] Batch loop iteration ${i + 1}/${validPrompts.length}`);
          const currentBatch = get().batchState;

          // Check if aborted
          if (currentBatch?.aborted) {
            console.log(`[SteeringStore] Batch aborted at iteration ${i + 1}`);
            set((state) => ({
              isGenerating: false,
              batchState: state.batchState
                ? { ...state.batchState, isRunning: false }
                : null,
            }));
            return;
          }

          const prompt = validPrompts[i];

          // Update current index and mark this prompt as running
          set((state) => ({
            progress: Math.round((i / validPrompts.length) * 100),
            progressMessage: `Processing prompt ${i + 1} of ${validPrompts.length}...`,
            batchState: state.batchState
              ? {
                  ...state.batchState,
                  currentIndex: i,
                  results: state.batchState.results.map((r, idx) =>
                    idx === i ? { ...r, status: 'running' } : r
                  ),
                }
              : null,
          }));

          try {
            const PROMPT_TIMEOUT_MS = 5 * 60 * 1000;
            // Blended per-prompt (combinedMode) vs per-feature Compare per prompt.
            let response: SteeringComparisonResponse;

            if (combinedMode) {
              const combinedRequest: CombinedSteeringRequest = {
                sae_id: selectedSAE.id,
                prompt,
                selected_features: applyIntensity(selectedFeatures, get().intensity, !!get().clusterBudget),
                generation_params: generationParams,
                advanced_params: advancedParams ?? undefined,
                include_baseline: includeUnsteered,
                compute_metrics: computeMetrics,
              };
              const taskResponse = await steeringApi.submitAsyncCombined(combinedRequest);
              console.log(`[SteeringStore] Batch prompt ${i + 1} (blended): task`, taskResponse.task_id);
              set({ taskId: taskResponse.task_id });
              await new Promise((resolve) => setTimeout(resolve, 50));
              const combined = await createCombinedResolver(
                taskResponse.task_id,
                PROMPT_TIMEOUT_MS,
                () => {
                  console.log(`[SteeringStore] Batch prompt ${i + 1} (blended): TIMEOUT`);
                  steeringApi.cancelTask(taskResponse.task_id).catch(() => {});
                },
              );
              // Adapt to the comparison shape the batch UI renders.
              response = combinedToComparison(combined, clusterContextAtStart);
            } else {
              const request: SteeringComparisonRequest = {
                sae_id: selectedSAE.id,
                prompt,
                // λ applies in batch Compare too (parity with Blended).
                selected_features: applyIntensity(selectedFeatures, get().intensity, !!get().clusterBudget),
                generation_params: generationParams,
                include_unsteered: includeUnsteered,
                compute_metrics: computeMetrics,
              };

              if (advancedParams) {
                request.advanced_params = advancedParams;
              }

              // Submit async task to Celery worker
              const taskResponse = await steeringApi.submitAsyncComparison(request);
              console.log(`[SteeringStore] Batch prompt ${i + 1}: async task submitted:`, taskResponse.task_id);

              // Store the task ID for this batch item - this triggers WebSocket subscription
              set({ taskId: taskResponse.task_id });

              // Brief delay to ensure React re-renders and WebSocket hook subscribes
              // The hook watches taskId and subscribes when it changes
              await new Promise(resolve => setTimeout(resolve, 50));

              // Create a promise that will be resolved by WebSocket events
              // handleAsyncCompleted/handleAsyncFailed will resolve/reject this
              // Includes automatic timeout cleanup to prevent memory leaks
              console.log(`[SteeringStore] Batch prompt ${i + 1}: creating resolver for task ${taskResponse.task_id}`);
              response = await createBatchResolver(
                taskResponse.task_id,
                PROMPT_TIMEOUT_MS,
                () => {
                  // Timeout callback - cancel the Celery task
                  console.log(`[SteeringStore] Batch prompt ${i + 1}: TIMEOUT for task ${taskResponse.task_id}`);
                  steeringApi.cancelTask(taskResponse.task_id).catch(() => {});
                }
              );
            }

            console.log(`[SteeringStore] Batch prompt ${i + 1}: resolver returned, updating state`);

            // Mark this prompt as completed
            set((state) => ({
              batchState: state.batchState
                ? {
                    ...state.batchState,
                    results: state.batchState.results.map((r, idx) =>
                      idx === i ? { ...r, status: 'completed', comparison: response } : r
                    ),
                  }
                : null,
            }));
            console.log(`[SteeringStore] Batch prompt ${i + 1}: iteration complete, continuing to next`);
          } catch (error) {
            const errorMessage = error instanceof Error ? error.message : 'Failed to generate';
            console.log(`[SteeringStore] Batch prompt ${i + 1}: failed:`, errorMessage);

            // Clean up the pending resolver properly (clears timeout). Clear
            // both — Blended uses the combined resolver, Compare the batch one.
            cleanupBatchResolver();
            cleanupCombinedResolver();

            // Mark this prompt as failed but continue with others
            set((state) => ({
              batchState: state.batchState
                ? {
                    ...state.batchState,
                    results: state.batchState.results.map((r, idx) =>
                      idx === i ? { ...r, status: 'failed', error: errorMessage } : r
                    ),
                  }
                : null,
            }));
            console.log(`[SteeringStore] Batch prompt ${i + 1}: marked as failed, continuing to next`);
          }
          console.log(`[SteeringStore] Batch iteration ${i + 1} end, looping...`);
        }

        // Batch complete - ensure resolvers are cleaned up (both kinds)
        console.log(`[SteeringStore] Batch loop finished, cleaning up`);
        cleanupBatchResolver();
        cleanupCombinedResolver();

        set((state) => ({
          isGenerating: false,
          progress: 100,
          progressMessage: 'Batch complete',
          batchState: state.batchState
            ? { ...state.batchState, isRunning: false }
            : null,
        }));
        console.log(`[SteeringStore] Batch complete!`);
      },

      // Abort batch processing (cancels current Celery task and stops processing)
      abortBatch: async () => {
        const { taskId } = get();

        // Set aborted flag first to prevent next prompt from starting
        set((state) => ({
          batchState: state.batchState
            ? { ...state.batchState, aborted: true }
            : null,
        }));

        // Reject the pending promise if any, with proper cleanup. In Compare
        // batch this is the batch resolver; in Blended batch it's the combined
        // resolver — reject whichever is outstanding so Stop Batch always frees
        // the loop.
        if (pendingBatchResolver) {
          console.log('[SteeringStore] Rejecting batch promise due to abort');
          const resolver = pendingBatchResolver;
          cleanupBatchResolver();  // Clear timeout and null resolver before rejecting
          resolver.reject(new Error('Batch aborted by user'));
        }
        cleanupCombinedResolver('Batch aborted by user');

        // Cancel the current Celery task if any
        if (taskId) {
          try {
            console.log('[SteeringStore] Cancelling Celery task:', taskId);
            await steeringApi.cancelTask(taskId);
          } catch (error) {
            console.warn('[SteeringStore] Failed to cancel task:', error);
          }
        }

        set({
          isGenerating: false,
          taskId: null,
          progress: 0,
          progressMessage: 'Batch aborted',
        });
      },

      // Clear batch results
      clearBatchResults: () => {
        set({
          batchState: null,
          progress: 0,
          progressMessage: null,
        });
      },

      // Run strength sweep (uses first prompt) - now uses async Celery-based API
      runStrengthSweep: async (featureIdx: number, layer: number, strengthValues: number[]) => {
        const { selectedSAE, prompts, generationParams } = get();
        const prompt = prompts[0] || '';

        if (!selectedSAE) {
          throw new Error('No SAE selected');
        }
        if (!prompt.trim()) {
          throw new Error('Prompt is required');
        }

        set({
          isSweeping: true,
          sweepResults: null,
          error: null,
          taskId: null,
        });

        try {
          const request: SteeringStrengthSweepRequest = {
            sae_id: selectedSAE.id,
            prompt,
            feature_idx: featureIdx,
            layer,
            strength_values: strengthValues,
            generation_params: generationParams,
          };

          // Submit async task to Celery worker
          const taskResponse = await steeringApi.submitAsyncSweep(request);
          console.log('[SteeringStore] Sweep async task submitted:', taskResponse.task_id);

          set({ taskId: taskResponse.task_id });

          // Brief delay to ensure React re-renders and WebSocket hook subscribes
          await new Promise(resolve => setTimeout(resolve, 50));

          // Create sweep resolver - WebSocket events will resolve/reject this
          // handleAsyncCompleted/handleAsyncFailed handle sweep via the resolver
          const SWEEP_TIMEOUT_MS = 5 * 60 * 1000;
          const sweepResult = await createSweepResolver(
            taskResponse.task_id,
            SWEEP_TIMEOUT_MS,
            () => {
              // Timeout callback - cancel the Celery task
              steeringApi.cancelTask(taskResponse.task_id).catch(() => {});
            }
          );

          console.log('[SteeringStore] Sweep completed via WebSocket');

          // Update state with results
          set({
            sweepResults: sweepResult,
            isSweeping: false,
            taskId: null,
          });

          return sweepResult;
        } catch (error) {
          const errorMessage = error instanceof Error ? error.message : 'Failed to run strength sweep';
          console.log('[SteeringStore] Sweep failed:', errorMessage);

          // Clean up sweep resolver properly
          cleanupSweepResolver();

          set({
            error: errorMessage,
            isSweeping: false,
            taskId: null,
          });
          throw error;
        }
      },

      // Clear sweep results
      clearSweepResults: () => {
        set({ sweepResults: null });
      },

      // Set combined mode (all features applied together vs. individual outputs)
      setCombinedMode: (enabled: boolean) => {
        set({ combinedMode: enabled });
      },

      // Generate with combined mode (all features applied together)
      generateCombined: async (includeBaseline = true, computeMetrics = true) => {
        const { selectedSAE, selectedFeatures, prompts, generationParams, advancedParams } = get();

        if (!selectedSAE) {
          throw new Error('No SAE selected');
        }

        if (selectedFeatures.length === 0) {
          throw new Error('No features selected');
        }

        // Use first prompt for combined mode (single prompt only)
        const prompt = prompts[0]?.trim();
        if (!prompt) {
          throw new Error('No prompt provided');
        }

        set({
          isCombinedGenerating: true,
          error: null,
          progress: 0,
          progressMessage: 'Submitting combined steering request...',
        });

        try {
          const request: CombinedSteeringRequest = {
            sae_id: selectedSAE.id,
            prompt,
            selected_features: applyIntensity(selectedFeatures, get().intensity, !!get().clusterBudget),
            generation_params: generationParams,
            advanced_params: advancedParams ?? undefined,
            include_baseline: includeBaseline,
            compute_metrics: computeMetrics,
          };

          // Snapshot provenance at submit: the result must be titled by the
          // context it was GENERATED under, not whatever is live at completion.
          const ctxAtSubmit = get().clusterContext;

          // Submit async task
          const taskResponse = await steeringApi.submitAsyncCombined(request);
          set({ taskId: taskResponse.task_id });

          console.log(`[SteeringStore] Combined task submitted: ${taskResponse.task_id}`);

          // Create resolver for WebSocket completion.
          // Kept below the backend combined task's hard limit (240s) so the
          // worker never outlives the client. With the KV cache retained for
          // hook-compatible models a combined run is seconds; this is headroom
          // for a cold start + long generation at up to 20 features.
          const COMBINED_TIMEOUT_MS = 230000; // ~3.8 min (backend SIGKILL at 240s)
          const result = await createCombinedResolver(
            taskResponse.task_id,
            COMBINED_TIMEOUT_MS,
            () => {
              set({
                isCombinedGenerating: false,
                error: 'Combined generation timed out. Try again or reduce parameters.',
              });
            }
          );

          set({
            combinedResults: result,
            combinedResultsTitle: blendedTitle(
              ctxAtSubmit,
              result.features_applied.length,
              result.features_applied.length === 1
                ? result.features_applied[0].label || `Feature #${result.features_applied[0].feature_idx}`
                : undefined,
            ),
            isCombinedGenerating: false,
            progress: 100,
            progressMessage: 'Complete',
          });

          return result;

        } catch (error) {
          const errorMessage = error instanceof Error ? error.message : 'Combined generation failed';
          set({
            error: errorMessage,
            isCombinedGenerating: false,
            progress: 0,
            progressMessage: null,
          });
          throw error;
        }
      },

      // Clear combined results
      clearCombinedResults: () => {
        set({ combinedResults: null, combinedResultsTitle: null });
      },

      // Update progress (called by WebSocket)
      updateProgress: (update: SteeringProgressUpdate) => {
        const { comparisonId } = get();
        if (update.comparison_id !== comparisonId) return;

        set({
          progress: update.progress,
          progressMessage: update.message,
        });

        // If completed or failed, update state
        if (update.status === 'completed' || update.status === 'failed') {
          set({ isGenerating: false });
        }
      },

      // Fetch experiments
      fetchExperiments: async (params?: { skip?: number; limit?: number; search?: string; sae_id?: string }) => {
        set({ experimentsLoading: true, error: null });
        try {
          const response = await steeringApi.getExperiments(params);
          set({
            experiments: response.data,
            experimentsPagination: {
              skip: response.pagination.skip,
              limit: response.pagination.limit,
              total: response.pagination.total,
              hasMore: response.pagination.has_more,
            },
            experimentsLoading: false,
          });
        } catch (error) {
          const errorMessage = error instanceof Error ? error.message : 'Failed to fetch experiments';
          set({ error: errorMessage, experimentsLoading: false });
        }
      },

      // Save current comparison as experiment
      saveExperiment: async (name: string, description?: string, tags?: string[]) => {
        const { comparisonId, currentComparison } = get();
        if (!comparisonId) {
          throw new Error('No comparison to save');
        }
        if (!currentComparison) {
          throw new Error('No comparison result to save');
        }

        try {
          const experiment = await steeringApi.saveExperiment({
            name,
            description,
            comparison_id: comparisonId,
            tags,
            result: currentComparison,
          });

          set((state) => ({
            experiments: [experiment, ...state.experiments],
            experimentsPagination: {
              ...state.experimentsPagination,
              total: state.experimentsPagination.total + 1,
            },
          }));

          return experiment;
        } catch (error) {
          const errorMessage = error instanceof Error ? error.message : 'Failed to save experiment';
          set({ error: errorMessage });
          throw error;
        }
      },

      // Load an experiment
      loadExperiment: (experiment: SteeringExperiment) => {
        set({
          selectedFeatures: experiment.selected_features,
          prompts: [experiment.prompt], // Convert single prompt to array
          generationParams: experiment.generation_params,
          currentComparison: experiment.results,
          comparisonId: experiment.results.comparison_id,
          batchState: null, // Clear batch state when loading experiment
          clusterContext: null, // selection replaced wholesale (Feature 012)
          activeProfile: null,
          clusterBudget: null,
          clusterNotice: null,
          combinedResults: null,
          combinedResultsTitle: null,
        });
      },

      // Delete an experiment
      deleteExperiment: async (id: string) => {
        try {
          await steeringApi.deleteExperiment(id);
          set((state) => ({
            experiments: state.experiments.filter((e) => e.id !== id),
            experimentsPagination: {
              ...state.experimentsPagination,
              total: state.experimentsPagination.total - 1,
            },
          }));
        } catch (error) {
          const errorMessage = error instanceof Error ? error.message : 'Failed to delete experiment';
          set({ error: errorMessage });
          throw error;
        }
      },

      // Delete multiple experiments
      deleteExperimentsBatch: async (ids: string[]) => {
        try {
          await steeringApi.deleteExperimentsBatch(ids);
          set((state) => ({
            experiments: state.experiments.filter((e) => !ids.includes(e.id)),
            experimentsPagination: {
              ...state.experimentsPagination,
              total: state.experimentsPagination.total - ids.length,
            },
          }));
        } catch (error) {
          const errorMessage = error instanceof Error ? error.message : 'Failed to delete experiments';
          set({ error: errorMessage });
          throw error;
        }
      },

      // Set error
      setError: (error: string | null) => {
        set({ error });
      },

      // Clear error
      clearError: () => {
        set({ error: null });
      },

      // Two-tier reset: First click clears results, second click clears everything
      // Track last reset time to detect double-click
      resetSession: () => {
        const now = Date.now();
        const lastReset = (window as unknown as { _steeringLastReset?: number })._steeringLastReset || 0;
        const timeSinceLastReset = now - lastReset;

        // If clicked within 3 seconds, do full reset
        if (timeSinceLastReset < 3000 && lastReset > 0) {
          // Full reset - clear everything including localStorage
          localStorage.removeItem('miStudio-steering');

          set({
            selectedSAE: null,
            selectedFeatures: [],
            clusterContext: null,
            activeProfile: null,
            clusterBudget: null,
            clusterNotice: null,
            intensity: 1,
            combinedResults: null,
            combinedResultsTitle: null,
            prompts: [''],
            generationParams: { ...DEFAULT_GENERATION_PARAMS },
            advancedParams: null,
            showAdvancedParams: false,
            isGenerating: false,
            comparisonId: null,
            taskId: null,
            progress: 0,
            progressMessage: null,
            currentComparison: null,
            recentComparisons: [],
            batchState: null,
            isSweeping: false,
            sweepResults: null,
            error: null,
          });

          (window as unknown as { _steeringLastReset?: number })._steeringLastReset = 0;
          console.log('[SteeringStore] FULL session reset complete (second click)');
        } else {
          // First click - only clear results, keep config
          set({
            isGenerating: false,
            comparisonId: null,
            taskId: null,
            progress: 0,
            progressMessage: null,
            currentComparison: null,
            batchState: null,
            isSweeping: false,
            sweepResults: null,
            combinedResults: null,
            combinedResultsTitle: null,
            error: null,
          });

          (window as unknown as { _steeringLastReset?: number })._steeringLastReset = now;
          console.log('[SteeringStore] Results reset (first click) - click again within 3s to reset all');
        }
      },

      // Hydration tracking for persistence
      setHasHydrated: (hydrated: boolean) => {
        set({ _hasHydrated: hydrated });
      },

      // Recover active task after page refresh
      recoverActiveTask: async () => {
        const { taskId, batchState, isGenerating, recentComparisons } = get();

        // If no task ID at all, nothing to recover
        if (!taskId) {
          console.log('[SteeringStore] No task ID to recover');
          return;
        }

        // Check if we already have this task in recent comparisons
        // If so, no need to recover from backend
        const alreadyInRecent = recentComparisons.some(r => r.id.includes(taskId.substring(0, 8)));
        if (alreadyInRecent && !isGenerating) {
          console.log('[SteeringStore] Task already in recent comparisons');
          return;
        }

        // If task was generating OR we have a taskId but empty recents, try to recover
        if (!isGenerating && recentComparisons.length > 0) {
          console.log('[SteeringStore] Not generating and have recents, skipping recovery');
          return;
        }

        console.log('[SteeringStore] Recovering active task:', taskId);

        try {
          // Check task status from backend
          const result = await steeringApi.getTaskResult(taskId);
          console.log('[SteeringStore] Task status:', result.status.status);

          if (result.status.status === 'success' && result.result) {
            // Task completed while we were away - load results
            console.log('[SteeringStore] Task completed - loading results');
            // Feature 012/QA guard: a recovered COMBINED result has no `steered`
            // array — casting it into currentComparison crashes ComparisonResults.
            const rawResult = result.result as unknown as Record<string, unknown>;
            if (rawResult && 'combined_id' in rawResult) {
              console.log('[SteeringStore] Recovered a combined result; restoring combinedResults instead');
              const combined = rawResult as unknown as CombinedSteeringResponse;
              set({
                combinedResults: combined,
                combinedResultsTitle: blendedTitle(
                  null,
                  combined.features_applied.length,
                  combined.features_applied.length === 1
                    ? combined.features_applied[0].label || `Feature #${combined.features_applied[0].feature_idx}`
                    : undefined,
                ),
                isGenerating: false,
                isCombinedGenerating: false,
                taskId: null,
                progress: 100,
                progressMessage: 'Recovered combined result',
              });
              return;
            }
            const comparison = rawResult as unknown as SteeringComparisonResponse;

            // Add to recent comparisons for persistence
            const { recentComparisons: currentRecent } = get();
            const newRecent = {
              id: comparison.comparison_id,
              prompt: comparison.prompt,
              timestamp: comparison.created_at || new Date().toISOString(),
              result: comparison,
            };
            const updatedRecent = [newRecent, ...currentRecent.filter(r => r.id !== comparison.comparison_id)].slice(0, 10);

            if (batchState) {
              // In batch mode - update the current result
              set((state) => ({
                isGenerating: false,
                progress: 100,
                progressMessage: 'Recovered completed result',
                recentComparisons: updatedRecent,
                batchState: state.batchState
                  ? {
                      ...state.batchState,
                      isRunning: false,
                      results: state.batchState.results.map((r, idx) =>
                        idx === state.batchState!.currentIndex
                          ? { ...r, status: 'completed', comparison }
                          : r
                      ),
                    }
                  : null,
              }));
            } else {
              // Single prompt mode
              set({
                isGenerating: false,
                currentComparison: comparison,
                comparisonId: comparison.comparison_id,
                progress: 100,
                progressMessage: 'Recovered completed result',
                recentComparisons: updatedRecent,
              });
            }
          } else if (result.status.status === 'failure') {
            // Task failed while we were away
            console.log('[SteeringStore] Task failed:', result.status.error);
            set({
              isGenerating: false,
              error: result.status.error || 'Task failed',
              progress: 0,
              progressMessage: null,
            });
          } else {
            // Task still running - WebSocket hook will handle progress
            console.log('[SteeringStore] Task still running - WebSocket will take over');
            set({
              progress: result.status.percent || 0,
              progressMessage: result.status.message || 'Reconnected to running task...',
            });
          }
        } catch (error) {
          console.error('[SteeringStore] Failed to recover task:', error);
          // Clear stale task state
          set({
            isGenerating: false,
            taskId: null,
            progress: 0,
            progressMessage: null,
            error: 'Failed to recover task - it may have expired',
          });
        }
      },
      }),
      {
        name: 'miStudio-steering',
        // Only persist essential state for recovery
        partialize: (state) => ({
          // Active task state (for recovery after refresh)
          taskId: state.taskId,
          isGenerating: state.isGenerating,
          batchState: state.batchState,
          progress: state.progress,
          progressMessage: state.progressMessage,
          // Configuration state (so user doesn't lose their setup)
          selectedSAE: state.selectedSAE,
          // clusterBudget/intensity are session-only, so rehydrated features
          // must not claim a governing budget: unpin and demote 'cluster' to
          // 'manual' (strength values themselves are kept).
          selectedFeatures: state.selectedFeatures.map((f) => ({
            ...f,
            pinned: false,
            strengthSource: f.strengthSource === 'cluster' ? ('manual' as const) : f.strengthSource,
          })),
          prompts: state.prompts,
          generationParams: state.generationParams,
          advancedParams: state.advancedParams,
          showAdvancedParams: state.showAdvancedParams,
          // Recent comparisons (for loading past results)
          recentComparisons: state.recentComparisons,
        }),
        onRehydrateStorage: () => (state) => {
          state?.setHasHydrated(true);
          console.log('[SteeringStore] State hydrated from localStorage');
        },
      }
    ),
    {
      name: 'SteeringStore',
    }
  )
);

// Selector for checking if ready to generate (uses first prompt)
export const selectCanGenerate = (state: SteeringState) =>
  state.selectedSAE !== null &&
  state.selectedFeatures.length > 0 &&
  (state.prompts[0] || '').trim().length > 0 &&
  !state.isGenerating;

// Selector for checking if ready to generate batch (at least one non-empty prompt)
export const selectCanGenerateBatch = (state: SteeringState) =>
  state.selectedSAE !== null &&
  state.selectedFeatures.length > 0 &&
  state.prompts.some((p) => p.trim().length > 0) &&
  !state.isGenerating &&
  !state.batchState?.isRunning;

// Selector for feature by instance_id
export const selectFeatureByInstanceId = (instanceId: string) => (state: SteeringState) =>
  state.selectedFeatures.find((f) => f.instance_id === instanceId);

// Selector for feature by index and layer (finds first match - use selectFeatureByInstanceId for exact match)
export const selectFeature = (featureIdx: number, layer: number) => (state: SteeringState) =>
  state.selectedFeatures.find((f) => f.feature_idx === featureIdx && f.layer === layer);

// Selector for available colors
export const selectAvailableColors = (state: SteeringState): FeatureColor[] => {
  const usedColors = state.selectedFeatures.map((f) => f.color);
  return FEATURE_COLOR_ORDER.filter((c) => !usedColors.includes(c));
};
