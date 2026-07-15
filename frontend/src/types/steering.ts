/**
 * Steering Types
 *
 * TypeScript interfaces for Feature Steering feature.
 * Matches backend Pydantic schemas in src/schemas/steering.py
 *
 * Backend API Contract:
 * - POST /api/v1/steering/compare - Generate steering comparison
 * - POST /api/v1/steering/sweep - Run strength sweep
 * - GET /api/v1/steering/experiments - List saved experiments
 * - POST /api/v1/steering/experiments - Save experiment
 * - GET /api/v1/steering/experiments/:id - Get experiment
 * - DELETE /api/v1/steering/experiments/:id - Delete experiment
 */

import type { StrengthSource } from '../utils/steeringStrength';

/**
 * Color options for selected features.
 * Up to 20 features (Feature 011). Colors are cosmetic — the original 4
 * (teal/blue/purple/amber) come first for continuity, then 16 more hues.
 * Uniqueness is no longer required; the backend dropped its unique-color check.
 */
export type FeatureColor =
  | 'teal'
  | 'blue'
  | 'purple'
  | 'amber'
  | 'rose'
  | 'cyan'
  | 'lime'
  | 'orange'
  | 'fuchsia'
  | 'sky'
  | 'emerald'
  | 'violet'
  | 'pink'
  | 'indigo'
  | 'yellow'
  | 'red'
  | 'green'
  | 'sapphire'
  | 'magenta'
  | 'gold';

/**
 * CSS classes for feature colors.
 *
 * IMPORTANT: every class string here must be a full literal — Tailwind purges
 * dynamically-built class names (`bg-${color}-500` would not survive the
 * production build). The last four names map to distinct Tailwind hues under
 * friendly aliases (sapphire→blue-600, magenta→pink-600, gold→amber-600) so all
 * 20 entries remain visually distinguishable.
 */
export const FEATURE_COLORS: Record<FeatureColor, {
  bg: string;
  border: string;
  text: string;
  light: string;
}> = {
  teal: { bg: 'bg-teal-500', border: 'border-teal-500', text: 'text-teal-400', light: 'bg-teal-500/10' },
  blue: { bg: 'bg-blue-500', border: 'border-blue-500', text: 'text-blue-400', light: 'bg-blue-500/10' },
  purple: { bg: 'bg-purple-500', border: 'border-purple-500', text: 'text-purple-400', light: 'bg-purple-500/10' },
  amber: { bg: 'bg-amber-500', border: 'border-amber-500', text: 'text-amber-400', light: 'bg-amber-500/10' },
  rose: { bg: 'bg-rose-500', border: 'border-rose-500', text: 'text-rose-400', light: 'bg-rose-500/10' },
  cyan: { bg: 'bg-cyan-500', border: 'border-cyan-500', text: 'text-cyan-400', light: 'bg-cyan-500/10' },
  lime: { bg: 'bg-lime-500', border: 'border-lime-500', text: 'text-lime-400', light: 'bg-lime-500/10' },
  orange: { bg: 'bg-orange-500', border: 'border-orange-500', text: 'text-orange-400', light: 'bg-orange-500/10' },
  fuchsia: { bg: 'bg-fuchsia-500', border: 'border-fuchsia-500', text: 'text-fuchsia-400', light: 'bg-fuchsia-500/10' },
  sky: { bg: 'bg-sky-500', border: 'border-sky-500', text: 'text-sky-400', light: 'bg-sky-500/10' },
  emerald: { bg: 'bg-emerald-500', border: 'border-emerald-500', text: 'text-emerald-400', light: 'bg-emerald-500/10' },
  violet: { bg: 'bg-violet-500', border: 'border-violet-500', text: 'text-violet-400', light: 'bg-violet-500/10' },
  pink: { bg: 'bg-pink-500', border: 'border-pink-500', text: 'text-pink-400', light: 'bg-pink-500/10' },
  indigo: { bg: 'bg-indigo-500', border: 'border-indigo-500', text: 'text-indigo-400', light: 'bg-indigo-500/10' },
  yellow: { bg: 'bg-yellow-500', border: 'border-yellow-500', text: 'text-yellow-400', light: 'bg-yellow-500/10' },
  red: { bg: 'bg-red-500', border: 'border-red-500', text: 'text-red-400', light: 'bg-red-500/10' },
  green: { bg: 'bg-green-500', border: 'border-green-500', text: 'text-green-400', light: 'bg-green-500/10' },
  sapphire: { bg: 'bg-blue-600', border: 'border-blue-600', text: 'text-blue-300', light: 'bg-blue-600/10' },
  magenta: { bg: 'bg-pink-600', border: 'border-pink-600', text: 'text-pink-300', light: 'bg-pink-600/10' },
  gold: { bg: 'bg-amber-600', border: 'border-amber-600', text: 'text-amber-300', light: 'bg-amber-600/10' },
};

/**
 * Available feature colors in order. Assigned round-robin; may repeat past 20.
 */
export const FEATURE_COLOR_ORDER: FeatureColor[] = [
  'teal', 'blue', 'purple', 'amber', 'rose', 'cyan', 'lime', 'orange', 'fuchsia', 'sky',
  'emerald', 'violet', 'pink', 'indigo', 'yellow', 'red', 'green', 'sapphire', 'magenta', 'gold',
];

/**
 * A feature selected for steering.
 */
export interface SelectedFeature {
  instance_id: string; // Unique identifier for this selection instance (allows duplicates of same feature)
  comparison_id?: string; // Links to the comparison job this instance was used in (set when comparison runs)
  feature_idx: number;
  layer: number;
  strength: number; // Raw coefficient (Neuronpedia-compatible: 0.07 subtle, 80 strong)
  additional_strengths?: number[]; // Up to 3 additional strengths for multi-strength testing
  label: string | null;
  color: FeatureColor;
  feature_id: string | null; // Database feature ID if extracted
  // Feature 011: stats carried for the frequency-based auto-baseline + tile display
  max_activation?: number | null;
  activation_frequency?: number | null;
  strengthSource?: StrengthSource; // 'auto' | 'default' | 'manual'
}

/**
 * Generation parameters for text generation.
 */
export interface GenerationParams {
  max_new_tokens: number;
  temperature: number;
  top_p: number;
  top_k: number;
  num_samples: number;
  seed?: number;
}

/**
 * Advanced generation parameters.
 */
export interface AdvancedGenerationParams {
  repetition_penalty: number;
  presence_penalty: number;
  frequency_penalty: number;
  do_sample: boolean;
  stop_sequences: string[];
}

/**
 * Default generation parameters.
 */
export const DEFAULT_GENERATION_PARAMS: GenerationParams = {
  max_new_tokens: 100,
  temperature: 0.7,
  top_p: 0.9,
  top_k: 50,
  num_samples: 1,
};

/**
 * Request to generate a steering comparison.
 */
export interface SteeringComparisonRequest {
  sae_id: string;
  model_id?: string;
  prompt: string;
  selected_features: SelectedFeature[];
  generation_params?: GenerationParams;
  advanced_params?: AdvancedGenerationParams;
  include_unsteered?: boolean;
  compute_metrics?: boolean;
}

/**
 * Request for strength sweep.
 */
export interface SteeringStrengthSweepRequest {
  sae_id: string;
  model_id?: string;
  prompt: string;
  feature_idx: number;
  layer: number;
  strength_values: number[];
  generation_params?: GenerationParams;
}

/**
 * Generation quality metrics.
 */
export interface GenerationMetrics {
  perplexity: number | null;
  coherence: number | null;
  behavioral_score: number | null;
  token_count: number;
  generation_time_ms: number;
}

/**
 * Single steered output.
 */
export interface SteeredOutput {
  text: string;
  feature_config: SelectedFeature;
  metrics: GenerationMetrics | null;
}

/**
 * Single strength result in multi-strength mode.
 */
export interface MultiStrengthResult {
  strength: number;
  text: string;
  metrics: GenerationMetrics | null;
}

/**
 * Multi-strength steered output (one per feature).
 */
export interface SteeredOutputMulti {
  feature_config: SelectedFeature;
  primary_result: MultiStrengthResult;
  additional_results: MultiStrengthResult[];
}

/**
 * Unsteered baseline output.
 */
export interface UnsteeredOutput {
  text: string;
  metrics: GenerationMetrics | null;
}

/**
 * Steering comparison response.
 */
export interface SteeringComparisonResponse {
  comparison_id: string;
  sae_id: string;
  model_id: string;
  prompt: string;
  unsteered: UnsteeredOutput | null;
  steered: SteeredOutput[];
  steered_multi?: SteeredOutputMulti[] | null; // Multi-strength results (when additional_strengths provided)
  metrics_summary: Record<string, any> | null;
  total_time_ms: number;
  created_at: string;
}

/**
 * Result for a single prompt in a batch operation.
 */
export interface BatchPromptResult {
  prompt: string;
  promptIndex: number;
  status: 'pending' | 'running' | 'completed' | 'failed';
  comparison: SteeringComparisonResponse | null;
  error: string | null;
}

/**
 * State for batch prompt processing.
 */
export interface BatchState {
  isRunning: boolean;
  currentIndex: number;
  totalPrompts: number;
  results: BatchPromptResult[];
  aborted: boolean;
}

/**
 * Single strength sweep result.
 */
export interface StrengthSweepResult {
  strength: number;
  text: string;
  metrics: GenerationMetrics | null;
}

/**
 * Strength sweep response.
 */
export interface StrengthSweepResponse {
  sweep_id: string;
  sae_id: string;
  model_id: string;
  prompt: string;
  feature_idx: number;
  layer: number;
  unsteered: UnsteeredOutput;
  results: StrengthSweepResult[];
  total_time_ms: number;
  created_at: string;
}

/**
 * Request to save a steering experiment.
 */
export interface SteeringExperimentSaveRequest {
  name: string;
  description?: string;
  comparison_id: string;
  tags?: string[];
  // Include the full result since comparisons are ephemeral (stored in Redis with TTL)
  result: SteeringComparisonResponse;
}

/**
 * Saved steering experiment.
 */
export interface SteeringExperiment {
  id: string;
  name: string;
  description: string | null;
  sae_id: string;
  model_id: string;
  prompt: string;
  selected_features: SelectedFeature[];
  generation_params: GenerationParams;
  results: SteeringComparisonResponse;
  tags: string[];
  created_at: string;
  updated_at: string;
}

/**
 * Paginated list of experiments.
 */
export interface SteeringExperimentListResponse {
  data: SteeringExperiment[];
  pagination: {
    skip: number;
    limit: number;
    total: number;
    has_more: boolean;
  };
}

/**
 * Real-time steering progress update (WebSocket).
 */
export interface SteeringProgressUpdate {
  comparison_id: string;
  status: string;
  current_config: string | null;
  progress: number;
  message: string | null;
}

/**
 * Feature activation analysis.
 */
export interface FeatureActivationAnalysis {
  feature_idx: number;
  activation_count: number;
  mean_activation: number;
  max_activation: number;
  activated_tokens: string[];
}

/**
 * Steering effect analysis (side effects).
 */
export interface SteeringEffectAnalysis {
  target_feature_idx: number;
  target_feature_activation_change: number;
  side_effects: FeatureActivationAnalysis[];
}

/**
 * Warning thresholds for steering strength (raw coefficients).
 * These are Neuronpedia-compatible values.
 */
export const STRENGTH_THRESHOLDS = {
  CAUTION_LOW: -50,
  CAUTION_HIGH: 100,
  EXTREME_LOW: -100,
  EXTREME_HIGH: 151,  // >150 is red, 150 is amber
};

/**
 * Get warning level for a strength value.
 */
export function getStrengthWarningLevel(strength: number): 'normal' | 'caution' | 'extreme' {
  if (strength <= STRENGTH_THRESHOLDS.EXTREME_LOW || strength >= STRENGTH_THRESHOLDS.EXTREME_HIGH) {
    return 'extreme';
  }
  if (strength <= STRENGTH_THRESHOLDS.CAUTION_LOW || strength >= STRENGTH_THRESHOLDS.CAUTION_HIGH) {
    return 'caution';
  }
  return 'normal';
}

/**
 * Calculate the multiplier from strength.
 * Neuronpedia-compatible: strength IS the raw coefficient.
 * Formula: multiplier = 1 + strength (coefficient = strength)
 */
export function strengthToMultiplier(strength: number): number {
  return 1 + strength;
}


// ============================================================================
// Combined Multi-Feature Steering Types
// ============================================================================

/**
 * Request to generate combined multi-feature steering output.
 * Applies ALL selected features simultaneously in a single generation pass.
 */
export interface CombinedSteeringRequest {
  sae_id: string;
  model_id?: string;
  prompt: string;
  selected_features: SelectedFeature[];
  generation_params?: GenerationParams;
  advanced_params?: AdvancedGenerationParams;
  include_baseline?: boolean;
  compute_metrics?: boolean;
}

/**
 * Feature applied in combined steering mode.
 */
export interface CombinedFeatureApplied {
  feature_idx: number;
  layer: number;
  strength: number;
  label: string | null;
  color: FeatureColor;
}

/**
 * Response from combined multi-feature steering.
 * Contains a single output where all features were applied together.
 */
export interface CombinedSteeringResponse {
  combined_id: string;
  sae_id: string;
  model_id: string;
  prompt: string;
  combined_output: string;
  features_applied: CombinedFeatureApplied[];
  baseline_output: string | null;
  combined_metrics: GenerationMetrics | null;
  baseline_metrics: GenerationMetrics | null;
  total_steering_strength: number;
  total_time_ms: number;
  created_at: string;
}
