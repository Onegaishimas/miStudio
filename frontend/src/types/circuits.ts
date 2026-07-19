/**
 * Circuit types (Feature 018) — mirrors the /circuits API. Rung language is
 * SERVER-rendered (rung_language / rung_next_step): the UI must display those
 * strings verbatim and never compose its own causal phrasing (IDL-35).
 */

import { EvidenceRung } from './evidenceLadder';

export type EdgeType = 'computed' | 'persistence' | 'attention_mediated';

export interface CircuitEdge {
  up: { layer: number; kind?: string; feature_idx?: number; cluster_profile_id?: string };
  down: { layer: number; kind?: string; feature_idx?: number; cluster_profile_id?: string };
  type: EdgeType;
  type_signals?: Record<string, unknown> | null;
  rung: EvidenceRung;
  tested_and_failed: EvidenceRung[];
  coactivation?: { pmi?: number; lift?: number; support?: number; null_percentile?: number;
                   replicated_heldout?: boolean } | null;
  weight_prior?: number | null;
  attribution?: { score?: number; sign_consistency?: number; method?: string } | null;
  validation_manifest_ref?: string | null;
  effect_size?: number | null;
}

export interface CircuitMember {
  layer: number;
  member_kind: 'feature_ref' | 'cluster_ref';
  feature?: { feature_idx: number; label?: string | null; strength: number } | null;
  cluster_profile_id?: string | null;
  cluster_name?: string | null;
  expanded_members?: { feature_idx: number; label?: string | null; strength: number }[] | null;
}

export interface CircuitSummary {
  id: string;
  name: string;
  granularity: 'feature' | 'cluster';
  layers: number[];
  member_count: number;
  edge_count: number;
  rung: EvidenceRung;
  rung_language: string;
  rung_next_step: string;
  promoted: boolean;
  model_id: string | null;
  updated_at: string;
}

export interface Circuit extends CircuitSummary {
  name: string;
  narrative: string | null;
  granularity: 'feature' | 'cluster';
  saes: { mistudio_sae_id?: string; layer?: number }[];
  members: CircuitMember[];
  edges: CircuitEdge[];
  budget: Record<string, unknown> | null;
  faithfulness: { necessity?: number; sufficiency?: number } | null;
  discovery: Record<string, unknown> | null;
  created_at: string;
}

// ── Feature 016: Capture ──────────────────────────────────────────────────

/** A cost estimate returned by a probe run (confirm=false). */
export interface CaptureEstimate {
  events?: number;
  bytes?: number;
  minutes?: number;
  [k: string]: unknown;
}

export interface CaptureAttentionConfig {
  layers: number[];
  heads?: number[] | null;
  top_k: number;
}

export interface CircuitCapture {
  id: string;
  status: string;
  progress: number | null;
  error_message?: string | null;
  corpus: unknown;
  model_id: string | null;
  layers: { layer: number; sae_id: string }[] | null;
  split: { heldout_count: number; [k: string]: unknown } | null;
  estimate: CaptureEstimate | null;
  attention_capture: CaptureAttentionConfig | null;
  counts: Record<string, unknown> | null;
  bytes: number | null;
  events_total: number | null;
  stale: boolean;
  created_at: string;
  updated_at: string;
}

export interface CircuitCaptureCreate {
  dataset_id: string;
  model_id?: string;
  layers: { layer: number; sae_id: string }[];
  epsilon?: number;
  theta_floor?: number;
  sample_cap?: number;
  split_seed?: number;
  attention_capture?: CaptureAttentionConfig;
  confirm: boolean;
}

// ── Feature 016: Discovery ────────────────────────────────────────────────

export type DiscoveryGranularity = 'feature' | 'cluster';
export type DiscoveryMode = 'seeded' | 'open';

/** A seed reference — either a feature index or a cluster profile, per layer. */
export interface DiscoverySeedRef {
  layer: number;
  feature_idx?: number;
  cluster_profile_id?: string;
}

export interface DiscoveryCreate {
  capture_run_id: string;
  granularity: DiscoveryGranularity;
  mode: DiscoveryMode;
  seed_refs?: DiscoverySeedRef[];
  s_min?: number;
  null_shuffles?: number;
  null_percentile?: number;
  fdr_q?: number;
  cohesion_floor?: number;
  seed?: number;
  force?: boolean;
}

/** A candidate node ref — feature index or cluster profile. */
export interface DiscoveryNodeRef {
  layer: number;
  feature_idx?: number;
  cluster_profile_id?: string;
  cluster_name?: string;
}

export interface DiscoveryCandidate {
  up: DiscoveryNodeRef;
  down: DiscoveryNodeRef;
  granularity: DiscoveryGranularity;
  stats: {
    pmi?: number;
    lift?: number;
    support?: number;
    spearman?: number;
    null_pct?: number;
    p_value?: number;
    pooled_p?: number;
  };
  replicated_heldout: boolean;
  attribution?: {
    score?: number;
    sign_consistency?: number;
    method?: string;
    rung1_gate?: boolean;
  } | null;
  orderings?: { coact_rank?: number; attr_rank?: number } | null;
}

export interface DiscoveryReport {
  granularity: DiscoveryGranularity;
  mode: DiscoveryMode;
  supernode_activation?: string;
  lag0_disclosure: string;
  null_summary: { method: string; shuffles: number; percentile: number };
  fdr: {
    discipline: string;
    p_source: string;
    q: number;
    tested: number;
    passed: number;
  };
  replication: { tested: number; replicated: number; rate: number | null };
  counts_by_stage: {
    pairs_considered: number;
    post_support: number;
    null_tested: number;
    post_fdr: number;
    candidates_persisted: number;
  };
  caps: {
    candidates_truncated: boolean;
    unit_cap_hit_layers: number[];
    null_cap_hit: boolean;
    [k: string]: unknown;
  };
  uncovered_seeds: { layer: number; ref: unknown; reason: string }[];
  attribution: Record<string, unknown> | null;
  uplift: null;
  wall_clock_seconds?: number;
}

export interface DiscoveryRun {
  id: string;
  capture_run_id: string;
  status: string;
  progress: number | null;
  error_message?: string | null;
  params: Record<string, unknown> | null;
  report: DiscoveryReport | null;
  candidate_count: number;
  candidates?: DiscoveryCandidate[];
  created_at: string;
  updated_at: string;
}
