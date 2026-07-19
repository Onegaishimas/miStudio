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

export interface Circuit {
  id: string;
  name: string;
  narrative: string | null;
  granularity: 'feature' | 'cluster';
  saes: { mistudio_sae_id?: string; layer?: number }[];
  members: CircuitMember[];
  edges: CircuitEdge[];
  budget: Record<string, unknown> | null;
  faithfulness: { necessity?: number; sufficiency?: number } | null;
  rung: EvidenceRung;
  rung_language: string;
  rung_next_step: string;
  promoted: boolean;
  created_at: string;
  updated_at: string;
}
