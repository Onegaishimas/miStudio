# Tier-2.5: Attention-Mediated Cross-Position Circuit Mining — Design

**Status:** Design deliverable of Feature 016 (BR-023). Implementation is a **named fast-follow** feature PRD, NOT part of the 016 increment.
**Related:** 016_FPRD §3.5 · BRD-MIS-CIRCUITS-002 BR-023 · IDL-32/IDL-36 · consumes the 016 capture store + `mistudio.circuit-definition/v1` contract fields (already present, populated-or-null).

---

## 0. Why this exists (the lag-0 limitation)

Feature 016 mines **lag-0** structure: it counts co-activation at the **same token position** (`(doc_id << 16) | token_pos` merge-join). That finds *computed* edges — an upstream feature's activation at token *t* correlating with a downstream feature's activation at the **same** token *t*, the signature of residual-stream computation within a position.

It cannot find the circuits users most want to name: **attention-mediated, cross-token-position computation** — an upstream feature firing at token *t_k* (a key) that drives a downstream feature at a *different* token *t_q* (a query) *because the model attended from t_q to t_k*. "The name mentioned three sentences ago → the pronoun-resolution feature here" is invisible to a same-position join.

This limitation is **disclosed everywhere** 016 results appear (UI Discovery tab notice, MCP tool docstrings, every run report's `lag0_disclosure` string). Tier-2.5 lifts it **without re-capturing**, because the store and the contract were built position- and attention-aware from day one.

## 1. What already exists (no migration required)

The 016 capture store and the v1 contract carry the Tier-2.5 join keys as first-class, populated-or-explicitly-null fields — enabling Tier-2.5 is turning on mining, not migrating artifacts:

| Substrate | Field | Where |
|---|---|---|
| Capture store | `token_pos` (u16, first-class sort column) | `layer_{L}.events` |
| Capture store | attention top-k sidecar `(doc, t_q, head, t_k, mass)` | `attn_l{L}.topk` (opt-in at capture; `attention_capture` manifest block) |
| Contract edge | `type: attention_mediated` (enum member, reserved) | `CircuitEdge.type` |
| Contract edge | `EdgePosition` fields (nullable q/k positions) | `CircuitEdge` position block |
| Contract edge | `mediating_heads` (Tier-2.5 evidence marker) | `CircuitEdge` (018 classifier keys `attention_mediated` off its non-null presence) |
| Discovery report | `lag0_disclosure` string | every run report |

A capture run that enabled `attention_capture` is Tier-2.5-mineable with zero recapture; a run that didn't needs one recapture with the toggle on (documented cost).

## 2. The five design points

### 2.1 Attention-join candidate generation

Replace the lag-0 same-position merge-join with an **attention-weighted cross-position join**. For a downstream unit *d* firing at query token *t_q*, the candidate upstream keys are not `{(doc, t_q)}` but the **attended-from** keys, weighted by attention mass:

```
for each downstream event (doc, t_q, feature=d):
    for (head, t_k, mass) in attn_topk[doc, t_q]:        # the sidecar
        for upstream unit u active at (doc, t_k):
            weighted_coactivation[u, d] += mass · a_u(t_k) · a_d(t_q)
```

The sidecar's top-k-per-query bound keeps this tractable (k≈4 keys × heads, not all-pairs). Candidate pairs are (u@t_k → d@t_q) with `EdgePosition` recording the representative (t_q, t_k) offset distribution and `mediating_heads` recording which heads carried the mass.

### 2.2 Shuffled-join null

The lag-0 null (within-document circular shift of a unit's positions, §016 stats) does not test the *attention join*. The Tier-2.5 null must hold the marginals of BOTH the activation streams AND the attention pattern fixed while destroying their alignment:

- **Shuffle the attention join, not the activations.** For each query token, permute its top-k key assignments across a within-document circular shift of the KEY positions (preserving per-query attention mass distribution and per-key activation counts, breaking only which key a query attends to). Recompute the weighted co-activation under K such shuffles → null distribution.
- This isolates "the model actually routes t_q→t_k AND the features co-fire there" from "the features are both just frequent." BH-FDR (pooled-standardized, as in 016) over the shuffled-join p-values.

### 2.3 Frozen-attention attribution

Tier-2 attribution (016) differentiates through the SAE pass-through with the model's own forward. For attention-mediated edges the attribution must credit the **path through the attention head**, which is nonlinear in general. The circuit-tracing remedy (frozen-attention linearization): **freeze the attention pattern** (detach the softmax weights to constants for the backward pass), making the query→key value-mixing linear, then run the same grouped-by-downstream `Σ_t ∂m/∂a_u(t_k) · a_u(t_k)` — now the gradient flows through the frozen head from the downstream query position to the upstream key position. Frozen-attention is disclosed as the linearization (it answers "given the model attended this way, how much did u@t_k drive d@t_q").

### 2.4 Position-restricted validation

017's intervention validation (suppress u, measure Δ in d) must, for an attention-mediated edge, **suppress at the key position t_k and measure at the query position t_q** — not same-position. The manifest records the (t_q, t_k) offsets so reproduction re-executes the exact cross-position intervention. The effect-size null is the support-matched shuffled-join non-edge (§2.2), position-restricted identically. Everything else in the 017 ladder (sign-consistency gate, tested_and_failed history, ES-vs-null) applies unchanged — only the intervention/measurement positions differ.

### 2.5 Contract fields already present

No contract amendment. Tier-2.5 populates fields that ship in v1 as nullable:
- `type = "attention_mediated"` (reserved enum member; 018's classifier already routes off `mediating_heads` presence).
- `EdgePosition` q/k offset block (null for computed/persistence edges).
- `mediating_heads` list (the heads §2.1 credited).
- `validation_manifest_ref` → a position-restricted manifest (§2.4).
The rung ladder, edge-type discipline, and export/import round-trip are unchanged: an attention-mediated edge is just an edge whose type is `attention_mediated` and whose position/head fields are non-null.

## 3. Sequencing & non-goals

- **Prerequisite:** capture runs with `attention_capture` enabled (already supported by 016; the sidecar writer + manifest block exist).
- **Depends on:** 017 (intervention primitives — position-restricted variants extend them) and 018 (the `attention_mediated` type + classifier + rung ladder — already shipped).
- **This is design only.** The mining implementation (attention-join candidate generator, shuffled-join null, frozen-attention attribution service, position-restricted validation) is the **named fast-follow feature PRD**, gated on this design and on demand for cross-position circuits.
- **Not Tier-3:** full attribution graphs with error nodes and transcoder substrates (BRD-MIS-CIRCUITS-002 §future / SUBSTRATE research track) remain out of scope.

---

*Authored as the BR-023 design deliverable of Feature 016, 2026-07-19. The capture store and v1 contract already carry every field this design consumes — Tier-2.5 is an implementation fast-follow, never a schema migration.*
