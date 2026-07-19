---
sidebar_position: 10
title: "Circuits"
description: "Discover cross-layer feature circuits — capture, statistically mine, and attribution-rank candidates"
---

# Circuits — Discovering Cross-Layer Structure

A cluster names features that mean the same thing at one layer. A **circuit** names how features at *different* layers work together — an upstream feature that drives a downstream one, across the model's depth. miStudio can now **discover** these, not just host hand-built ones.

Discovery is deliberately honest about what it has and hasn't proven. It produces **candidates** with disclosed statistics; it never dresses an association up as a proven mechanism. The evidence gets stronger in stages (the [evidence ladder](/concepts/evidence-ladder)), and every surface tells you which rung a circuit is on.

## The loop

1. **Capture** the per-token, multi-layer feature activations over an evaluation corpus.
2. **Discover** candidate edges by mining that store with statistically sound methods.
3. **Attribution-rank** the shortlist with a gradient pass before the expensive causal tier.
4. (Feature 017) **Validate** the top edges causally; (Feature 018) review, promote, and steer them.

## Capture

The **Circuits → Capture** tab records what discovery needs and today's extraction throws away: for every token, which features fired at each selected layer, **including the token's position** and the SAE's reconstruction-error norm.

Configure the layer set (each layer with the SAE trained on it), the evaluation dataset, a sparsity threshold, and a sample cap. Before committing GPU time you get a **cost estimate** — projected events, disk size, and wall-clock, extrapolated from a small probe batch. Confirm to launch the managed capture (progress, cancel, guardrails).

The store is built **Tier-2.5-ready**: token positions are first-class and an optional attention sidecar can be captured, so [attention-mediated discovery](/concepts/tier-2.5) is a future implementation, never a re-capture. At capture time the corpus is split **80/20 into discovery and held-out partitions** (seeded and recorded) — every downstream run uses the same split, so replication is measured, not assumed.

Capture stores are listed, reusable, and deletable. If a referenced SAE changes, the store is **flagged stale** (not deleted) — mining a stale store requires an explicit override.

## Discovery

The **Circuits → Discovery** tab mines a completed capture store. Two choices shape the run:

- **Granularity** — **feature-level** (individual features) or **cluster-level supernodes** (your curated clusters as single units; a supernode fires wherever any member fires). Cluster granularity is the recommended default for seeded runs.
- **Mode** — **seeded** (mine the upstream/downstream partners of chosen features or clusters) or **open-corpus** (mine broadly). Both are first-class.

### What makes a candidate trustworthy

Naive co-occurrence counting is worse than nothing — it surfaces whatever is frequent. Discovery instead:

- ranks by **PMI** (pointwise mutual information / lift), not raw counts, so two independently-frequent features do **not** rise to the top;
- requires a **minimum support** before a pair is ranked at all;
- tests each pair against a **null model** — a within-document circular shift that preserves each feature's firing rate and burstiness while destroying its alignment with the partner;
- applies **Benjamini–Hochberg FDR** for multiple comparisons;
- **re-tests survivors on the held-out partition** and reports the **replication rate** as a first-class number.

The **run report**, pinned above the candidate list, is the trust surface: it states the null method, the FDR discipline, the held-out replication rate, the counts at each stage, and — crucially — **any caps that were hit** (unit caps, candidate truncation). Nothing is silently dropped.

:::note Lag-0 limitation
All co-activation here is **same-token-position**. That finds within-position computation; it does **not** find attention-mediated, cross-position circuits (a feature three sentences back driving one here). Every discovery surface says so, and [Tier-2.5](/concepts/tier-2.5) is the designed path to lift it.
:::

## Attribution pass

A discovery candidate is a statistical association. The **Run attribution pass** action spends one forward + one backward per prompt (with the SAE reconstruction error held as a stop-gradient constant) to compute a **gradient-attribution score** per candidate — a second, independent evidence signal. It **re-ranks** the shortlist so the causal validation tier (017) is spent on the most promising edges, and it records both orderings so 017 can later report whether the re-ranking actually raised the validation survival rate.

Attribution earns a candidate the **attribution-supported** rung — one step up the ladder, and explicitly *not* a causal claim. That rung comes only from intervention (Feature 017).

## Doing it from an agent

The [MCP server](/advanced/mcp-server) exposes the whole loop in the `circuits` category: `start_circuit_capture`, `list_circuit_captures`, `run_circuit_discovery`, `get_discovery_results` (which returns the full report), and `run_attribution_pass`. An agent can capture, mine at either granularity, attribution-rank, and read the disciplined report without the UI — and the tool descriptions carry the same lag-0 and rung-language discipline the UI does.
