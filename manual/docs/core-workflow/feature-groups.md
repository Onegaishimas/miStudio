---
sidebar_position: 5
title: "Feature Groups"
description: "Cross-feature analysis — browse features that share a top activating token"
---

# Feature Groups — Finding Concept Clusters

A single SAE extraction can produce tens of thousands of features, and related concepts are scattered across them: one feature fires on *love* in romantic contexts, another on *love* in "would love to" constructions, a third on *amour*. The **Feature Groups** panel finds these candidate concept clusters automatically — features that share the same **top activating token** (case- and tokenizer-marker-insensitive) with **similar surrounding context**.

The same capability powers the [MCP server's](/advanced/mcp-server) grouping tools — the UI and agents query identical endpoints, so there is exactly one view of the data.

## Building the Index

1. Open **Feature Groups** in the sidebar and pick a completed extraction
2. Click **Compute Index** — a CPU-only background job that:
   - reads each feature's top activating examples,
   - normalizes prime tokens (strips `▁`/`Ġ`/`##` markers, case-folds),
   - builds a token→feature index,
   - and splits each shared-token bucket into subgroups by context similarity (TF-IDF cosine over the ±5-token windows)
3. Progress streams live; a few minutes for very large extractions

Extractions are immutable, so the index never goes stale — labels and stars shown in groups are always joined live from the current feature records. Recompute only if you want different parameters.

## Browsing Groups

Each group row shows its shared token, member count, and a **cohesion** score (mean pairwise context similarity — higher means the members fire in more similar contexts). Expand a group to see members with:

- current label (italic grey = still auto-labeled), star color, and stats
- a context snippet showing the token firing in situ (`prefix *token* suffix`)
- a similarity score to the group centroid

Click any member to open the standard Feature Detail modal. The **link icon** finds features related to that member across the whole extraction — via shared tokens, context overlap, or correlation analysis.

Search by token, filter by minimum group size, and sort by size/cohesion/token.

## Validating a Group with Steering

The point of a group is a hypothesis: *"these N features encode roughly the same concept."* Steering is how you test it:

1. Check the members you want to test
2. Click **Steer selected** — the features land pre-populated in the [Steering panel](/core-workflow/steering)
3. Generate with the group members individually and combined; if the hypothesized concept shifts the output in the predicted direction, the group is real

## API

Everything here is available programmatically — see the [Feature Groups API reference](/reference/api/features-labeling#cross-feature-grouping-feature-groups) and the [MCP tool catalog](/advanced/mcp-server#tool-catalog-33-tools).
