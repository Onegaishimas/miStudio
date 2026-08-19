# Clustering document set

Every 0xcc document that carries a clustering requirement, copied 2026-08-19.
Copies, not moves — the originals are unchanged and remain authoritative.

## The CLUSTERS chain — cluster discovery, authoring, strength
BRD-MIS-CLUSTERS-001.md ................. the increment BRD (114 "cluster", 0 "causal")
012_F{PRD,TDD,TID,TASKS}|Clusters_UX .... rename Feature Groups→Clusters, blended results
013_F{PRD,TDD,TID,TASKS}|Cluster_Strength_Model
                                          frequency-derived budget, similarity-weighted
                                          allocation, B = B_dir/max(G,floor)^γ, γ=0
014_F{PRD,TDD,TID,TASKS}|Cluster_Definitions
                                          authoring + portable JSON export/import

PPRD rows 13-15 (§3.13-3.15) · PADR IDL-28 (terminology), IDL-29 (strength budget),
IDL-30 (portable definitions).

## Where the grouping algorithm itself is specified
004_FPRD|Feature_Discovery.md ........... feature extraction the grouping runs over
010_FPRD|MCP_Server.md .................. the increment that shipped FeatureGroupingService
                                          (TF-IDF context subgroups) + the grouping tables

## The seam with causal influence
BRD-MIS-CIRCUITS-001.md ................. circuits arc BRD
BRD-MIS-CIRCUITS-002.md ................. rigor supplement; **BR-016 §3.3 + Appendix A.4**
                                          define cluster-level supernode mining. Appendix A
                                          is normative math; 002 wins conflicts with 001.
016_FPRD|Circuit_Discovery.md ........... the granularity toggle that surfaces it

A cluster becomes a circuit member as `member_kind: cluster_ref` and then earns
evidence on the same ladder as any other member. That is the ONLY route by which
a cluster acquires a causal claim.

## Project level
000_PPRD|miStudio.md · 000_PADR|miStudio.md

## What is NOT here, and why
No cluster document mentions causal, rung, or intervention — not once across all
thirteen. A cluster is a correlational object: features grouped by co-occurrence
and TF-IDF context similarity. The strength budget allocates a DIAL, not evidence.
Causal standing lives entirely in the CIRCUITS chain (017 Circuit_Validation is
rung-2), and reaches clusters only through BR-016.
