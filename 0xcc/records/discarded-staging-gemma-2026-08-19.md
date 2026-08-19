# Discarded staged artifact — gemma-2-2b-it, 2026-08-19

Debris from an interrupted fit on 2026-08-04, sitting in
`/data/jlens/gemma-2-2b-it.staging/`. Discarded by an operator decision so the
first hardware acquisition could reach the quality gate.

It was NOT junk: a conformant, converged fit, and the Phase-4 staging guard
protected it on its first real outing.

| | staged (discarded) | published (kept) |
|---|---|---|
| n_prompts | 549 | **634** |
| converged | true | true |
| fitted_layers | 0–24 | 0–24 |
| target_layer | penultimate | penultimate |
| corpus | openwebtext-2m-1200docs | — |
| sha256 | 4bd371cde1f800ce… | 285f8c48f06d543a… |

Discarded because it is strictly weaker than the published artifact on the only
axis the quality gate compares — fewer prompts, same convergence — so it could
never have displaced it and had no further use.

Its full recipe is preserved below.
    model: google/gemma-2-2b-it
    d_model: 2304
    n_layers: 26
    n_vocab: 256000
    dtype: fp16
    target_layer: penultimate
    target_position_scope: all_subsequent
    source_position_aggregation: mean_over_all_positions
    differentiation_mode: reverse
    aggregation: mean
    seq_len: 66.1
    attention_gradients_requested: frozen_qk
    norm_statistics: differentiated
    attention_gradients_applied_to_layers: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25]
    fitted_layers: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]
    degenerate_layers: [24]
    corpus: openwebtext-2m-1200docs
    n_prompts: 549
    converged: true
    convergence_delta: 0.001
    layer_scales:
      0: 1.0
      1: 1.0
      2: 1.0
      3: 1.0
      4: 1.0
      5: 1.0
      6: 1.0
      7: 1.0
      8: 1.0
      9: 1.0
      10: 1.0
      11: 1.0
      12: 1.0
      13: 1.0
      14: 1.0
      15: 1.0
      16: 1.0
      17: 1.0
      18: 1.0
      19: 1.0
      20: 1.0
      21: 1.0
      22: 1.0
      23: 1.0
      24: 1.0
    linearisation_residual_mean:
      0: 5.36956
      1: 5.55173
      2: 5.70901
      3: 5.14975
      4: 4.74821
      5: 4.71122
      6: 3.96149
      7: 4.50235
      8: 4.1599
      9: 3.88342
      10: 3.53896
      11: 3.43873
      12: 3.07796
      13: 3.12544
      14: 2.72811
      15: 2.68325
      16: 2.73693
      17: 2.54781
      18: 2.44275
      19: 2.29011
      20: 2.14261
      21: 2.18323
      22: 1.93939
      23: 1.66309
      24: 0
    linearisation_residual_max:
      0: 8.19562
      1: 8.44357
      2: 8.09482
      3: 7.05395
      4: 6.57434
      5: 6.53602
      6: 5.52896
      7: 5.95306
      8: 5.49062
      9: 5.19878
      10: 4.92219
      11: 4.82478
      12: 4.47913
      13: 4.45578
      14: 3.98662
      15: 3.90454
      16: 3.87002
      17: 3.51983
      18: 3.2753
      19: 3.00743
      20: 2.77053
      21: 2.72161
      22: 2.39758
      23: 2.1153
      24: 0
    per_layer_applicability:
      - layer: 0
        has_attention: true
        frozen_qk_applicable: true
      - layer: 1
        has_attention: true
        frozen_qk_applicable: true
      - layer: 2
        has_attention: true
        frozen_qk_applicable: true
      - layer: 3
        has_attention: true
        frozen_qk_applicable: true
      - layer: 4
        has_attention: true
        frozen_qk_applicable: true
      - layer: 5
        has_attention: true
        frozen_qk_applicable: true
      - layer: 6
        has_attention: true
        frozen_qk_applicable: true
      - layer: 7
        has_attention: true
        frozen_qk_applicable: true
      - layer: 8
        has_attention: true
        frozen_qk_applicable: true
      - layer: 9
        has_attention: true
        frozen_qk_applicable: true
      - layer: 10
        has_attention: true
        frozen_qk_applicable: true
      - layer: 11
        has_attention: true
        frozen_qk_applicable: true
      - layer: 12
        has_attention: true
        frozen_qk_applicable: true
      - layer: 13
        has_attention: true
        frozen_qk_applicable: true
      - layer: 14
        has_attention: true
        frozen_qk_applicable: true
      - layer: 15
        has_attention: true
        frozen_qk_applicable: true
      - layer: 16
        has_attention: true
        frozen_qk_applicable: true
      - layer: 17
        has_attention: true
        frozen_qk_applicable: true
      - layer: 18
        has_attention: true
        frozen_qk_applicable: true
      - layer: 19
        has_attention: true
        frozen_qk_applicable: true
      - layer: 20
        has_attention: true
        frozen_qk_applicable: true
      - layer: 21
        has_attention: true
        frozen_qk_applicable: true
      - layer: 22
        has_attention: true
        frozen_qk_applicable: true
      - layer: 23
        has_attention: true
        frozen_qk_applicable: true
      - layer: 24
        has_attention: true
        frozen_qk_applicable: true
      - layer: 25
        has_attention: true
        frozen_qk_applicable: true
