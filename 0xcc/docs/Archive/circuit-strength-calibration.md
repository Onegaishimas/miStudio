# Circuit strength calibration — measured envelope (2026-07-21)

**Model:** LFM2.5-1.2B-Instruct · **Circuit:** 2 layers, 1 edge
**Members:** L12 `#3121 casual_gaming` (max_act 6.81) → L13 `#8154 casino_game` (max_act 10.20)
**Method:** strengths set as a fraction of each feature's own `max_activation`,
served through miLLM's OpenAI endpoint, same prompt each run, `temperature=0.3`.

## Why this exists

A serving circuit (`closeout-l10-l14`) was producing pure token soup for every
caller:

> `lédot léusiveIMP出版IMP IMPcastle chevIMPIMPIMP…`

It ran **5 layers at strengths 40/35/30/25/20 — 150 units total**. Deactivating
it restored fluent English immediately, isolating steering as the sole cause.
That circuit was a *plumbing fixture* (it proved 5 SAEs attach and serve); its
strengths were never calibrated against generated text, and identical
`max_activation: 10.0` on every member marks them as placeholders.

This is the calibration run that fixture never had.

## Result

| fraction | L12 | L13 | **total** | coherence | outcome |
|---|---|---|---|---|---|
| 0.00 | 0.00 | 0.00 | 0.00 | 0.79 | baseline, fluent |
| 0.02 | 0.14 | 0.20 | **0.34** | 0.80 | fluent |
| 0.05 | 0.34 | 0.51 | **0.85** | 0.74 | fluent |
| 0.08 | 0.54 | 0.82 | **1.36** | 0.79 | fluent |
| 0.12 | 0.82 | 1.22 | **2.04** | 0.81 | fluent |
| 0.18 | 1.23 | 1.84 | **3.07** | 0.82 | fluent — **last good** |
| 0.25 | 1.70 | 2.55 | **4.25** | **0.29** | **collapsed** |
| 0.50 | 3.40 | 5.10 | 8.50 | — | `**InInIn.**.` (12 chars) |
| 1.00 | 6.81 | 10.20 | 17.01 | — | **empty output** |
| 2.00+ | — | — | 34+ | — | empty / `--` |

**The cliff sits between total 3.07 and 4.25** — a *sharp* transition, not a
gradual degradation. One step past it, output collapses to repeated determiners
(`The. An. In. In.`); two steps past, generation stops entirely.

### The headline number

**Usable envelope ≈ 3 units total across 2 layers — roughly 0.18 × max_activation
per member.** The overdriven fixture ran at **150**, about **50× the measured
ceiling**. It was not marginally too strong; it was off the scale.

This is *tighter* than the earlier close-out finding (two layers at strength 5
destroying generation). Consistent in direction, and it locates the boundary
rather than just confirming failure past it.

## Caveats — read before generalising

- **One model, one prompt, one feature pair.** Indicative, not a law. The
  envelope may differ by feature (a high-frequency feature likely tolerates
  less), by layer depth, and by prompt.
- **Coherence here is a garbage detector, not a quality score.** It is
  type-token ratio over words. The first metric attempted — fraction of ASCII
  word tokens — scored `"The.InIn. An. The."` at **1.00**, i.e. it rated
  obvious garbage as perfect. Repetition, not character set, is the failure
  mode; any future metric must catch that.
- **Steering effect was not isolated from noise.** Novel-vocabulary-vs-baseline
  ran 0.60–0.78 across the fluent range with no monotonic trend, which at
  `temperature=0.3` is as consistent with sampling variance as with steering.
  **This run establishes where the circuit stops being safe, not that it
  steers toward gaming/casino semantics.** Demonstrating the intended effect
  needs a targeted prompt set and multiple samples per strength.

## How to reproduce

`build_sweep.py` (session scratchpad) imports the circuit at each strength,
activates with `acknowledge_unvalidated=true` (rung 0 — hand-authored, no
causal validation), generates, then deactivates and deletes. It leaves no
circuit behind; verified `active circuits: 0` afterwards.

## Practical guidance

- Start new circuits at **≤0.15 × max_activation** per member and sweep upward.
- **Read the generated text.** The cliff is sharp enough that a single
  too-large step produces unusable output with no warning.
- Prefer **2 layers** while calibrating; every added layer adds to the same
  residual sum.
- `max_activation` must be a **real measured value**. Placeholder values make
  a fraction-of-max strength meaningless — which is how the fixture ended up
  at 50× the ceiling.
