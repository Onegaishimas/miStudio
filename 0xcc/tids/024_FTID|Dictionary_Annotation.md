# TID: Dictionary Annotation & Weight-Space Readouts

**Document ID:** 024_FTID|Dictionary_Annotation · **Version:** 1.0

## Implementation order
1. `services/jlens_annotation.py` — projection via `LensTransport`, the two fields, disagreement.
2. `ml/jlens_weight_readout.py` — arbitrary weight-space direction → ranked tokens.
3. The distributional shape check.
4. Persistence + the queue (filter/sort).
5. MCP tools + reachability.

## Pitfalls

1. **Reuse `LensTransport`.** A second projection path can disagree with the readout the user sees.
2. **Kurtosis alone labels every motor feature workspace.** Two fields, always.
3. **No band report ⇒ no behavioural field.** Absent, not guessed. There is no principled "middle"
   of the stack without one.
4. **Absent, never zero** — for every field, as everywhere in this arc.
5. **Do not auto-resolve disagreement.** The lens is rung 0 and does not overrule a label.
6. **Encoder and decoder are SEPARATE readouts.** Combined, the transformation is invisible.
7. **A sweep over a full dictionary is a cost envelope**, not a request — 32k features × layers.
   Queue it, bound it, and report the estimate before running (BR-028).
8. **The distributional check is a SHAPE observation**, not validation of the lens. Reporting it as
   validation is a rung claim it has not earned.

## Testing
- Motor-shaped fixture is NOT classified workspace.
- Behavioural field absent without a band report.
- Annotation projection == readout projection on the same direction.
- Disagreement filterable and sortable.
- Distributional check fails when most features are workspace.
- Encoder and decoder readouts differ on a transcoder whose two matrices differ.

**Mutation controls:** classify on kurtosis alone; guess a band boundary; coerce an absent field to
zero; auto-resolve disagreement; merge encoder and decoder; make the distributional check advisory.
