# TID: Intervention Engine Extension

**Document ID:** 025_FTID|Intervention_Engine · **Version:** 1.0

## Pitfalls

1. **Make the control a constructor argument.** A validating check is the thing that gets bypassed
   with a flag when someone is in a hurry; an unconstructable object is not.
2. **Size-match the control and record its seed.** "A random direction" is not a control; "k random
   directions from seed s" is.
3. **Clamp per (position, layer).** Holding a coordinate at some positions produces a mediation
   result that is not about what it claims.
4. **Dynamic top-k must EXCLUDE clean-pass coordinates**, or it ablates ordinary behaviour and
   reports it as an effect.
5. **Record the primitive and its parameters.** An unrecorded primitive makes a run incomparable.
6. **Derive the swap's layer default from n_layers.** A constant tuned on a large model oversteers
   a small one — BR-017's v0.2 amendment exists because this happened.
7. **Causal language belongs ONLY to a run with a control** (feature 026). Without one there is no
   result to describe.

## Testing
- Constructing a result without a control is a TypeError, not a validation error.
- Control is size-matched; its seed round-trips.
- Clamping holds every named (position, layer).
- Dynamic top-k on a clean-pass coordinate is a no-op.
- The swap default differs between a 16-layer and a 64-layer model.

**Mutation controls:** default the control to None; drop the seed; clamp per-position only; include
clean-pass coordinates in dynamic top-k; fix the swap layer count.
