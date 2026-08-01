# TDD: Intervention Engine Extension

**Document ID:** 025_FTDD|Intervention_Engine · **Version:** 1.0

## 1. The control is a CONSTRUCTOR ARGUMENT, not a field to fill in

BR-018 says a run without its control is invalid. There are two ways to honour that:

- validate after construction, and hope every path validates;
- make the result **impossible to build** without one.

The second is chosen. `InterventionResult` requires `control` positionally; there is no default and
no `Optional`. A caller who has not run the control cannot produce a result object to report.

This matters because the failure mode is social, not technical: under time pressure the control is
the step that gets skipped, and a validating check is the step that gets bypassed with a flag.

## 2. Paired run with clamping (BR-016)

```
clean_pass    -> per-position lens coordinates, recorded
intervened    -> same prompt, with the intervention applied
clamped       -> named coordinates HELD at their clean-pass values throughout
```

Clamping is per (position, layer) — holding a coordinate at one position but not another produces a
mediation result that is not about the thing it names.

## 3. Primitives (BR-017)

Each is a small, named transformation of the activation at a hook point. The engine records WHICH
was applied and with what parameters, because a result whose primitive is unrecorded cannot be
compared to another run.

`DYNAMIC_TOPK_ABLATION` excludes coordinates that were already top-k in the CLEAN pass — otherwise
it ablates the model's ordinary behaviour and reports the result as an intervention effect.

## 4. Scale-aware swap default

Coordinate swaps oversteer at small scale. The default layer count is derived from `n_layers` rather
than fixed, so a 16-layer model does not get a default tuned for a 64-layer one.

## 5. Risks

| risk | mitigation |
|---|---|
| A run is reported without its control | structurally impossible: constructor argument |
| The control is not size-matched | control construction records `k` and asserts it |
| The control cannot be reproduced | seed required, as in the band report |
| Clamping applied at some positions only | clamp spec is per (position, layer); asserted |
| Dynamic top-k ablates ordinary behaviour | clean-pass exclusion, asserted |
| A swap default tuned for large models oversteers a small one | derived from n_layers |
