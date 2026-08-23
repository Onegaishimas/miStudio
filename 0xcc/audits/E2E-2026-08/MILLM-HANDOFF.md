# MILLM-HANDOFF — findings whose confirmation needs the miLLM repo

The audit's locked repo boundary was **miStudio only**; miLLM gets its own
end-to-end assessment. These items were observed from this side and deliberately
not chased. Start there rather than rediscovering them.

## 1. The contract mirror is hand-written and can silently drop a field

`millm/api/schemas/*.py` mirrors miStudio's vendored contract by hand. The recorded
memory `circuit-contract-mirror-drift` says a new field is silently dropped on
import/re-export when the mirror is not updated. **Check whether `calibration`
survives a miLLM round-trip** — miStudio's own import path drops it (MIS-E2E-037),
so a round-trip through both planes may lose it twice for different reasons.

## 2. miLLM currently contains miStudio's bugs' containment

Three miStudio findings are **contained** by miLLM-side clamps, verified from this
side by reading miLLM source:

- **MIS-E2E-034** — miStudio lets a circuit's `intensity` reach `1e9` via an unbounded
  `intensity_range`. `millm/api/schemas/circuit.py:123` has `ge=0.0, le=2.0`, and
  `millm/core/steering_range.py:declared_intensity_range` intersects any authored
  range with `[0, 2]`, normalises swapped pairs, degrades malformed content to `None`,
  and NaN is guarded at `millm/ml/circuit_steering.py:400-410`.
- **MIS-E2E-035** — the evidence rung is self-asserted in the document.
  `millm/services/circuit_service.py:302` and `:1338` gate activation on
  `is_validated(circuit.rung)` — reading a number the document asserts about itself.
  `validation_manifest_ref` is declared at `millm/api/schemas/circuit.py:99` and
  **never dereferenced anywhere in miLLM**.

**The handoff question:** these clamps are miLLM's defence against a *miStudio*
defect. When miStudio fixes 034 and 035, do not remove them — they are the only thing
standing between a hand-authored document and the serving plane.

## 3. Verify from the serving side

- Does miLLM ever receive a circuit whose `rung` it cannot substantiate? A rung-3
  circuit with no resolvable manifest is importable today.
- Does the sensing/claims machinery assume `Feature.training_id`? MIS-E2E-135 found
  three miStudio consumers assuming it; the same shape may exist across the boundary.

## 4. Not investigated at all

The miLLM runtime — circuit/cluster serving, sensing, the OpenAI-compatible API — was
out of scope by decision. Nothing here should be read as a clean bill.
