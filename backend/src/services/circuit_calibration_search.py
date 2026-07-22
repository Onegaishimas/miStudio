"""Adaptive usable-band search for circuit strength calibration (IDL-37).

Two thresholds, two DIFFERENT detectors — this separation is the whole point:

  ONSET (min influence above none) is a DIFFERENCE test: the smallest dial where
  the steered output diverges from the unsteered baseline past the model's own
  sampling-noise floor. No judge — just "did the output move".

  CORRECTNESS CLIFF (max before facts break) is a PROPERTY test: the largest dial
  where every probe answer is still CORRECT, decided by an LLM judge. Perplexity
  and theme metrics CANNOT find this — the observed cliff sat between two adjacent
  dials a perplexity delta could not separate, one giving a correct answer with
  light tint, the next confidently false. A cheap collapse signal may only LOWER
  the search ceiling before judging; it never decides the cliff.

The search is a pure function of injected callbacks (generation, judging,
divergence), so it is unit-testable without a GPU or an LLM — including the
load-bearing-judge negative control, where perplexity is flat across the cliff
but the judge flips, and the search must place the cliff at the JUDGE flip.

Assumes correctness is roughly MONOTONE in strength (observed universally: tint
grows, then breaks). A break below a pass is not smoothed away — it is reported
as `non_monotone` and the cliff is taken at the LOWEST break (conservative).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Literal, Optional

logger = logging.getLogger(__name__)

Verdict = Literal["correct", "degrading", "broken"]

#: A generation callback: given a dial value, return the model's output text.
GenFn = Callable[[float], str]
#: A judge callback: given (generation, expected_answer), return a verdict.
JudgeFn = Callable[[str, str], Verdict]
#: A divergence callback: given two texts, return a distance in [0, ~2].
DivergenceFn = Callable[[str, str], float]


@dataclass
class CalibrationResult:
    onset: float
    sweet_spot: float
    cliff: float
    non_monotone: bool = False
    steps_used: int = 0
    converged: bool = True
    #: (dial, probe_prompt, verdict, reason?) — every judged generation, for the manifest.
    trace: List[Dict] = field(default_factory=list)


def _coarse_steps(lo: float, hi: float, n: int) -> List[float]:
    if n < 2 or hi <= lo:
        return [lo]
    span = hi - lo
    return [round(lo + span * i / (n - 1), 4) for i in range(n)]


def find_onset(
    gen_at: Callable[[float, str], str],
    baseline_at: Callable[[str], str],
    divergence: DivergenceFn,
    *,
    lo: float,
    hi: float,
    probes: List[Dict[str, str]],
    coarse: int = 6,
) -> tuple[float, float, List[Dict]]:
    """Smallest dial whose steered output diverges from baseline past the noise
    floor. The floor is baseline-vs-baseline variation, so onset means "above the
    model's own sampling noise", not an absolute constant.

    `gen_at(dial, prompt)` and `baseline_at(prompt)` are the generation surfaces.
    Returns (onset, floor, trace). If nothing crosses (an inert circuit), onset
    is `lo`.
    """
    if not probes:
        return lo, 0.0, []

    # Noise floor: two UNSTEERED generations per probe, their divergence — the
    # model's own sampling noise. onset must beat THIS, not an absolute constant.
    floor = max(
        divergence(baseline_at(p["prompt"]), baseline_at(p["prompt"]))
        for p in probes
    )

    trace: List[Dict] = []
    for dial in _coarse_steps(lo, hi, coarse):
        if dial <= lo:
            continue
        d = sum(
            divergence(baseline_at(p["prompt"]), gen_at(dial, p["prompt"]))
            for p in probes
        ) / len(probes)
        trace.append({"dial": dial, "phase": "onset", "divergence": d, "floor": floor})
        if d > floor:
            return dial, floor, trace
    return lo, floor, trace


def _worst(verdicts: List[Verdict]) -> Verdict:
    order = {"correct": 0, "degrading": 1, "broken": 2}
    return max(verdicts, key=lambda v: order.get(v, 2)) if verdicts else "broken"


def find_cliff(
    gen_at: Callable[[float, str], str],
    judge: JudgeFn,
    *,
    lo: float,
    hi: float,
    probes: List[Dict[str, str]],
    collapse: Optional[Callable[[str], bool]] = None,
    max_steps: int = 10,
    tol: float = 0.02,
) -> tuple[float, bool, int, bool, List[Dict]]:
    """Largest dial where EVERY probe is still judged CORRECT (the cliff), by
    bisection between a known-correct lo and a known-broken hi.

    `collapse(text) -> bool` is an OPTIONAL cheap shortcut that may lower `hi`
    before judging (skip obviously-degenerate dials) — it never decides the
    cliff. The judge is the only thing that sets the cliff.

    Returns (cliff, non_monotone, steps_used, converged, trace).
    """
    trace: List[Dict] = []
    steps = 0

    def verdict_at(dial: float) -> Verdict:
        nonlocal steps
        vs: List[Verdict] = []
        for p in probes:
            text = gen_at(dial, p["prompt"])
            v = judge(text, p["expected"])
            vs.append(v)
            trace.append({"dial": dial, "phase": "cliff", "probe": p["prompt"],
                          "verdict": v})
        steps += 1
        return _worst(vs)

    # Optional collapse shortcut: walk hi DOWN past any degenerate dial so the
    # judge is not spent on token soup. Cheap signal, ceiling only.
    if collapse is not None:
        guard = 0
        while hi > lo and guard < max_steps:
            sample = gen_at(hi, probes[0]["prompt"]) if probes else ""
            if not collapse(sample):
                break
            hi = round(lo + (hi - lo) / 2, 4)
            guard += 1

    v_lo = verdict_at(lo)
    v_hi = verdict_at(hi)
    non_monotone = False

    # lo already broken → the whole band is unusable, cliff at lo.
    if v_lo != "correct":
        return lo, False, steps, True, trace

    # hi still correct → the usual monotone case says the usable region extends
    # to hi. But bisection would never look BELOW hi, so a broken DIP between lo
    # and hi would be missed. Spot-check a few interior dials: if one breaks
    # while hi is correct, that is non-monotone — flag it and take the cliff at
    # the FIRST (lowest) break, conservatively (FPRD §3.3 item 10).
    if v_hi == "correct":
        for dial in _coarse_steps(lo, hi, 5):
            if dial <= lo or dial >= hi:
                continue
            if verdict_at(dial) != "correct":
                return dial, True, steps, True, trace
        return hi, False, steps, True, trace

    # Standard bisection: lo correct, hi broken. Narrow to the boundary.
    last_correct = lo
    first_bad = hi
    while steps < max_steps and (first_bad - last_correct) > tol:
        mid = round((last_correct + first_bad) / 2, 4)
        v = verdict_at(mid)
        if v == "correct":
            last_correct = mid
        else:
            first_bad = mid
    converged = (first_bad - last_correct) <= tol
    return last_correct, non_monotone, steps, converged, trace


def calibrate(
    gen_at: Callable[[float, str], str],
    baseline_gen: Callable[[str], str],
    judge: JudgeFn,
    divergence: DivergenceFn,
    *,
    probes: List[Dict[str, str]],
    lo: float,
    hi: float,
    max_steps: int = 10,
    margin: float = 0.15,
    collapse: Optional[Callable[[str], bool]] = None,
) -> CalibrationResult:
    """Full band search: onset (drift) then cliff (judge), then a sweet-spot
    with margin below the cliff.

    `gen_at(dial, prompt) -> text` and `baseline_gen(prompt) -> text` are the
    real generation surfaces; `judge` and `divergence` the scorers. All injected
    so the algorithm is testable without a GPU/LLM.
    """
    onset, floor, onset_trace = find_onset(
        gen_at=gen_at, baseline_at=baseline_gen,
        divergence=divergence, lo=lo, hi=hi, probes=probes,
    )

    cliff, non_monotone, cliff_steps, converged, cliff_trace = find_cliff(
        gen_at=gen_at, judge=judge, lo=onset, hi=hi, probes=probes,
        collapse=collapse, max_steps=max_steps,
    )

    # Sweet-spot: strongest still-correct point with a safety margin below the
    # cliff; never below onset.
    sweet = max(onset, round(cliff - margin, 4))
    if sweet > cliff:            # margin larger than the band → sit at the cliff
        sweet = cliff
    if sweet < onset:
        sweet = onset

    return CalibrationResult(
        onset=onset, sweet_spot=sweet, cliff=cliff,
        non_monotone=non_monotone, steps_used=cliff_steps, converged=converged,
        trace=onset_trace + cliff_trace,
    )
