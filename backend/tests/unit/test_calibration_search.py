"""The usable-band search finds onset by DRIFT and the cliff by the JUDGE (IDL-37).

The centrepiece is `test_the_judge_is_load_bearing`: a synthetic world where
perplexity/collapse is FLAT across the cliff but the judge flips correct→broken.
The search must place the cliff where the JUDGE flips — and a perplexity-only
calibrator must miss it. This pins the exact mistake the placeholder sweep made
(it declared 1.40 usable because the output was fluent, while it was false).
"""

from src.services.circuit_calibration_search import (
    calibrate, find_cliff, find_onset)


# ── synthetic world ─────────────────────────────────────────────────────────
# Correctness as a function of dial: correct below CLIFF, broken above. Fluency
# (no collapse) holds far past the cliff — so a fluency/collapse signal cannot
# find the cliff; only the judge can.
CLIFF = 0.6
ONSET = 0.2


def gen_at(dial, prompt):
    # Text encodes the dial so divergence + judge can read it deterministically.
    return f"dial={dial}|answer-for:{prompt}"


def baseline_at(prompt):
    return f"dial=0.0|answer-for:{prompt}"


def divergence(a, b):
    # Distance = |dial_a - dial_b|, parsed from the encoded text. Baseline vs
    # baseline = 0 (floor); steered vs baseline grows with the dial.
    def dial_of(t):
        return float(t.split("|", 1)[0].split("=")[1])
    return abs(dial_of(a) - dial_of(b))


def judge(text, expected):
    dial = float(text.split("|", 1)[0].split("=")[1])
    if dial <= CLIFF:
        return "correct"
    return "broken"


PROBES = [{"prompt": "q1", "expected": "a1"}, {"prompt": "q2", "expected": "a2"}]


class TestOnset:
    def test_onset_is_the_first_dial_above_the_noise_floor(self):
        onset, floor, _ = find_onset(
            gen_at, baseline_at, divergence, lo=0.0, hi=1.0, probes=PROBES, coarse=6)
        assert floor == 0.0            # baseline vs baseline is identical here
        assert onset > 0.0             # some steered dial crossed it
        assert onset <= 0.2 + 1e-9     # first coarse step (0.2) already diverges

    def test_an_inert_circuit_has_onset_at_lo(self):
        # gen == baseline regardless of dial → never crosses the floor.
        flat = lambda d, p: baseline_at(p)
        onset, floor, _ = find_onset(
            flat, baseline_at, divergence, lo=0.0, hi=1.0, probes=PROBES)
        assert onset == 0.0


class TestCliff:
    def test_cliff_lands_at_the_judge_boundary(self):
        cliff, non_mono, steps, converged, _ = find_cliff(
            gen_at, judge, lo=ONSET, hi=1.0, probes=PROBES, max_steps=12, tol=0.02)
        assert abs(cliff - CLIFF) <= 0.05      # bisection converges to the boundary
        assert cliff <= CLIFF + 1e-9           # never reports a BROKEN dial as usable
        assert non_mono is False

    def test_all_broken_returns_lo(self):
        allbad = lambda t, e: "broken"
        cliff, *_ = find_cliff(gen_at, allbad, lo=ONSET, hi=1.0, probes=PROBES)
        assert cliff == ONSET

    def test_all_correct_returns_hi(self):
        allgood = lambda t, e: "correct"
        cliff, *_ = find_cliff(gen_at, allgood, lo=ONSET, hi=1.0, probes=PROBES)
        assert cliff == 1.0

    def test_worst_probe_decides__one_broken_probe_caps_the_dial(self):
        # Probe q2 breaks at a LOWER dial than q1; the worst (lower) governs.
        def judge_split(text, expected):
            dial = float(text.split("|", 1)[0].split("=")[1])
            limit = 0.4 if expected == "a2" else 0.8
            return "correct" if dial <= limit else "broken"
        cliff, *_ = find_cliff(gen_at, judge_split, lo=ONSET, hi=1.0, probes=PROBES,
                               max_steps=14, tol=0.02)
        assert cliff <= 0.4 + 0.05     # capped by the stricter probe, not 0.8


class TestTheJudgeIsLoadBearing:
    """The negative control this whole feature exists for."""

    def test_a_collapse_only_calibrator_MISSES_the_cliff(self):
        # collapse() never fires (output is always fluent), so if the cliff were
        # decided by collapse the usable band would run all the way to hi=1.0 —
        # PAST the real cliff at 0.6. This is the placeholder-sweep failure.
        never_collapses = lambda text: False
        cliff_by_collapse_only, *_ = find_cliff(
            gen_at, lambda t, e: "correct",   # a judge that only ever sees fluency
            lo=ONSET, hi=1.0, probes=PROBES, collapse=never_collapses)
        assert cliff_by_collapse_only == 1.0   # collapse-only sails past 0.6 — WRONG

    def test_the_JUDGE_places_the_cliff_correctly_despite_flat_fluency(self):
        # Same fluent world, but the real judge reads correctness → cliff at 0.6.
        cliff, *_ = find_cliff(
            gen_at, judge, lo=ONSET, hi=1.0, probes=PROBES,
            collapse=lambda text: False, max_steps=12, tol=0.02)
        assert abs(cliff - CLIFF) <= 0.05      # the judge finds what collapse could not


class TestFullCalibrate:
    def test_band_is_ordered_and_sweet_spot_sits_below_the_cliff(self):
        res = calibrate(
            gen_at, baseline_at, judge, divergence,
            probes=PROBES, lo=0.0, hi=1.0, max_steps=12, margin=0.15)
        assert res.onset <= res.sweet_spot <= res.cliff
        assert abs(res.cliff - CLIFF) <= 0.05
        assert res.sweet_spot <= res.cliff        # margin keeps it inside
        assert res.trace                          # every judged step recorded

    def test_non_monotone_is_flagged_when_a_break_sits_below_a_pass(self):
        # World where the TOP is correct but there is a broken dip below it:
        # hi=1.0 passes, but dials in [0.4, 0.55] break. Bisection will accept a
        # pass above the dip and then hit a break below it → the dip must flag.
        def judge_dip(text, expected):
            dial = float(text.split("|", 1)[0].split("=")[1])
            return "broken" if 0.4 <= dial <= 0.55 else "correct"
        _c, non_mono, _s, _conv, _tr = find_cliff(
            gen_at, judge_dip, lo=ONSET, hi=0.7, probes=PROBES,
            max_steps=16, tol=0.01)
        assert non_mono is True   # the break-below-a-pass was surfaced, not smoothed

    def test_a_clean_monotone_world_is_NOT_flagged(self):
        # Specificity: the flag must not fire on a normal monotone cliff.
        _c, non_mono, *_ = find_cliff(gen_at, judge, lo=ONSET, hi=1.0,
                                      probes=PROBES, max_steps=12, tol=0.02)
        assert non_mono is False
