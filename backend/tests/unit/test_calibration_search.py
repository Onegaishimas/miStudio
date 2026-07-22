"""The usable-band search finds onset by DRIFT and the cliff by the JUDGE (IDL-37).

Two guarantees pinned here (Round-1 review findings):
  * the returned cliff/sweet_spot is ALWAYS a dial the judge called correct —
    never a broken dial reported as usable (S1, S2);
  * `usable_band` is False when no correct dial exists above onset, so the caller
    does not clamp a served dial to a degenerate band (S2, A8);
  * a broken DIP anywhere in the band is caught by the dense scan (S4);
  * the noise floor is a real multi-seed estimate, so a floor=0 mutation is
    caught (A2/S3).

The centrepiece remains the load-bearing-judge control: a world where fluency is
flat across the cliff but the judge flips — the judge must place the cliff, and a
collapse-only calibrator must miss it.
"""

from src.services.circuit_calibration_search import (
    calibrate, find_cliff, find_onset)

CLIFF = 0.6
ONSET = 0.2


def gen_at(dial, prompt):
    return f"dial={dial}|answer-for:{prompt}"


def baseline_at(prompt, seed):
    # Deterministic per (prompt, seed); DIFFERENT seeds differ slightly so the
    # noise floor is nonzero (real sampling jitter).
    return f"dial=0.0|seed={seed}|answer-for:{prompt}"


def _dial_of(t):
    return float(t.split("|", 1)[0].split("=")[1])


def divergence(a, b):
    # Distance dominated by the dial gap; a small seed-difference term gives a
    # nonzero-but-small baseline-vs-baseline floor.
    da, db = _dial_of(a), _dial_of(b)
    seed_a = a.split("seed=")[1].split("|")[0] if "seed=" in a else "x"
    seed_b = b.split("seed=")[1].split("|")[0] if "seed=" in b else "x"
    jitter = 0.02 if seed_a != seed_b else 0.0
    return abs(da - db) + jitter


def judge(text, expected):
    return "correct" if _dial_of(text) <= CLIFF else "broken"


PROBES = [{"prompt": "q1", "expected": "a1"}, {"prompt": "q2", "expected": "a2"}]


class TestOnset:
    def test_floor_is_nonzero_from_seed_jitter(self):
        _onset, floor, _ = find_onset(
            gen_at, baseline_at, divergence, lo=0.0, hi=1.0, probes=PROBES, coarse=6)
        assert floor > 0.0, "the noise floor must reflect baseline sampling jitter"

    def test_onset_clears_the_floor_bar(self):
        onset, floor, _ = find_onset(
            gen_at, baseline_at, divergence, lo=0.0, hi=1.0, probes=PROBES, coarse=6)
        assert onset > 0.0

    def test_an_inert_circuit_has_onset_at_lo(self):
        flat = lambda d, p: baseline_at(p, 0)   # steered == a baseline draw
        onset, _floor, _ = find_onset(
            flat, baseline_at, divergence, lo=0.0, hi=1.0, probes=PROBES)
        assert onset == 0.0


class TestCliffIsAlwaysJudgeCorrect:
    def test_cliff_is_a_correct_dial_at_the_boundary(self):
        cliff, usable, non_mono, steps, converged, _ = find_cliff(
            gen_at, judge, lo=ONSET, hi=1.0, probes=PROBES, max_steps=12, tol=0.02)
        assert usable is True
        assert judge(gen_at(cliff, "q1"), "a1") == "correct", (
            "the returned cliff must itself be judged correct — never the break")
        assert cliff <= CLIFF + 1e-9

    def test_all_broken_reports_no_usable_band(self):
        allbad = lambda t, e: "broken"
        cliff, usable, *_ = find_cliff(gen_at, allbad, lo=ONSET, hi=1.0, probes=PROBES)
        assert usable is False           # lo itself broke → no band
        assert cliff == ONSET

    def test_all_correct_returns_hi_and_it_is_correct(self):
        allgood = lambda t, e: "correct"
        cliff, usable, *_ = find_cliff(gen_at, allgood, lo=ONSET, hi=1.0, probes=PROBES)
        assert usable is True
        assert cliff == 1.0

    def test_worst_probe_governs(self):
        def judge_split(text, expected):
            limit = 0.4 if expected == "a2" else 0.8
            return "correct" if _dial_of(text) <= limit else "broken"
        cliff, usable, *_ = find_cliff(gen_at, judge_split, lo=ONSET, hi=1.0,
                                       probes=PROBES, max_steps=16, tol=0.02)
        assert usable is True
        assert cliff <= 0.4 + 0.05
        assert judge_split(gen_at(cliff, "q2"), "a2") == "correct"


class TestBrokenDipIsCaught:
    def test_a_narrow_dip_below_a_correct_hi_is_flagged_and_the_cliff_stays_below_it(self):
        # broken only in [0.4, 0.55]; correct elsewhere incl. hi.
        def judge_dip(text, expected):
            d = _dial_of(text)
            return "broken" if 0.4 <= d <= 0.55 else "correct"
        cliff, usable, non_mono, _s, _c, _t = find_cliff(
            gen_at, judge_dip, lo=ONSET, hi=1.0, probes=PROBES, max_steps=16, tol=0.02)
        assert usable is True
        assert non_mono is True, "a break below a later pass must be flagged"
        assert cliff < 0.4, "the cliff must stay BELOW the dip, not above it"
        assert judge_dip(gen_at(cliff, "q1"), "a1") == "correct"

    def test_a_clean_monotone_world_is_not_flagged(self):
        _c, _u, non_mono, *_ = find_cliff(gen_at, judge, lo=ONSET, hi=1.0,
                                          probes=PROBES, max_steps=12, tol=0.02)
        assert non_mono is False


class TestTheJudgeIsLoadBearing:
    """A world where fluency is flat across the cliff: only the judge can find it."""

    def test_a_collapse_only_calibrator_MISSES_the_cliff(self):
        # A perplexity/collapse-only calibrator = judge that only sees fluency
        # (never fires) → it would call every fluent dial correct and place the
        # cliff at hi, PAST the real break at 0.6.
        never_collapses = lambda text: False
        fluent_judge = lambda t, e: "correct"   # fluency-only: can't see falsity
        cliff, usable, *_ = find_cliff(
            gen_at, fluent_judge, lo=ONSET, hi=1.0, probes=PROBES,
            collapse=never_collapses)
        assert cliff == 1.0, "collapse/fluency-only sails past the real cliff (0.6)"

    def test_the_real_JUDGE_places_it_correctly_despite_flat_fluency(self):
        cliff, usable, *_ = find_cliff(
            gen_at, judge, lo=ONSET, hi=1.0, probes=PROBES,
            collapse=lambda text: False, max_steps=12, tol=0.02)
        assert usable is True
        assert abs(cliff - CLIFF) <= 0.06
        assert judge(gen_at(cliff, "q1"), "a1") == "correct"

    def test_collapse_shortcut_never_sets_the_cliff_below_the_judge(self):
        # Collapse fires on very high dials only (token soup at >0.8); the judge
        # cliff (0.6) is below that, so the shortcut must not move the cliff.
        collapse_high = lambda text: _dial_of(text) > 0.8
        cliff, usable, *_ = find_cliff(
            gen_at, judge, lo=ONSET, hi=1.0, probes=PROBES,
            collapse=collapse_high, max_steps=12, tol=0.02)
        assert abs(cliff - CLIFF) <= 0.06   # still the judge's boundary


class TestFullCalibrate:
    def test_band_ordered_and_usable(self):
        res = calibrate(gen_at, baseline_at, judge, divergence,
                        probes=PROBES, lo=0.0, hi=1.0, max_steps=12, margin=0.15)
        assert res.usable_band is True
        assert res.onset <= res.sweet_spot <= res.cliff
        assert abs(res.cliff - CLIFF) <= 0.06
        assert judge(gen_at(res.sweet_spot, "q1"), "a1") == "correct"
        assert judge(gen_at(res.cliff, "q1"), "a1") == "correct"
        assert res.floor > 0.0
        assert res.trace

    def test_shipped_sweet_spot_is_re_judged_correct_even_over_a_missed_dip(self):
        # A dip the coarse scan can miss: broken only in a narrow window that the
        # grid may straddle. Whatever the scan does, the RETURNED sweet_spot must
        # be re-judged correct (never a broken dial shipped as the default).
        def judge_dip(text, expected):
            d = _dial_of(text)
            # cliff at 0.9, but a broken dip right where sweet would land.
            if 0.72 <= d <= 0.77:
                return "broken"
            return "correct" if d <= 0.9 else "broken"
        res = calibrate(gen_at, baseline_at, judge_dip, divergence,
                        probes=PROBES, lo=0.0, hi=1.0, max_steps=10, margin=0.15)
        if res.usable_band:
            assert judge_dip(gen_at(res.sweet_spot, "q1"), "a1") == "correct", (
                "the shipped sweet_spot must be a judge-correct dial")

    def test_a_judge_that_fails_the_UNSTEERED_baseline_is_flagged_unreliable(self):
        # Hardware-informed (E2E): a weak judge called the unsteered "capital of
        # France" answer broken. That is a JUDGE failure, not "the circuit has no
        # band" — must surface judge_reliable=False, NOT a false no-band claim.
        bad_judge = lambda t, e: "broken"   # calls everything broken, incl. dial 0
        res = calibrate(gen_at, baseline_at, bad_judge, divergence,
                        probes=PROBES, lo=0.0, hi=1.0, max_steps=10)
        assert res.judge_reliable is False
        assert res.usable_band is False

    def test_a_reliable_judge_is_not_flagged(self):
        res = calibrate(gen_at, baseline_at, judge, divergence,
                        probes=PROBES, lo=0.0, hi=1.0, max_steps=12)
        assert res.judge_reliable is True

    def test_no_usable_band_when_broken_from_onset(self):
        # A judge that breaks everywhere above 0.05 — onset (drift) will land
        # above that, so lo is already broken → no usable band.
        judge_early = lambda t, e: "correct" if _dial_of(t) <= 0.05 else "broken"
        res = calibrate(gen_at, baseline_at, judge_early, divergence,
                        probes=PROBES, lo=0.0, hi=1.0, max_steps=12)
        assert res.usable_band is False
        assert res.onset == res.sweet_spot == res.cliff   # degenerate, flagged
