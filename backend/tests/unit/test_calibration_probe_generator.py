"""Probes must be NEUTRAL-topic and falsifiable (IDL-37).

The whole point of the generator: a probe on the circuit's OWN concept has no
falsifiable answer, so the judge could never detect the correctness cliff. These
pin that the generator (a) targets unrelated topics, (b) drops any probe that
drifted onto the concept, and (c) degrades safely to neutral fallbacks rather
than calibrating against an undetectable cliff.
"""

import json

from src.services.circuit_probe_generator import (
    _FALLBACK_PROBES, _looks_on_concept, generate_probes)


class TestNeutrality:
    def test_a_probe_on_the_concept_is_recognised_as_on_concept(self):
        assert _looks_on_concept(
            {"prompt": "Tell me a parody of a news report", "expected": "..."},
            ["parody", "comedians", "humor"]) is True

    def test_a_neutral_probe_is_not_flagged(self):
        assert _looks_on_concept(
            {"prompt": "What is the capital of France?", "expected": "Paris"},
            ["parody", "comedians", "humor"]) is False

    def test_generated_probes_that_drift_onto_the_concept_are_DROPPED(self):
        """The LLM returns a mix; only the neutral ones survive."""
        def fake_llm(prompt, max_tokens):
            return json.dumps([
                {"prompt": "Write a parody about weather", "expected": "n/a"},   # on-concept
                {"prompt": "What is the capital of Japan?", "expected": "Tokyo"},  # neutral
            ])

        probes = generate_probes(["parody", "humor"], llm_call=fake_llm, n=3)
        assert probes == [{"prompt": "What is the capital of Japan?", "expected": "Tokyo"}]

    def test_all_on_concept_falls_back_to_NEUTRAL_probes(self):
        """If every generated probe is on-concept, calibrating against them
        would be blind — fall back to the neutral static set instead."""
        def fake_llm(prompt, max_tokens):
            return json.dumps([
                {"prompt": "Write a comedic parody", "expected": "n/a"},
                {"prompt": "Tell a humor joke about comedians", "expected": "n/a"},
            ])

        probes = generate_probes(["parody", "humor", "comedians"], llm_call=fake_llm, n=3)
        assert probes == _FALLBACK_PROBES[:3]
        # and the fallbacks are themselves neutral to this concept
        for p in probes:
            assert not _looks_on_concept(p, ["parody", "humor", "comedians"])


class TestRobustness:
    def test_no_llm_uses_neutral_fallback(self):
        probes = generate_probes(["parody"], llm_call=None, n=2)
        assert probes == _FALLBACK_PROBES[:2]

    def test_garbage_llm_output_falls_back(self):
        probes = generate_probes(["parody"], llm_call=lambda p, m: "not json at all", n=3)
        assert probes == _FALLBACK_PROBES[:3]

    def test_fenced_json_is_parsed(self):
        def fake_llm(prompt, max_tokens):
            return ('Here you go:\n```json\n'
                    '[{"prompt":"What is 2+2?","expected":"4"}]\n```')
        probes = generate_probes(["parody"], llm_call=fake_llm, n=1)
        assert probes == [{"prompt": "What is 2+2?", "expected": "4"}]

    def test_every_probe_carries_prompt_and_expected(self):
        """A probe missing its expected answer cannot be judged; it is dropped."""
        def fake_llm(prompt, max_tokens):
            return json.dumps([
                {"prompt": "What is the capital of Peru?", "expected": "Lima"},
                {"prompt": "Missing answer?"},  # no expected → dropped
            ])
        probes = generate_probes(["parody"], llm_call=fake_llm, n=3)
        assert probes == [{"prompt": "What is the capital of Peru?", "expected": "Lima"}]
