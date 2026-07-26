"""Labeling must survive a model that narrates before answering.

MEASURED against granite-4.1-8b via miLLM:

  unconstrained        11.7s   205 output tokens   reply starts with prose
  + JSON-only system    5.8s    99 output tokens   reply starts with '{'

Two independent failures came from that prose:

1. LATENCY. Generation cost is linear in tokens emitted, so ~100 tokens of
   narration is ~6s of pure waste per feature. At 32,693 features that is days.

2. SILENT DATA LOSS. `json.JSONDecoder.raw_decode` only parses from position 0,
   so a leading preamble made the parser fail and fall back to
   category='uncategorized' with an empty description — WHILE THE PROGRESS
   COUNTER ADVANCED. A production run produced 317 such rows that looked like
   completed work.

Note: miLLM ignores the OpenAI `response_format` parameter (verified — byte
identical output with and without), so the constraint has to live in the prompt.

MUTATION CONTROLS:
  * remove the scan-forward-to-'{' fallback -> the preamble tests fail
  * make _enforce_json_only a passthrough -> the directive tests fail
"""

import pytest

from src.services.openai_labeling_service import (
    JSON_ONLY_DIRECTIVE,
    _enforce_json_only,
    OpenAILabelingService,
)


class _Svc:
    """Minimal stand-in exposing the one helper _parse_dual_label needs."""

    def _clean_label(self, v):
        return str(v).strip().lower().replace(" ", "_")


def _parse(response, fallback="fallback_label"):
    return OpenAILabelingService._parse_dual_label(_Svc(), response, fallback)


class TestParserToleratesPreamble:
    def test_recovers_json_after_a_prose_preamble(self):
        """THE regression that cost 317 labels."""
        response = (
            'The provided examples appear to be a repeating sequence of tokens:\n\n'
            '- "Ghana"\n- "places"\n\n'
            '{"category": "repetition", "specific": "repeated tokens", "description": "d"}'
        )
        out = _parse(response)
        assert out["category"] == "repetition"
        assert out["specific"] == "repeated_tokens"
        assert out["description"] == "d"

    def test_recovers_json_after_a_markdown_heading_preamble(self):
        response = (
            "**Reasoning Process:**\n"
            "  - Ghana, places, but\n\n"
            '{"category": "geography", "specific": "place names"}'
        )
        out = _parse(response)
        assert out["category"] == "geography"
        assert out["specific"] == "place_names"

    def test_ignores_a_brace_that_is_not_valid_json(self):
        """A stray '{' must not abort the scan before the real object."""
        response = 'Consider the set { of tokens } ...\n{"category":"c","specific":"s"}'
        out = _parse(response)
        assert out["category"] == "c"
        assert out["specific"] == "s"

    def test_clean_json_still_works(self):
        out = _parse('{"category":"c","specific":"s","description":"d"}')
        assert (out["category"], out["specific"], out["description"]) == ("c", "s", "d")

    def test_think_tags_still_stripped(self):
        out = _parse('<think>reasoning</think>{"category":"c","specific":"s"}')
        assert out["category"] == "c"

    def test_markdown_fence_still_stripped(self):
        out = _parse('```json\n{"category":"c","specific":"s"}\n```')
        assert out["category"] == "c"

    def test_genuinely_unparseable_still_falls_back(self):
        """The fallback must remain — this test guards against over-eager parsing."""
        out = _parse("no json at all here", fallback="fb")
        assert out["category"] == "uncategorized"
        assert out["specific"] == "fb"


class TestJsonOnlyDirective:
    def test_directive_is_appended_to_a_custom_system_message(self):
        result = _enforce_json_only("You are an expert.")
        assert result.startswith("You are an expert.")
        assert "CRITICAL OUTPUT RULE" in result

    def test_directive_is_not_duplicated(self):
        once = _enforce_json_only("Base.")
        twice = _enforce_json_only(once)
        assert twice.count("CRITICAL OUTPUT RULE") == 1

    def test_empty_system_message_still_gets_the_rule(self):
        assert "CRITICAL OUTPUT RULE" in _enforce_json_only("")

    def test_directive_demands_json_only(self):
        """The wording is what halved output tokens; keep it explicit."""
        d = JSON_ONLY_DIRECTIVE.lower()
        assert "nothing else" in d
        assert "no preamble" in d
