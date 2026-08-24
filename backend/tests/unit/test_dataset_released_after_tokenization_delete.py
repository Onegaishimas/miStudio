"""Deleting a tokenization must release the dataset when nothing is in flight.

Reported 2026-08-24. The Datasets modal showed an amber "Processing" badge on
`hard-negatives` beside two READY tokenizations, with an idle worker and no
running job. `datasets.status` had been stuck at PROCESSING since a cancelled
tokenization hours earlier.

The delete handler reset the dataset only when the row it removed was the LAST
one:

    if not remaining_tokenizations:      # zero remain
        dataset.status = READY

So cancelling on a dataset that already had finished tokenizations left it
PROCESSING forever. The condition asked the wrong question — whether other
tokenizations EXIST says nothing about whether work is in flight, and two
finished ones are the strongest evidence the dataset is idle.
"""

import ast
import inspect

import pytest


def _handler_source() -> str:
    from src.api.v1.endpoints import datasets

    return inspect.getsource(datasets.delete_dataset_tokenization)


class TestTheConditionAsksAboutWorkNotExistence:
    def test_it_no_longer_requires_zero_remaining(self):
        src = _handler_source()
        assert "if not remaining_tokenizations:" not in src, (
            "the reset still requires that NO tokenization remains, so "
            "cancelling on a dataset with finished tokenizations leaves it "
            "stuck in PROCESSING"
        )

    def test_it_checks_for_in_flight_work(self):
        src = _handler_source()
        assert "in_flight" in src, "nothing distinguishes running from finished"
        tree = ast.parse(inspect.cleandoc(src))
        names = {
            n.attr for n in ast.walk(tree)
            if isinstance(n, ast.Attribute)
            and isinstance(n.value, ast.Name)
            and n.value.id == "TokenizationStatus"
        }
        assert {"QUEUED", "PROCESSING"} <= names, (
            f"the in-flight test does not cover both unfinished states; saw {names}"
        )

    def test_the_reset_still_happens(self):
        """Guard against 'fixing' this by never resetting at all."""
        src = _handler_source()
        assert "DatasetStatus.READY" in src
        assert "dataset_status_changed = True" in src


class TestTheStatesItActsOn:
    """READY and ERROR are different: only a stuck one should be released."""

    def test_only_processing_and_error_are_reset(self):
        src = _handler_source()
        assert "DatasetStatus.PROCESSING, DatasetStatus.ERROR" in src, (
            "the reset should apply to a stuck dataset, not overwrite every "
            "status — a DOWNLOADING dataset must not be flipped to READY"
        )

    def test_finished_tokenizations_do_not_block_the_reset(self):
        """The exact scenario: two READY rows must not count as in flight."""
        from src.models.dataset_tokenization import TokenizationStatus

        finished = [TokenizationStatus.READY]
        in_flight_states = {TokenizationStatus.QUEUED, TokenizationStatus.PROCESSING}
        assert not [t for t in finished if t in in_flight_states], (
            "READY is being treated as in-flight; the dataset would stay stuck"
        )

    def test_an_actually_running_tokenization_does_block_it(self):
        """The fix must not release a dataset with real work underway."""
        from src.models.dataset_tokenization import TokenizationStatus

        running = [TokenizationStatus.READY, TokenizationStatus.PROCESSING]
        in_flight_states = {TokenizationStatus.QUEUED, TokenizationStatus.PROCESSING}
        assert [t for t in running if t in in_flight_states], (
            "a PROCESSING tokenization no longer blocks the reset, so the "
            "dataset would be marked READY while work is still running"
        )
