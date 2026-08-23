"""MIS-E2E-151 — cancel must not destroy tokenizations it was not asked to cancel.

Two independent defects on one path:

  1. The cleanup loop iterated **every** `dataset.tokenizations` and
     `shutil.rmtree`d each `tokenized_path`, ungated by status. Cancelling a
     download in progress therefore deleted the files of tokenizations that had
     completed weeks earlier for other models. The DB rows survived, so the UI
     kept listing them while their directories were gone. The raw-file cleanup
     immediately above it *was* status-gated — the sibling was simply missed.

  2. The endpoint called the cancel task with no `task_id`, under a comment
     asserting none was stored. Both dispatch sites store one. The revoke branch
     was dead for its entire life, so "cancel" returned success while the worker
     ran to completion.

The manual states each tokenization "can be cancelled or deleted independently".
These tests hold the code to that sentence.
"""

import ast
import inspect

import pytest

from src.models.dataset_tokenization import TokenizationStatus


# ── (1) the cleanup is scoped to in-flight work ────────────────────────────

def _cancel_source() -> str:
    from src.workers import dataset_tasks

    src = inspect.getsource(dataset_tasks)
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name == "cancel_dataset_download":
                return ast.get_source_segment(src, node)
    raise AssertionError("cancel_dataset_download not found")


def test_cancel_skips_tokenizations_that_are_not_in_flight():
    """The loop must consult status before deleting.

    Asserted structurally rather than by running the Celery task, which needs a
    live DB session and a bound `self`. What matters is that the guard is ON the
    loop — the pre-fix code had no reference to status anywhere in it.
    """
    src = _cancel_source()
    loop_start = src.index("for tokenization in dataset.tokenizations")
    loop_body = src[loop_start:loop_start + 900]
    assert "tokenization.status" in loop_body, (
        "the tokenization cleanup loop deletes without consulting status — it "
        "will destroy completed tokenizations belonging to other models"
    )
    assert "continue" in loop_body, "status is read but nothing is skipped"


def test_in_flight_set_excludes_completed_and_errored():
    """READY and ERROR tokenizations are finished work; their files must survive.

    Pinning the SET, not just that a filter exists: a filter of
    `{QUEUED, PROCESSING, READY}` would pass a "does it check status" test and
    still delete every completed tokenization.
    """
    src = _cancel_source()
    assert "TokenizationStatus.QUEUED" in src
    assert "TokenizationStatus.PROCESSING" in src
    assert "TokenizationStatus.READY" not in src, (
        "READY tokenizations are completed work and must not be cleaned up"
    )
    assert "TokenizationStatus.ERROR" not in src, (
        "an errored tokenization is not the one being cancelled"
    )


def test_tokenization_status_enum_still_has_the_four_members():
    """Negative control for the two tests above.

    Both read the enum by NAME out of source text. If a member is renamed they
    would silently assert nothing — the `not in` assertions in particular pass
    vacuously against a typo. This pins the names they depend on.
    """
    assert {s.name for s in TokenizationStatus} == {
        "QUEUED", "PROCESSING", "READY", "ERROR"
    }


# ── (2) the revoke branch is reachable ─────────────────────────────────────

def test_cancel_endpoint_passes_a_task_id():
    """The endpoint must forward the stored task_id, or revoke can never fire."""
    from src.api.v1.endpoints import datasets

    src = inspect.getsource(datasets)
    tree = ast.parse(src)
    target = None
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            body = ast.get_source_segment(src, node) or ""
            if "cancel_task(" in body:
                target = body
                break
    assert target, "no caller of cancel_task found"
    assert "task_id=" in target, (
        "cancel_task is called without task_id — the worker's revoke branch is "
        "dead and the job runs to completion after a successful-looking cancel"
    )
    assert 'extra_metadata' in target, (
        "task_id must come from where the dispatch sites actually store it"
    )


def test_both_dispatch_sites_still_store_the_task_id():
    """Negative control for the test above.

    It asserts the endpoint READS `extra_metadata['task_id']`. That is worthless
    if nothing writes it — which is precisely what the removed comment claimed.
    This pins the writers, so the read cannot quietly become a lookup of a key
    that is never set.
    """
    from src.api.v1.endpoints import datasets

    src = inspect.getsource(datasets)
    writes = src.count("metadata['task_id'] = task.id") + src.count(
        'metadata["task_id"] = task.id'
    )
    assert writes >= 2, (
        f"expected both the download and tokenize dispatch sites to record "
        f"task.id, found {writes}"
    )


def test_worker_accepts_the_task_id_and_revokes():
    """The worker's signature and revoke branch must both still be there."""
    from src.workers.dataset_tasks import cancel_dataset_download

    sig = inspect.signature(cancel_dataset_download)
    assert "task_id" in sig.parameters
    assert "revoke" in _cancel_source()
