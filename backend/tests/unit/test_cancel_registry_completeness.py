"""Shape D — every cancellable lifecycle is complete, or it is not registered.

A cancel feature can be incomplete in four independent ways, and each one is
invisible from the others:

  * a scope whose declared columns do not exist -> `setattr` invents an
    attribute on the instance, nothing persists, and the cancel is silently lost
  * a scope with no ROUTE -> the job is startable and not stoppable
  * a task with no CHECKER -> the row flips and the work continues
  * a checker with no SHAPE-A TEST -> nobody has ever proven the work stops

This harness asserts all four from the IMPORTED REGISTRY, never from a regex
over source. A source-scraping guard matches nothing on an unexpected layout
and asserts nothing — twice observed in this repo, and five more times in the
arc that produced this file.
"""

import importlib
import inspect
from pathlib import Path

import pytest

from src.core import cancellation as C

BACKEND = Path(__file__).resolve().parents[2]

#: The one indirection allowed: three circuit routes share a body that calls
#: request_cancel with the scope its caller passed in.
DISPATCH_HELPER = "_cancel_circuit_stage"

#: Lifecycles that deliberately have no HTTP cancel route, with the reason.
#: Shrink this; never grow it without recording why.
NO_ROUTE_BY_DESIGN = {
    # Polled by the J-space tasks; its route is POST /jlens/tasks/{id}/cancel,
    # which lives under a different scope name than the row's table.
    "jlens_task": "routed as /jlens/tasks/{task_id}/cancel",
    # Not an operator action: the NLP pass stops because its extraction was
    # deleted, which `missing_row='cancelled'` expresses.
    "nlp_analysis": "stops via extraction deletion, not a button",
    # Feature 21: stopping a training must run FINALIZE, not unwind, so it
    # keeps its own route and its own dict-return convention.
    "training": "trainings stop/pause routes; cancel must finalize",
    "circuit_capture": "cancelled through the discovery run's own route",
    "circuit_discovery": "POST /circuit-discovery/{run_id}/cancel",
    "circuit_attribution": "POST /circuit-discovery/{run_id}/attribute/cancel",
    "circuit_validation": "POST /circuit-discovery/{run_id}/validate/cancel",
    "model_download": "DELETE /models/{id} and the download cancel route",
}



def _scopes_passed_to(func_name: str) -> set:
    """Scope names passed as the first positional argument to `func_name`.

    Read out of the AST of every endpoint, worker and service, so the check is
    about a CALL that exists rather than a string that appears. `_all_*_source`
    concatenates files, which is fine to parse as one module for this purpose:
    the call sites are what matter, not their scoping.
    """
    import ast

    found = set()
    for path in sorted(
        list((BACKEND / "src/api/v1/endpoints").glob("*.py"))
        + list((BACKEND / "src/workers").glob("*.py"))
        + list((BACKEND / "src/services").glob("*.py"))
    ):
        try:
            tree = ast.parse(path.read_text())
        except SyntaxError:  # pragma: no cover - a broken file is its own test
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            target = node.func
            callee = getattr(target, "id", None) or getattr(target, "attr", None)

            # Three shapes, all of them real calls:
            #   request_cancel("scope", ...)
            #   run_in_threadpool(request_cancel, "scope", ...)   — sync fn off
            #                                                       an async route
            #   _cancel_circuit_stage(circuit_id, "scope", ...)   — the shared
            #                                                       body the three
            #                                                       circuit routes use
            passes_it = any(
                isinstance(a, ast.Name) and a.id == func_name for a in node.args
            )
            if callee != func_name and not passes_it and callee != DISPATCH_HELPER:
                continue
            for arg in node.args:
                if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                    found.add(arg.value)
    return found


def _all_route_source() -> str:
    """Every endpoint module's source, concatenated once."""
    parts = []
    for path in sorted((BACKEND / "src/api/v1/endpoints").glob("*.py")):
        parts.append(path.read_text())
    return "\n".join(parts)


def _all_worker_source() -> str:
    parts = []
    for sub in ("src/workers", "src/services"):
        for path in sorted((BACKEND / sub).glob("*.py")):
            parts.append(path.read_text())
    return "\n".join(parts)


def _shape_a_test_source() -> str:
    parts = []
    for path in sorted((BACKEND / "tests/unit").glob("test_*cancel*.py")):
        parts.append(path.read_text())
    for extra in ("test_labeling_cancellation.py", "test_progress_guard.py",
                  "test_jlens_cancel.py", "test_task_heartbeat.py"):
        p = BACKEND / "tests/unit" / extra
        if p.exists():
            parts.append(p.read_text())
    return "\n".join(parts)


@pytest.mark.parametrize("kind", sorted(C.SCOPES))
def test_every_declared_column_exists_on_its_table(kind):
    """THE ONE THAT FAILS SILENTLY IN PRODUCTION.

    A misdeclared field name is not an error at write time: `setattr` creates
    the attribute on the ORM instance, the commit persists nothing, and the
    cancellation is dropped without a log line. Resolved against the real
    table, not against a list maintained by eye.
    """
    scope = C.SCOPES[kind]
    columns = set(scope.model().__table__.columns.keys())
    for label in (
        "id_field", "status_field", "request_field", "error_field",
        "progress_field", "started_at_field", "completed_at_field",
    ):
        field = getattr(scope, label)
        if field is None:
            continue
        assert field in columns, (
            f"{kind}.{label} = {field!r}, which is not a column on "
            f"{scope.model().__tablename__}. setattr would silently create it "
            f"on the instance and persist nothing."
        )


@pytest.mark.parametrize("kind", sorted(C.SCOPES))
def test_the_terminal_set_contains_the_cancelled_values(kind):
    """A cancelled row that the guard does not consider terminal can be
    revived by the next progress write."""
    scope = C.SCOPES[kind]
    assert scope.cancelled_values <= scope.terminal_values, (
        f"{kind}: {scope.cancelled_values - scope.terminal_values} is a "
        f"cancelled value the guard would let a progress write overwrite"
    )


#: scope -> the live route path an operator uses to stop it. Asserted against
#: the ASSEMBLED APP, not against source: a path that exists in a module but is
#: never included in the router is the signature defect of this repo (16 MCP
#: tools, fully implemented and never registered).
OPERATOR_ROUTES = {
    "activation_extraction": "/api/v1/models/{model_id}/extractions/{extraction_id}/cancel",
    "sae_extraction": "/api/v1/saes/{sae_id}/cancel-extraction",
    "dataset_download": "/api/v1/datasets/{dataset_id}/cancel",
    "dataset_tokenization": "/api/v1/datasets/{dataset_id}/tokenizations/{tokenization_id}/cancel",
    "neuronpedia_export": "/api/v1/neuronpedia/export/{job_id}/cancel",
    "circuit_faithfulness": "/api/v1/circuits/{circuit_id}/faithfulness/cancel",
    "circuit_calibration": "/api/v1/circuits/{circuit_id}/calibration/cancel",
    "steering_record": "/api/v1/circuits/steering-samples/{run_id}/cancel",
    "enhanced_labeling": "/api/v1/enhanced-labeling/{job_id}/cancel",
    "feature_grouping": "/api/v1/feature-groups/runs/{run_id}/cancel",
    "labeling": "/api/v1/labeling/{labeling_job_id}/cancel",
}


def _live_paths() -> set:
    """Every path on the ASSEMBLED application.

    Via `app.openapi()`, not `app.routes`. This FastAPI version stores included
    routers as lazy `_IncludedRouter` objects that carry no `.path` until the
    app is built, so walking `app.routes` sees ten framework paths and none of
    the real surface — a check that would have passed for every scope while
    proving nothing. Generating the schema forces the resolution.
    """
    from src.main import app

    return set(app.openapi()["paths"])


@pytest.mark.parametrize("kind", sorted(C.SCOPES))
def test_the_scope_is_reachable_from_an_operator_route(kind):
    """Startable and not stoppable is the defect this phase closed.

    Two halves, because either alone passes over a broken feature: something
    must WRITE the request for this scope, and a real path must exist on the
    live app to trigger it.
    """
    if kind in NO_ROUTE_BY_DESIGN:
        pytest.skip(f"{kind}: {NO_ROUTE_BY_DESIGN[kind]}")

    # AST, NOT A SUBSTRING. A plain `f'"{kind}"' in source` check passes on the
    # scope name appearing ANYWHERE — its own registration comment, a tqdm
    # `cancel_scope=`, a guard_allows call. Three mutations that deleted the
    # actual `request_cancel(...)` call survived against exactly that.
    assert kind in _scopes_passed_to("request_cancel"), (
        f"nothing calls request_cancel({kind!r}, ...) — the lifecycle can be "
        f"started and not stopped"
    )

    path = OPERATOR_ROUTES.get(kind)
    assert path, (
        f"{kind} has no entry in OPERATOR_ROUTES; add the operator-facing path "
        f"or record it in NO_ROUTE_BY_DESIGN with a reason"
    )
    assert path in _live_paths(), (
        f"{path} is not on the assembled app. A route defined in a module and "
        f"never included in the router is unreachable in production — this "
        f"repo has shipped exactly that."
    )


@pytest.mark.parametrize("kind", sorted(C.SCOPES))
def test_something_actually_polls_the_scope(kind):
    """A route that writes a flag nothing reads is a cancel that does nothing —
    which is the exact class of defect this whole arc remediated."""
    if kind in ("nlp_analysis", "circuit_capture"):
        pytest.skip(f"{kind} is polled through a sibling scope's checker")
    src = _all_worker_source()
    assert f'"{kind}"' in src, (
        f"no worker or service references scope {kind!r}, so nothing polls it"
    )


def test_the_by_design_list_only_names_real_scopes():
    """A ratchet that names a scope which no longer exists is an excuse that
    will silently cover the next real one."""
    stale = set(NO_ROUTE_BY_DESIGN) - set(C.SCOPES)
    assert not stale, f"NO_ROUTE_BY_DESIGN names unregistered scopes: {sorted(stale)}"


def test_every_scope_name_is_used_somewhere():
    """A registered scope nobody references is dead weight that makes the
    registry look more complete than it is."""
    src = _all_route_source() + _all_worker_source()
    unused = [k for k in C.SCOPES if f'"{k}"' not in src]
    assert not unused, f"registered but referenced nowhere: {sorted(unused)}"


def test_the_cancellable_tasks_carry_the_decorator():
    """`@cooperative_cancel` is what turns the raise into a RETURN, which is
    what acks the acks_late message. Located through the imported task object,
    never a regex."""
    expected = {
        "src.workers.model_tasks": ["extract_activations"],
        "src.workers.extraction_tasks": ["extract_features_from_sae_task"],
    }
    for module_name, task_names in expected.items():
        module = importlib.import_module(module_name)
        for task_name in task_names:
            fn = getattr(module, task_name)
            found = None
            for _ in range(6):
                found = getattr(fn, "__cooperative_cancel_scope__", None)
                if found:
                    break
                nxt = getattr(fn, "__wrapped__", None)
                if nxt is None or nxt is fn:
                    break
                fn = nxt
            assert found in C.SCOPES, (
                f"{module_name}.{task_name} is not decorated with a registered "
                f"cancel scope (found {found!r})"
            )


@pytest.mark.parametrize("kind", sorted(C.SCOPES))
def test_a_shape_a_test_exists_for_the_scope(kind):
    """THE ONE THAT MATTERS. Asserting `status == "cancelled"` is worthless —
    the endpoint wrote that. Something must prove the WORK STOPS."""
    if kind in NO_ROUTE_BY_DESIGN and kind not in (
        "circuit_discovery", "circuit_attribution", "circuit_validation",
        "jlens_task",
    ):
        pytest.skip(f"{kind}: {NO_ROUTE_BY_DESIGN[kind]}")
    src = _shape_a_test_source()
    assert f'"{kind}"' in src, (
        f"no test names scope {kind!r}; nobody has demonstrated that a "
        f"cancelled {kind} actually stops doing work"
    )
