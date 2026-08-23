"""Task 12 — deployment defects that make a deploy or a diagnosis wrong.

MIS-E2E-144  `k8s_deploy` re-applied a STALE second manifest, reverting the
             queue-split and SQL-echo incident fixes — at the moment the
             break-glass procedure is most likely to be used.
MIS-E2E-145  postgres and redis are Deployments over hostPath with the default
             RollingUpdate, so two pods briefly share one data directory.
MIS-E2E-146  compose published the Celery broker and the database on 0.0.0.0.
MIS-E2E-147  the compose frontend port, and an `&&`-chain that reported any
             failure as a schema warning and returned 0.
MIS-E2E-148  a guard that fails open, an ingress exposing `/api/internal`, and
             a global apt keyring.
"""

from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[3]
K8S_BASE = REPO / "k8s" / "base"
COMPOSE = REPO / "docker-compose.yml"
HELPERS = REPO / "scripts" / "k8s-helpers.sh"


def _docs(path: Path):
    return [d for d in yaml.safe_load_all(path.read_text()) if d]


def test_the_repo_layout_is_what_this_file_assumes():
    """Fail closed. Every assertion below reads a file; if the paths moved,
    they would all vanish silently — the failure mode MIS-E2E-148 names."""
    for p in (K8S_BASE, COMPOSE, HELPERS):
        assert p.exists(), f"{p} not found — this guard would pass vacuously"


# ── MIS-E2E-144 · one manifest ─────────────────────────────────────────────

def test_the_stale_standalone_manifest_is_gone():
    """It duplicated `k8s/base` and drifted: no `celery-worker-cpu`, no
    `CELERY_QUEUES`, no `ENVIRONMENT=production`."""
    assert not (REPO / "k8s" / "mistudio-deployment.yaml").exists(), (
        "the standalone manifest is back; k8s_deploy applying it reverts the "
        "queue-split and SQL-echo fixes"
    )


def test_the_deploy_helper_applies_the_kustomize_base():
    src = HELPERS.read_text()
    assert "kubectl apply -k" in src, (
        "k8s_deploy does not apply the kustomize base ArgoCD deploys"
    )
    assert "K8S_MANIFEST" not in src.replace("# `K8S_MANIFEST`", ""), (
        "the stale manifest variable is still in use"
    )


def test_the_deploy_helper_restarts_the_mcp_deployment():
    """`mistudio-mcp` runs the SAME backend image, and was never restarted —
    so new MCP tools stayed invisible after a break-glass deploy."""
    assert "deployment/mistudio-mcp" in HELPERS.read_text()


# ── MIS-E2E-147 · a failed deploy must fail ────────────────────────────────

def test_the_deploy_helper_does_not_swallow_failures_in_an_and_chain():
    """The body was one `&&` chain ending in `|| echo "WARNING: Schema
    verification failed"`, so a failed pull, apply or rollout printed a message
    about SCHEMA and returned 0."""
    src = HELPERS.read_text()
    start = src.index("k8s_deploy()")
    body = src[start: src.index("\n}", start)]
    assert "DEPLOY FAILED at:" in body, (
        "no step reports its own failure; a failed deploy still returns 0"
    )
    # The advisory warning must apply ONLY to schema verification.
    assert body.count("|| \\") == 0, "the &&-chain is back"


# ── MIS-E2E-145 · no two writers on one hostPath ───────────────────────────

@pytest.mark.parametrize("name", ["postgres", "redis"])
def test_stateful_deployments_use_recreate(name):
    """A Deployment over hostPath with RollingUpdate starts the new pod before
    terminating the old one, so two processes briefly hold the same directory."""
    dep = next(
        d for d in _docs(K8S_BASE / f"{name}.yaml") if d.get("kind") == "Deployment"
    )
    strategy = dep["spec"].get("strategy", {})
    assert strategy.get("type") == "Recreate", (
        f"{name} uses {strategy or 'the default RollingUpdate'} over a hostPath "
        f"volume — two pods can hold the same data directory"
    )


# ── MIS-E2E-146 · the broker is not on the LAN ─────────────────────────────

@pytest.mark.parametrize("service", ["postgres", "redis"])
def test_compose_binds_stateful_ports_to_loopback(service):
    """Redis is the CELERY BROKER: LAN reachability meant anyone could enqueue
    GPU jobs and read queued payloads."""
    compose = yaml.safe_load(COMPOSE.read_text())
    ports = compose["services"][service].get("ports", [])
    for spec in ports:
        text = str(spec)
        assert text.count(":") >= 2, (
            f"{service} publishes {text!r} on all interfaces; bind it to an "
            f"address (127.0.0.1 by default)"
        )
        assert "0.0.0.0:" not in text


def test_compose_frontend_targets_the_unprivileged_port():
    """nginx-unprivileged listens on 8080; compose still mapped to 80, so
    http://localhost:3000 was dead."""
    compose = yaml.safe_load(COMPOSE.read_text())
    ports = [str(p) for p in compose["services"]["frontend"].get("ports", [])]
    assert any(p.endswith(":8080") for p in ports), ports


# ── MIS-E2E-148 · the ingress and the keyring ──────────────────────────────

def test_every_host_serving_api_also_denies_api_internal():
    """Both nginx configs deny `/api/internal`; the ingress must agree.

    Parametrised over the hosts found in the file rather than a named pair —
    the `.net` host had the same gap and is the internet-facing one.
    """
    ingresses = [d for d in _docs(K8S_BASE / "ingress.yaml") if d.get("kind") == "Ingress"]
    assert ingresses, "no Ingress found — the scan broke"

    checked = 0
    for ing in ingresses:
        for rule in ing["spec"]["rules"]:
            paths = [p["path"] for p in rule["http"]["paths"]]
            if "/api" in paths:
                checked += 1
                assert "/api/internal" in paths, (
                    f"{rule['host']} exposes /api without denying /api/internal"
                )
    assert checked >= 2, f"only {checked} hosts serve /api — expected both"


def test_the_dockerfile_scopes_its_apt_keyring():
    """`apt-key adv` trusts the key for EVERY repository; `signed-by=` binds it
    to this source alone."""
    src = (REPO / "backend" / "Dockerfile").read_text()
    assert "apt-key adv" not in src.replace("# `apt-key adv`", "")
    assert "signed-by=/usr/share/keyrings/deadsnakes.gpg" in src


def test_the_queue_coverage_guard_fails_closed():
    """The fourth source-scrape guard in this audit to skip when its input
    moved. A guard that vanishes silently is worse than none."""
    src = (REPO / "backend" / "tests" / "unit" / "test_worker_queue_coverage.py").read_text()
    assert "pytest.skip(f\"manifest not found" not in src
    assert "assert MANIFEST.exists()" in src
