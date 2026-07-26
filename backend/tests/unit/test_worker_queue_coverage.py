"""Every routed queue must have a consumer, and the GPU slot must stay free.

BACKGROUND (2026-07-26)
-----------------------
`task_routes` carefully sorted work into queues — `extraction` for GPU jobs,
`low_priority` for CPU post-processing — while the deployment ran ONE
`--pool=solo -c 1` worker subscribed to all of them. Routing was decorative: a
single slot ran everything in arrival order.

A 12.6-hour NLP pass took that slot. Behind it: the next two extractions of the
batch, and every `cleanup_stuck_*` beat task — the janitor that un-wedges jobs
could not run while jobs were wedged.

These tests pin both halves of the fix:
  * no task is routed to a queue nothing consumes (silent black hole)
  * `low_priority` is NOT on the GPU worker (or the split does nothing)

MUTATION CONTROLS:
  * add low_priority back to the GPU worker's CELERY_QUEUES -> separation test fails
  * route a task to a queue no container lists -> coverage test fails
  * give both workers the same CELERY_WORKER_NAME -> uniqueness test fails
"""

from pathlib import Path

import pytest
import yaml

MANIFEST = Path(__file__).resolve().parents[3] / "k8s" / "base" / "backend.yaml"

# The steering queue has no manifest container by design: a dedicated GPU
# worker is spawned on demand by the API and reaped by exit_steering_mode. The
# exemption is verified below rather than assumed — if the spawner disappears,
# the queue really does become a black hole.
DYNAMIC_QUEUES = {"steering"}


def _worker_containers():
    if not MANIFEST.exists():           # pragma: no cover - repo layout guard
        pytest.skip(f"manifest not found at {MANIFEST}")
    workers = {}
    for doc in yaml.safe_load_all(MANIFEST.read_text()):
        if not doc or doc.get("kind") != "Deployment":
            continue
        for c in doc["spec"]["template"]["spec"]["containers"]:
            env = {e["name"]: e.get("value") for e in c.get("env", [])}
            if env.get("SERVICE_TYPE") != "celery-worker":
                continue
            workers[c["name"]] = {
                "queues": [q for q in (env.get("CELERY_QUEUES") or "").split(",") if q],
                "name": env.get("CELERY_WORKER_NAME"),
                "gpu": (c.get("resources", {}).get("limits", {}) or {}).get("nvidia.com/gpu"),
            }
    assert workers, "no celery-worker containers found in the manifest"
    return workers


def _routed_queues():
    from src.core.celery_app import celery_app

    queues = set()
    for route in (celery_app.conf.task_routes or {}).values():
        if isinstance(route, dict) and route.get("queue"):
            queues.add(route["queue"])
    for entry in (celery_app.conf.beat_schedule or {}).values():
        q = (entry.get("options") or {}).get("queue")
        if q:
            queues.add(q)
    return queues


class TestQueueCoverage:
    def test_every_routed_queue_has_a_consumer(self):
        consumed = set()
        for w in _worker_containers().values():
            consumed |= set(w["queues"])

        orphaned = _routed_queues() - consumed - DYNAMIC_QUEUES
        assert not orphaned, (
            f"tasks are routed to {sorted(orphaned)} but no worker container "
            "subscribes — those tasks would sit in Redis forever"
        )

    def test_every_worker_declares_its_queues_explicitly(self):
        """An unset CELERY_QUEUES falls back to the entrypoint's all-queues
        default, which silently re-merges the split."""
        for name, w in _worker_containers().items():
            assert w["queues"], (
                f"{name} does not set CELERY_QUEUES, so it inherits the "
                "all-queues default and re-creates the single-slot bottleneck"
            )


class TestGpuSlotIsNotBlockedByCpuWork:
    def test_low_priority_is_not_on_the_gpu_worker(self):
        workers = _worker_containers()
        gpu_workers = [n for n, w in workers.items() if "extraction" in w["queues"]]
        assert gpu_workers, "no worker consumes the extraction queue"

        for name in gpu_workers:
            assert "low_priority" not in workers[name]["queues"], (
                f"{name} consumes both 'extraction' and 'low_priority'. With "
                "--pool=solo -c 1 a multi-hour NLP/cleanup task blocks every "
                "extraction and training behind it."
            )

    def test_low_priority_has_a_dedicated_consumer(self):
        workers = _worker_containers()
        consumers = [n for n, w in workers.items() if "low_priority" in w["queues"]]
        assert consumers, (
            "nothing consumes low_priority — NLP, grouping, finalize, prune and "
            "every cleanup_stuck_* beat task would never run"
        )

    def test_worker_node_names_are_unique(self):
        """Celery routes revoke/ping by node name; duplicates make control
        messages land on an arbitrary worker."""
        names = [w["name"] for w in _worker_containers().values()]
        assert all(names), "a worker container has no CELERY_WORKER_NAME"
        assert len(set(names)) == len(names), f"duplicate worker names: {names}"


class TestEntrypointHonoursTheOverrides:
    def test_pool_queues_and_name_are_parameterised(self):
        entrypoint = Path(__file__).resolve().parents[2] / "docker-entrypoint.sh"
        text = entrypoint.read_text()
        for var in ("CELERY_QUEUES", "CELERY_POOL", "CELERY_WORKER_NAME"):
            assert f"${{{var}" in text, (
                f"{var} is not honoured by docker-entrypoint.sh — the manifest "
                "sets it and nothing reads it"
            )
        assert "--pool=${CELERY_POOL:-solo}" in text, (
            "the pool default must remain solo: fork breaks CUDA init"
        )


class TestDynamicQueueExemptionIsReal:
    def test_the_steering_worker_is_actually_spawned(self):
        """`steering` is exempt from the coverage check only because something
        starts a worker for it at runtime. Verify that thing exists."""
        src = (
            Path(__file__).resolve().parents[2]
            / "src" / "api" / "v1" / "endpoints" / "steering.py"
        ).read_text()
        assert '"-Q", "steering"' in src, (
            "no on-demand steering worker spawner found — the DYNAMIC_QUEUES "
            "exemption is now hiding a queue with no consumer at all"
        )


class TestProductionEnvironmentIsDeclared:
    """SQLAlchemy echo must be off in the deployed containers.

    `Settings.environment` defaults to "development", and nothing in the
    manifest set it — so `echo=settings.is_development` was True in production.
    The NLP pass commits per feature, which produced ~20,700 log lines in 13
    minutes and rotated away the exact window a 2026-07-26 incident needed.

    MUTATION CONTROL: remove ENVIRONMENT from any container -> this fails.
    """

    def test_every_container_declares_environment_production(self):
        if not MANIFEST.exists():          # pragma: no cover
            pytest.skip(f"manifest not found at {MANIFEST}")

        missing = []
        for doc in yaml.safe_load_all(MANIFEST.read_text()):
            if not doc or doc.get("kind") != "Deployment":
                continue
            for c in doc["spec"]["template"]["spec"]["containers"]:
                env = {e["name"]: e.get("value") for e in c.get("env", [])}
                if env.get("ENVIRONMENT") != "production":
                    missing.append(f"{c['name']}={env.get('ENVIRONMENT')!r}")

        assert not missing, (
            "containers do not declare ENVIRONMENT=production, so Settings "
            f"falls back to 'development' and turns on SQL echo: {missing}"
        )
