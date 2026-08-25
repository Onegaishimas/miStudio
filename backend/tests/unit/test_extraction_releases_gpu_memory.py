"""
Unloading the model after an extraction must be measured, not announced.

Observed 2026-08-25 on the RTX 3090. Four consecutive extractions of a 12B
model logged "Model cleaned up from GPU 0 memory" every time. The readings
underneath told a different story:

    09:48:10  after_cleanup   Allocated: 5.38 GB   Reserved: 14.02 GB
    09:49:49  after_cleanup   Allocated: 9.32 GB   Reserved: 19.55 GB
    10:31:29  after_cleanup   Allocated: 0.01 GB   Reserved:  6.99 GB

The last one freed every tensor and still never handed the 6.99 GB pool back.
`nvidia-smi` showed the worker holding 7,474 MiB nine hours later, with nothing
alive in it -- unavailable to miLLM, to the next extraction, and to the VRAM
gauge in the UI.

The success line was logged unconditionally, so none of this could surface. A
claim that measures nothing cannot report its own failure. These tests pin the
measurement, and the config that lets `empty_cache()` actually shrink the pool.
"""

from unittest.mock import patch

import pytest
import yaml

from pathlib import Path

from src.services.activation_service import ActivationService


def _service():
    return ActivationService.__new__(ActivationService)


class TestItReportsWhatTheCardDid:
    def test_a_pool_that_never_came_back_is_a_warning(self, caplog):
        svc = _service()
        before = {"allocated": 7.16, "reserved": 19.72}
        after = {"allocated": 0.01, "reserved": 6.99}   # the live incident

        with patch.object(ActivationService, "_gpu_memory", staticmethod(lambda gpu_id=0: after)):
            with caplog.at_level("INFO"):
                report = svc._report_cleanup(0, before)

        assert report["after"]["reserved"] == 6.99
        warnings = [r for r in caplog.records if r.levelname == "WARNING"]
        assert warnings, (
            "6.99 GB reserved with nothing allocated was reported as success"
        )
        assert "did not give the memory back" in warnings[0].message

    def test_a_real_release_is_not_a_warning(self, caplog):
        svc = _service()
        before = {"allocated": 7.16, "reserved": 19.72}
        after = {"allocated": 0.0, "reserved": 0.3}     # context only

        with patch.object(ActivationService, "_gpu_memory", staticmethod(lambda gpu_id=0: after)):
            with caplog.at_level("INFO"):
                report = svc._report_cleanup(0, before)

        assert report["released"] == pytest.approx(19.42)
        assert not [r for r in caplog.records if r.levelname == "WARNING"]

    def test_memory_still_allocated_is_a_warning(self, caplog):
        """The 5.38 GB and 9.32 GB cases, which read as success for months."""
        svc = _service()
        after = {"allocated": 9.32, "reserved": 19.55}

        with patch.object(ActivationService, "_gpu_memory", staticmethod(lambda gpu_id=0: after)):
            with caplog.at_level("INFO"):
                svc._report_cleanup(0, {"allocated": 16.47, "reserved": 19.83})

        assert [r for r in caplog.records if r.levelname == "WARNING"], (
            "9.32 GB still allocated was reported as a successful unload"
        )

    def test_cleanup_returns_the_measurement(self):
        """The caller must be able to act on it, so it cannot be None."""
        svc = _service()

        class _Model:
            def parameters(self): return []
            def buffers(self): return []
            def cpu(self): return self
            def named_children(self): return []

        with patch("src.services.activation_service.torch.cuda.is_available", return_value=False), \
             patch.object(ActivationService, "_log_gpu_memory", lambda *a, **k: None):
            report = svc._cleanup_model(_Model(), gpu_id=0)

        assert isinstance(report, dict)
        assert {"before", "after", "released"} <= set(report)


class TestTheAllocatorCanActuallyShrink:
    """Measurement alone changes nothing; the pool has to be returnable."""

    def test_the_gpu_worker_uses_expandable_segments(self):
        manifest = Path(__file__).resolve().parents[3] / "k8s" / "base" / "backend.yaml"
        if not manifest.exists():          # pragma: no cover
            pytest.skip("manifest not found")

        gpu_worker_conf = None
        for doc in yaml.safe_load_all(manifest.read_text()):
            if not doc or doc.get("kind") != "Deployment":
                continue
            for c in doc["spec"]["template"]["spec"]["containers"]:
                env = {e["name"]: e.get("value") for e in c.get("env", [])}
                if (
                    env.get("SERVICE_TYPE") == "celery-worker"
                    and env.get("CELERY_WORKER_NAME") == "gpu"
                ):
                    gpu_worker_conf = env.get("PYTORCH_CUDA_ALLOC_CONF")

        assert gpu_worker_conf is not None, (
            "the GPU worker sets no PYTORCH_CUDA_ALLOC_CONF, so the default "
            "fixed-size segments cannot be handed back once fragmented"
        )
        assert "expandable_segments:True" in gpu_worker_conf
