"""End-to-end synthetic pin for CircuitDiscoveryService (016 Task 3.3):
planted edge surfaces with full report disclosure; independent
high-base-rate pair does not; seeded mode covers/uncovers correctly.

Uses a real on-disk synthetic store + a real sync session against the test
DB (the service is worker-facing/sync)."""

import numpy as np
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.models.circuit_runs import CircuitCaptureRun, CircuitDiscoveryRun
from src.services.circuit_capture_store import open_writers
from src.services.circuit_discovery_service import (
    CircuitDiscoveryService,
    DiscoveryConfigError,
)


@pytest.fixture
def sync_db(async_engine):
    """A SYNC session (the discovery service is worker-facing/sync) against
    the SAME test DB the async_engine fixture just built the tables in.

    The sync URL is derived from `async_engine.url` — NOT from a separate
    DATABASE_URL_SYNC env var, whose credentials can differ in CI and cause
    an auth failure (the test DB user/password must match what conftest
    actually connected with)."""
    # async_engine.url already points at the _test database (conftest swapped
    # it) with the credentials that actually connected — reuse both, only
    # swapping the async driver for the sync one.
    sync_url = async_engine.url.set(
        drivername=async_engine.url.drivername.replace("+asyncpg", "")
                                              .replace("+psycopg", ""))
    engine = create_engine(sync_url)
    Session = sessionmaker(bind=engine)
    session = Session()
    yield session
    session.rollback()
    session.query(CircuitDiscoveryRun).delete()
    session.query(CircuitCaptureRun).delete()
    session.commit()
    session.close()
    engine.dispose()


N_DOCS, DOC_LEN = 40, 96


def _build_store(tmp_path):
    """L13 feature 1 drives L14 feature 2 (planted); L13 f3 / L14 f4 are
    independent high base-rate."""
    rng = np.random.default_rng(9)
    ev13, en13, _ = open_writers(tmp_path, 13)
    ev14, en14, _ = open_writers(tmp_path, 14)
    for d in range(N_DOCS):
        a_pos = rng.choice(DOC_LEN, size=6, replace=False)
        co = a_pos[:5]
        x_pos = rng.choice(DOC_LEN, size=30, replace=False)
        y_pos = rng.choice(DOC_LEN, size=30, replace=False)
        ev13.append(np.full(6, d), a_pos, np.full(6, 1), rng.uniform(0.5, 2, 6))
        ev13.append(np.full(30, d), x_pos, np.full(30, 3), rng.uniform(0.5, 2, 30))
        ev14.append(np.full(5, d), co, np.full(5, 2), rng.uniform(0.5, 2, 5))
        ev14.append(np.full(30, d), y_pos, np.full(30, 4), rng.uniform(0.5, 2, 30))
        for en in (en13, en14):
            en.append(np.full(DOC_LEN, d), np.arange(DOC_LEN),
                      rng.uniform(0, 0.2, DOC_LEN))
    for w in (ev13, ev14, en13, en14):
        w.finalize()


def _capture_row(db, tmp_path, *, stale=False):
    heldout = list(range(32, 40))
    run = CircuitCaptureRun(
        status="completed", stale=stale, store_path=str(tmp_path),
        manifest={
            "corpus": {"dataset_id": "ds_x", "tokenization_id": "tok_x",
                       "sample_cap": N_DOCS},
            "model_id": "m_x",
            "layers": [{"layer": 13, "sae_id": "sae_l13"},
                       {"layer": 14, "sae_id": "sae_l14"}],
            "split": {"method": "per_document", "ratio": 0.8, "seed": 42,
                      "heldout_docs": heldout},
            "doc_lengths": {str(d): DOC_LEN for d in range(N_DOCS)},
            "counts": {"documents": N_DOCS, "events": 0,
                       "tokens": N_DOCS * DOC_LEN},
        })
    db.add(run)
    db.commit()
    db.refresh(run)
    return run


class TestDiscoveryEndToEnd:
    def test_planted_edge_surfaces_with_full_report(self, sync_db, tmp_path):
        _build_store(tmp_path)
        cap = _capture_row(sync_db, tmp_path)
        run = CircuitDiscoveryService.create_run(sync_db, {
            "capture_run_id": cap.id, "granularity": "feature",
            "mode": "open", "s_min": 20, "null_shuffles": 50})
        result = CircuitDiscoveryService.run(sync_db, run.id)
        assert result["status"] == "completed"
        sync_db.refresh(run)
        cands = run.candidates
        # Planted 13:1 → 14:2 must be the TOP candidate; base-rate 3→4 absent
        # (its support passes but PMI≈0 + null kills it).
        assert cands, "no candidates surfaced"
        top = cands[0]
        assert top["up"] == {"layer": 13, "feature_idx": 1}
        assert top["down"] == {"layer": 14, "feature_idx": 2}
        assert not any(c["up"].get("feature_idx") == 3 and
                       c["down"].get("feature_idx") == 4 for c in cands)
        # Report discipline disclosure (FPRD §3.6).
        rep = run.report
        assert rep["null_summary"]["method"] == "within_document_circular_shift"
        assert rep["fdr"]["discipline"] == "benjamini_hochberg"
        assert rep["replication"]["rate"] is not None
        assert "lag-0" in rep["lag0_disclosure"]
        assert rep["counts_by_stage"]["pairs_considered"] > 0
        # Planted edge replicates on held-out.
        assert top["replicated_heldout"] is True
        assert top["orderings"] == {"coact_rank": 0, "attr_rank": None}

    def test_seeded_mode_restricts_and_reports_uncovered(self, sync_db, tmp_path):
        _build_store(tmp_path)
        cap = _capture_row(sync_db, tmp_path)
        run = CircuitDiscoveryService.create_run(sync_db, {
            "capture_run_id": cap.id, "granularity": "feature",
            "mode": "seeded", "s_min": 20, "null_shuffles": 50,
            "seed_refs": [{"layer": 13, "feature_idx": 1},
                          {"layer": 13, "feature_idx": 999}]})
        CircuitDiscoveryService.run(sync_db, run.id)
        sync_db.refresh(run)
        assert all(c["up"] == {"layer": 13, "feature_idx": 1}
                   for c in run.candidates)
        assert run.report["uncovered_seeds"] == [
            {"layer": 13, "ref": "feature:999",
             "reason": "below support floor or absent from store"}]

    def test_stale_store_refused_without_force(self, sync_db, tmp_path):
        _build_store(tmp_path)
        cap = _capture_row(sync_db, tmp_path, stale=True)
        with pytest.raises(DiscoveryConfigError, match="STALE"):
            CircuitDiscoveryService.create_run(sync_db, {
                "capture_run_id": cap.id, "mode": "open"})
        # force override works
        run = CircuitDiscoveryService.create_run(sync_db, {
            "capture_run_id": cap.id, "mode": "open", "force": True})
        assert run.id

    def test_seeded_requires_seed_refs(self, sync_db, tmp_path):
        _build_store(tmp_path)
        cap = _capture_row(sync_db, tmp_path)
        with pytest.raises(DiscoveryConfigError, match="seed_refs"):
            CircuitDiscoveryService.create_run(sync_db, {
                "capture_run_id": cap.id, "mode": "seeded"})


class TestCancellation:
    """R1 Test-P2 / CR#5/#6: a cancel actually stops the mine and is not
    clobbered by a 'completed' write."""

    def test_cancel_during_run_stops_and_persists_cancelled(self, sync_db, tmp_path):
        _build_store(tmp_path)
        cap = _capture_row(sync_db, tmp_path)
        run = CircuitDiscoveryService.create_run(sync_db, {
            "capture_run_id": cap.id, "granularity": "feature",
            "mode": "open", "s_min": 20, "null_shuffles": 50})

        calls = {"n": 0}

        def cancel_after_two():
            calls["n"] += 1
            return calls["n"] >= 2

        result = CircuitDiscoveryService.run(
            sync_db, run.id, cancel_check=cancel_after_two)
        assert result["status"] == "cancelled"
        sync_db.refresh(run)
        assert run.status == "cancelled"
        assert run.candidates is None  # never wrote 'completed' results
