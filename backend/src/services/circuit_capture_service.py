"""
Circuit capture service (Feature 016, BR-005/BR-006/BR-023 — FTDD §1).

Sync (Celery-worker-facing) orchestration of a capture run:

  probe (~PROBE_SAMPLES docs) → cost estimate → [stop if not confirmed]
  → full batch loop: ONE forward per batch, multi-layer residual hooks →
    per-layer SAE encode on-GPU → threshold max(θ_floor, ε·max_act_i)
    (per-feature max from the probe; missing ⇒ floor-only, never skip) →
    event/errnorm append (+ optional attention top-k sidecar)
  → per-document 80/20 split (seeded, recorded) → manifest.json (atomic)

The DB row (circuit_capture_runs.manifest) mirrors manifest.json exactly so
listings never touch disk. Stale-flagging: capture records SAE fingerprints;
`mark_stale_for_sae` flips `stale` when a referenced SAE changes.
"""

import hashlib
import json
import logging
import os
import shutil
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from ..core.config import settings
from ..models.circuit_runs import CircuitCaptureRun
from ..models.dataset import Dataset
from ..models.dataset_tokenization import DatasetTokenization, TokenizationStatus
from ..models.external_sae import ExternalSAE
from .circuit_capture_store import open_writers

logger = logging.getLogger(__name__)

PROBE_SAMPLES = 32
MAX_SEQ_LENGTH = 512  # capture cap — also keeps token_pos comfortably in u16
DEFAULT_EPSILON = 0.1
DEFAULT_THETA_FLOOR = 0.01
DEFAULT_SPLIT_RATIO = 0.8
DEFAULT_SAMPLE_CAP = 2000
EVENT_BYTES = 12  # sizeof EVENT_DTYPE row (u32 feature_idx widened it from 10)
STORE_SIZE_MULTIPLIER = 5.0   # abort a capture exceeding 5× its own estimate
MIN_FREE_DISK_BYTES = 5 * 2**30  # refuse/abort if <5 GB free on the data volume


class CaptureConfigError(ValueError):
    """Invalid capture configuration — surfaces as a 422."""


class CaptureConflictError(RuntimeError):
    """A GPU circuit task is already active — surfaces as a 409."""


def captures_dir() -> Path:
    return settings.data_dir / "circuit_captures"


def size_ceiling_bytes(events_est: int) -> float:
    """The store-size abort threshold (R1 QA-P1): 5× the probe estimate, with
    a 64 MB floor so a tiny estimate doesn't abort a legitimate small capture."""
    return max(events_est * EVENT_BYTES * STORE_SIZE_MULTIPLIER, 64 * 2**20)


def exceeds_size_ceiling(buffered_events: int, events_est: int) -> bool:
    return buffered_events * EVENT_BYTES > size_ceiling_bytes(events_est)


class CircuitCaptureService:
    # ── concurrency guard ────────────────────────────────────────────────

    # Postgres advisory-lock key: serializes the check-then-insert so two
    # concurrent requests can't both pass the guard (R2 B2).
    _GPU_LOCK_KEY = 0x1C1C_C0DE

    @staticmethod
    def assert_no_active_gpu_run(db) -> None:
        """One GPU circuit task at a time on the single 3090 (R1 QA-P1 / R2
        Q1). Covers BOTH captures (this table) AND attribution passes (on the
        discovery row) — attribution loads a model too (R2 Q1). Serialized by
        a transaction-scoped advisory lock so the check-then-insert can't race
        (R2 B2); the lock releases at commit/rollback."""
        from sqlalchemy import text

        from ..models.circuit_runs import CircuitDiscoveryRun

        db.execute(text("SELECT pg_advisory_xact_lock(:k)"),
                   {"k": CircuitCaptureService._GPU_LOCK_KEY})
        active = db.query(CircuitCaptureRun).filter(
            CircuitCaptureRun.status.in_(
                ("pending", "estimating", "running"))).first()
        if active is not None:
            raise CaptureConflictError(
                f"Capture {active.id} is already {active.status} — one GPU "
                f"circuit task runs at a time; wait or cancel it first")
        attr = db.query(CircuitDiscoveryRun).filter(
            CircuitDiscoveryRun.attribution_status.in_(
                ("pending", "running"))).first()
        if attr is not None:
            raise CaptureConflictError(
                f"Attribution pass on {attr.id} is {attr.attribution_status} — "
                f"one GPU circuit task runs at a time; wait or cancel it first")
        # Validation is a GPU task too (R1 #7/Q1 — the guard missed it, so a
        # capture/attribution could run concurrently with a validation pass).
        val = db.query(CircuitDiscoveryRun).filter(
            CircuitDiscoveryRun.validation_status.in_(
                ("pending", "running"))).first()
        if val is not None:
            raise CaptureConflictError(
                f"Validation pass on {val.id} is {val.validation_status} — "
                f"one GPU circuit task runs at a time; wait or cancel it first")
        # Faithfulness runs on a circuit and loads a model too (R2 B-5).
        from ..models.circuit import Circuit
        faith = db.query(Circuit).filter(
            Circuit.faithfulness_status.in_(("pending", "running"))).first()
        if faith is not None:
            raise CaptureConflictError(
                f"Faithfulness pass on {faith.id} is {faith.faithfulness_status} "
                f"— one GPU circuit task runs at a time; wait or cancel it first")
        # Calibration (Feature 20) also runs on a circuit and loads a model —
        # same single-GPU guard, or a calibration could race a capture/
        # faithfulness and OOM as an opaque task failure.
        calib = db.query(Circuit).filter(
            Circuit.calibration_status.in_(("pending", "running"))).first()
        if calib is not None:
            raise CaptureConflictError(
                f"Calibration pass on {calib.id} is {calib.calibration_status} "
                f"— one GPU circuit task runs at a time; wait or cancel it first")
        # Steered-transcript recording (circuit/cluster/feature) also loads a
        # model on the single GPU — its marker lives in steering_record_runs
        # (cluster/feature jobs have no circuit row).
        from ..models.steering_record_run import SteeringRecordRun
        rec = db.query(SteeringRecordRun).filter(
            SteeringRecordRun.status.in_(("pending", "running"))).first()
        if rec is not None:
            raise CaptureConflictError(
                f"A steering-record job ({rec.id}) is {rec.status} — one GPU "
                f"task runs at a time; wait or cancel it first")

    # ── run creation / validation (called from the endpoint) ─────────────

    @staticmethod
    def create_run(db, config: Dict[str, Any]) -> CircuitCaptureRun:
        """Validate config against the DB and create the run row (pending)."""
        dataset_id = config.get("dataset_id")
        layers = config.get("layers") or []
        if not dataset_id:
            raise CaptureConfigError("dataset_id is required")
        if not layers:
            raise CaptureConfigError("at least one {layer, sae_id} entry is required")
        seen_layers = set()
        for entry in layers:
            if "layer" not in entry or "sae_id" not in entry:
                raise CaptureConfigError("each layers[] entry needs layer and sae_id")
            if entry["layer"] in seen_layers:
                raise CaptureConfigError(f"duplicate layer {entry['layer']}")
            seen_layers.add(entry["layer"])

        dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
        if dataset is None:
            raise CaptureConfigError(f"Dataset {dataset_id} not found")

        model_id = config.get("model_id")
        saes: Dict[str, ExternalSAE] = {}
        for entry in layers:
            sae = db.query(ExternalSAE).filter(
                ExternalSAE.id == entry["sae_id"]).first()
            if sae is None:
                raise CaptureConfigError(f"SAE {entry['sae_id']} not found")
            if not sae.local_path:
                raise CaptureConfigError(f"SAE {sae.id} has no local path")
            if sae.layer is not None and sae.layer != entry["layer"]:
                raise CaptureConfigError(
                    f"SAE {sae.id} was trained on layer {sae.layer}, "
                    f"config asks for layer {entry['layer']} — own-layer rule")
            saes[entry["sae_id"]] = sae
            if model_id is None:
                model_id = sae.model_id
            # Wide-SAE bound: feature_idx is u32, but assert d_sae fits so a
            # broken SAE record surfaces at config time, not mid-capture.
            if sae.n_features is not None and sae.n_features > 2**32:
                raise CaptureConfigError(
                    f"SAE {sae.id} has {sae.n_features} features — exceeds u32")

        # model_id must resolve, else the tokenization filter matches NULL and
        # fails confusingly (R1 CR#4).
        if model_id is None:
            raise CaptureConfigError(
                "model_id could not be resolved — pass it explicitly or use "
                "SAEs whose model_id is set")

        tokenization = db.query(DatasetTokenization).filter(
            DatasetTokenization.dataset_id == dataset_id,
            DatasetTokenization.model_id == model_id,
        ).first()
        if tokenization is None or tokenization.status != TokenizationStatus.READY:
            raise CaptureConfigError(
                f"Dataset {dataset_id} has no READY tokenization for model "
                f"{model_id} — tokenize it first")

        epsilon = float(config.get("epsilon", DEFAULT_EPSILON))
        if not (0.0 <= epsilon < 1.0):
            raise CaptureConfigError("epsilon must be in [0, 1)")
        sample_cap = int(config.get("sample_cap", DEFAULT_SAMPLE_CAP))
        if sample_cap < PROBE_SAMPLES:
            raise CaptureConfigError(f"sample_cap must be >= {PROBE_SAMPLES}")

        manifest = {
            "corpus": {
                "dataset_id": dataset_id,
                "tokenization_id": tokenization.id,
                "sample_cap": sample_cap,
            },
            "model_id": model_id,
            "layers": [
                {"layer": e["layer"], "sae_id": e["sae_id"],
                 "threshold_mode": "epsilon_max" if epsilon > 0 else "floor",
                 "epsilon": epsilon,
                 "theta_floor": float(config.get("theta_floor", DEFAULT_THETA_FLOOR))}
                for e in layers
            ],
            "split": {
                "method": "per_document",
                "ratio": DEFAULT_SPLIT_RATIO,
                "seed": int(config.get("split_seed", 42)),
                "heldout_docs": [],  # filled at capture completion
            },
            "attention_capture": config.get("attention_capture"),  # {layers, heads, top_k}|None
            "created_at": datetime.utcnow().isoformat(),
            "stale": False,
        }
        run = CircuitCaptureRun(status="pending", manifest=manifest)
        db.add(run)
        db.commit()
        db.refresh(run)
        return run

    # ── stale flagging ───────────────────────────────────────────────────

    @staticmethod
    def mark_stale_for_sae(db, sae_id: str) -> int:
        """Flag (never delete) every completed run referencing this SAE."""
        runs = db.query(CircuitCaptureRun).filter(
            CircuitCaptureRun.status == "completed",
            CircuitCaptureRun.stale == False,  # noqa: E712
        ).all()
        n = 0
        for run in runs:
            if any(l.get("sae_id") == sae_id
                   for l in (run.manifest or {}).get("layers", [])):
                run.stale = True
                manifest = dict(run.manifest)
                manifest["stale"] = True
                run.manifest = manifest
                n += 1
        if n:
            db.commit()
        return n

    # ── deletion ─────────────────────────────────────────────────────────

    @staticmethod
    def delete_run(db, run: CircuitCaptureRun) -> None:
        if run.status == "running":
            raise CaptureConfigError("Cannot delete a running capture — cancel first")
        if run.store_path:
            store = settings.resolve_data_path(run.store_path)
            # Containment: never rm outside the captures root.
            if store.is_dir() and captures_dir() in store.parents:
                shutil.rmtree(store, ignore_errors=True)
        db.delete(run)
        db.commit()

    # ── worker body ──────────────────────────────────────────────────────

    @staticmethod
    def run_capture(db, run_id: str, *, confirmed: bool,
                    cancel_check=None, progress_cb=None) -> Dict[str, Any]:
        """Execute (probe [+ capture]) for a run. Heavy imports stay inside
        so the module imports cleanly on GPU-less API processes."""
        import torch
        from datasets import load_from_disk

        from ..ml.forward_hooks import HookManager, HookType
        from ..ml.model_loader import load_model_from_hf
        from ..models.model import Model
        from .extraction_service import cleanup_gpu_memory

        run = db.query(CircuitCaptureRun).filter(
            CircuitCaptureRun.id == run_id).first()
        if run is None:
            raise ValueError(f"Capture run {run_id} not found")
        manifest = dict(run.manifest)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model_record = db.query(Model).filter(
            Model.id == manifest["model_id"]).first()
        if model_record is None:
            raise CaptureConfigError(f"Model {manifest['model_id']} not found")

        tokenization = db.query(DatasetTokenization).filter(
            DatasetTokenization.id == manifest["corpus"]["tokenization_id"]).first()
        dataset = load_from_disk(
            str(settings.resolve_data_path(tokenization.tokenized_path)))
        sample_cap = min(int(manifest["corpus"]["sample_cap"]), len(dataset))
        dataset = dataset.select(range(sample_cap))

        model = tokenizer = None
        saes: Dict[int, Any] = {}
        try:
            from ..models.model import QuantizationFormat
            resolved_model_path = (settings.resolve_data_path(model_record.file_path)
                                   if model_record.file_path else None)
            model_is_downloaded = bool(resolved_model_path
                                       and resolved_model_path.exists())
            model, tokenizer, _config, _meta = load_model_from_hf(
                repo_id=model_record.repo_id,
                quant_format=QuantizationFormat(model_record.quantization),
                cache_dir=resolved_model_path,
                device_map=device,
                local_files_only=model_is_downloaded,
            )
            model.eval()

            fingerprints = {}
            for entry in manifest["layers"]:
                sae_record = db.query(ExternalSAE).filter(
                    ExternalSAE.id == entry["sae_id"]).first()
                sae = _load_sae_sync(sae_record, device)
                saes[entry["layer"]] = sae
                fingerprints[entry["sae_id"]] = _sae_fingerprint(sae)
            manifest["sae_fingerprints"] = fingerprints

            layer_indices = sorted(saes.keys())
            attn_cfg = manifest.get("attention_capture") or None

            run.status = "estimating"
            run.progress = 0.0
            db.commit()

            # ── probe: per-feature max + event-rate estimate ────────────
            t0 = time.monotonic()
            probe_max, probe_events, probe_tokens = _probe(
                model, tokenizer, dataset, saes, layer_indices, device)
            probe_seconds = time.monotonic() - t0
            total_tokens_est = probe_tokens / PROBE_SAMPLES * sample_cap
            events_est = int(probe_events / max(probe_tokens, 1) * total_tokens_est)
            estimate = {
                "events": events_est,
                "bytes": events_est * EVENT_BYTES,
                "minutes": round(probe_seconds / PROBE_SAMPLES
                                 * sample_cap / 60.0, 1),
                "probe_samples": PROBE_SAMPLES,
                "probe_events": int(probe_events),
            }
            manifest["estimate"] = estimate
            run.manifest = manifest
            if not confirmed:
                run.status = "estimated"
                run.progress = None
                db.commit()
                return {"status": "estimated", "estimate": estimate}

            # ── full capture ─────────────────────────────────────────────
            store_dir = captures_dir() / run.id
            store_dir.mkdir(parents=True, exist_ok=True)
            # Persist store_path NOW, not at success (R3 B-R3-1): if the worker
            # is OOM-killed mid-capture, cleanup_stuck_circuit_runs can still
            # rmtree the orphaned partial store (its guard is `if store_path`).
            run.status = "running"
            run.store_path = str(store_dir)
            db.commit()
            # Store-size guardrail (R1 QA-P1): abort if the true event rate
            # blows past the probe estimate, or the volume runs low on space.
            if shutil.disk_usage(store_dir).free < MIN_FREE_DISK_BYTES:
                run.status = "failed"
                run.error_message = "Insufficient free disk on the data volume"
                db.commit()
                shutil.rmtree(store_dir, ignore_errors=True)
                return {"status": "failed", "reason": "disk"}
            writers = {L: open_writers(store_dir, L,
                                       attention=bool(attn_cfg and L in
                                                      (attn_cfg.get("layers") or [])))
                       for L in layer_indices}

            batch_size = 8
            n_docs = len(dataset)
            doc_lengths: Dict[int, int] = {}
            for batch_start in range(0, n_docs, batch_size):
                if cancel_check is not None and cancel_check():
                    run.status = "cancelled"
                    db.commit()
                    shutil.rmtree(store_dir, ignore_errors=True)
                    return {"status": "cancelled"}
                batch = dataset[batch_start:min(batch_start + batch_size, n_docs)]
                input_ids, attention_mask, lengths = _pad_batch(
                    batch, tokenizer)
                for i, L in enumerate(lengths):
                    doc_lengths[batch_start + i] = L
                _capture_batch(
                    model, saes, layer_indices, writers,
                    input_ids.to(model.device), attention_mask.to(model.device),
                    batch_start, lengths,
                    epsilon_by_layer={e["layer"]: e["epsilon"]
                                      for e in manifest["layers"]},
                    floor_by_layer={e["layer"]: e["theta_floor"]
                                    for e in manifest["layers"]},
                    probe_max=probe_max, attn_cfg=attn_cfg)
                # Running byte estimate from buffered events (u32 idx + u16
                # pos + u32 doc + f16 act ≈ EVENT_BYTES/event, plus errnorm).
                buffered = sum(ev.count for ev, _en, _at in writers.values())
                if exceeds_size_ceiling(buffered, events_est):
                    run.status = "failed"
                    run.error_message = (
                        f"Capture exceeded {STORE_SIZE_MULTIPLIER}× its size "
                        f"estimate ({buffered} events) — aborted to protect "
                        f"the data volume; lower sample_cap or raise epsilon")
                    db.commit()
                    shutil.rmtree(store_dir, ignore_errors=True)
                    return {"status": "failed", "reason": "size_ceiling"}
                pct = min(99.0, (batch_start + batch_size) / n_docs * 100.0)
                run.progress = pct
                db.commit()
                if progress_cb is not None:
                    progress_cb(pct)

            # finalize writers
            events_total = 0
            for L, (ev, en, at) in writers.items():
                events_total += ev.finalize()
                en.finalize()
                if at is not None:
                    at.finalize()

            # per-document split, seeded, recorded
            rng = np.random.default_rng(int(manifest["split"]["seed"]))
            all_docs = np.arange(n_docs)
            perm = rng.permutation(all_docs)
            cut = int(len(perm) * float(manifest["split"]["ratio"]))
            heldout = sorted(int(d) for d in perm[cut:])
            manifest["split"]["heldout_docs"] = heldout
            manifest["doc_lengths"] = {str(k): v for k, v in doc_lengths.items()}
            manifest["counts"] = {"documents": n_docs, "events": events_total,
                                  "tokens": int(sum(doc_lengths.values()))}
            bytes_total = sum(f.stat().st_size for f in store_dir.iterdir())
            manifest["bytes"] = bytes_total

            _write_manifest_atomic(store_dir / "manifest.json", manifest)

            # Last-writer race: a cancel between the final cancel-check and here
            # must NOT be clobbered by 'completed' (R1 CR#6). Re-read status.
            db.refresh(run)
            if run.status == "cancelled":
                shutil.rmtree(store_dir, ignore_errors=True)
                return {"status": "cancelled"}
            run.manifest = manifest
            run.store_path = str(store_dir)
            run.events_total = events_total
            run.bytes_total = bytes_total
            run.status = "completed"
            run.progress = 100.0
            db.commit()
            return {"status": "completed", "events": events_total,
                    "bytes": bytes_total, "heldout_docs": len(heldout)}
        finally:
            cleanup_gpu_memory(
                [m for m in [model, *saes.values()] if m is not None],
                context=f"circuit_capture:{run_id}")


# ── helpers ──────────────────────────────────────────────────────────────

def _load_sae_sync(sae_record: "ExternalSAE", device: str):
    """Sync SAE load (worker context) — the same path SteeringService.load_sae
    uses internally: auto-detect format → create → load weights → eval."""
    import torch

    from ..ml.community_format import load_sae_auto_detect
    from ..ml.sparse_autoencoder import create_sae

    sae_path = settings.resolve_data_path(sae_record.local_path)
    state_dict, config, _fmt = load_sae_auto_detect(sae_path, device="cpu")
    d_in = sae_record.d_model or (
        state_dict["encoder.weight"].shape[1] if "encoder.weight" in state_dict
        else state_dict["W_enc"].shape[0])
    d_sae = sae_record.n_features or (
        state_dict["encoder.weight"].shape[0] if "encoder.weight" in state_dict
        else state_dict["W_enc"].shape[1])
    architecture = (sae_record.architecture or "standard").lower()
    sae = create_sae(architecture, hidden_dim=d_in, latent_dim=d_sae)
    cleaned = {k.removeprefix("model."): v for k, v in state_dict.items()}
    sae.load_state_dict(cleaned, strict=False)
    sae.to(device).eval()
    return sae


def _sae_fingerprint(sae) -> str:
    """Cheap decoder identity: shape + parameter checksum."""
    import torch

    from .steering_service import resolve_decoder_weight
    w = resolve_decoder_weight(sae)
    if w is None:
        return "unknown"
    with torch.no_grad():
        return f"{tuple(w.shape)}:{float(w.float().abs().sum()):.3f}"


def _pad_batch(batch: Dict[str, Any], tokenizer):
    """HF-dict batch → right-padded (input_ids, attention_mask, lengths)."""
    import torch

    rows = batch["input_ids"]
    seqs: List[List[int]] = []
    for row in rows:
        ids = row.tolist() if hasattr(row, "tolist") else list(row)
        seqs.append(ids[:MAX_SEQ_LENGTH])
    lengths = [len(s) for s in seqs]
    max_len = max(lengths) if lengths else 0
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id or 0
    input_ids = torch.full((len(seqs), max_len), pad_id, dtype=torch.long)
    mask = torch.zeros((len(seqs), max_len), dtype=torch.long)
    for i, s in enumerate(seqs):
        input_ids[i, :len(s)] = torch.tensor(s, dtype=torch.long)
        mask[i, :len(s)] = 1
    return input_ids, mask, lengths


def _encode_layer(sae, acts):
    """acts [n_tokens, d_model] fp32 → z [n_tokens, d_sae] (no grad)."""
    z = sae.encode(acts)
    if isinstance(z, tuple):  # JumpReLU return_pre_activations styles
        z = z[0]
    return z


def _probe(model, tokenizer, dataset, saes, layer_indices, device):
    """~PROBE_SAMPLES docs → per-layer per-feature max + event-rate sample."""
    import torch

    from ..ml.forward_hooks import HookManager, HookType

    probe_max: Dict[int, "torch.Tensor"] = {}
    events = 0
    tokens = 0
    n = min(PROBE_SAMPLES, len(dataset))
    with HookManager(model) as hm:
        hm.register_hooks(layer_indices, [HookType.RESIDUAL],
                          getattr(model.config, "model_type", "auto"))
        for start in range(0, n, 8):
            batch = dataset[start:min(start + 8, n)]
            input_ids, mask, lengths = _pad_batch(batch, tokenizer)
            with torch.no_grad():
                _ = model(input_ids=input_ids.to(model.device),
                          attention_mask=mask.to(model.device))
            for L in layer_indices:
                acts = hm.activations[f"layer_{L}_residual"][-1]  # [b, s, h]
                flat = acts.to(device).float().reshape(-1, acts.shape[-1])
                with torch.no_grad():
                    z = _encode_layer(saes[L], flat)
                fmax = z.max(dim=0).values
                probe_max[L] = (torch.maximum(probe_max[L], fmax)
                                if L in probe_max else fmax)
                events += int((z > 0).sum())
            tokens += int(sum(lengths))
            hm.clear_activations()
    return probe_max, events, tokens


def _capture_batch(model, saes, layer_indices, writers, input_ids, mask,
                   doc_base, lengths, *, epsilon_by_layer, floor_by_layer,
                   probe_max, attn_cfg):
    import torch

    from ..ml.forward_hooks import HookManager, HookType

    with HookManager(model) as hm:
        hm.register_hooks(layer_indices, [HookType.RESIDUAL],
                          getattr(model.config, "model_type", "auto"))
        with torch.no_grad():
            out = model(input_ids=input_ids, attention_mask=mask,
                        output_attentions=bool(attn_cfg))
        for L in layer_indices:
            acts = hm.activations[f"layer_{L}_residual"][-1]  # [b, s, h] cpu
            ev_w, en_w, at_w = writers[L]
            b, s, h = acts.shape
            flat = acts.to(input_ids.device).float().reshape(-1, h)
            with torch.no_grad():
                z = _encode_layer(saes[L], flat)          # [b*s, d_sae]
                recon = saes[L].decode(z)
                err = (flat - recon).norm(dim=-1)          # [b*s]
            eps = epsilon_by_layer[L]
            floor = floor_by_layer[L]
            pm = probe_max.get(L)
            if eps > 0 and pm is not None:
                thresh = torch.clamp(eps * pm, min=floor)  # per-feature
            else:
                thresh = torch.full((z.shape[-1],), floor, device=z.device)
            hits = (z > thresh.unsqueeze(0)).nonzero(as_tuple=False)  # [n, 2]
            if len(hits):
                vals = z[hits[:, 0], hits[:, 1]].cpu().numpy()
                tok_flat = hits[:, 0].cpu().numpy()
                feats = hits[:, 1].cpu().numpy()
                docs_rel = tok_flat // s
                poss = tok_flat % s
                # drop padding positions
                keep = poss < np.array(lengths)[docs_rel]
                ev_w.append((docs_rel[keep] + doc_base).astype(np.uint32),
                            poss[keep], feats[keep], vals[keep])
            # errnorm: every REAL token
            for i, L_i in enumerate(lengths):
                row = err[i * s:i * s + L_i].cpu().numpy()
                en_w.append(np.full(L_i, doc_base + i, dtype=np.uint32),
                            np.arange(L_i), row)
            # attention sidecar
            if at_w is not None and attn_cfg and out.attentions is not None:
                _append_attention(at_w, out.attentions[L], attn_cfg,
                                  doc_base, lengths)
        hm.clear_activations()


def _append_attention(at_w, attn, cfg, doc_base, lengths):
    """attn [b, heads, q, k] → top-k keys per (head, query)."""
    import torch

    top_k = int(cfg.get("top_k", 4))
    heads = cfg.get("heads")  # list | None = all
    b, n_heads, q_len, _ = attn.shape
    head_ids = heads if heads else list(range(n_heads))
    for i, L_i in enumerate(lengths):
        for hd in head_ids:
            probs = attn[i, hd, :L_i, :L_i]
            k = min(top_k, probs.shape[-1])
            mass, keys = torch.topk(probs, k, dim=-1)
            q_idx = torch.arange(L_i).unsqueeze(-1).expand_as(keys)
            at_w.append(
                np.full(q_idx.numel(), doc_base + i, dtype=np.uint32),
                q_idx.reshape(-1).cpu().numpy(),
                np.full(q_idx.numel(), hd, dtype=np.uint16),
                keys.reshape(-1).cpu().numpy(),
                mass.reshape(-1).float().cpu().numpy())


def _write_manifest_atomic(path: Path, manifest: Dict[str, Any]) -> None:
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False,
                                     suffix=".tmp") as f:
        json.dump(manifest, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
        tmp = f.name
    os.replace(tmp, path)
