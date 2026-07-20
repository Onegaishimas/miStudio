"""
Circuit intervention/validation service (Feature 017, IDL-34 — A.5 normative).

The GPU orchestrator for edge validation:
  for each edge (u→d) in top-K of the chosen ordering:
    prompts = windows around u's STRONGEST firings (016 store, SAME tokenization)
    clean pass → a_d(t);  intervened pass (suppress u) → a_d(t)
    Δ_p = mean_t[a_d clean − a_d intervened] over tokens where clean F_u(t)=1
    ES  = mean_p(Δ_p) / σ_d          # σ_d from the SAME capture store
  null = shuffled NON-edge pairs (random support-matched u' in u's layer, d fixed)
  verdict: rung-2 iff |ES| > null percentile AND sign-consistent ≥ frac (math module)
  batch: survival per ordering → uplift = survival(attr) − survival(coact)
  persist: an edge_batch manifest + write rung-2 results onto the discovery-run
           candidates AND, for a promoted circuit, its edges VIA CircuitService
           (never raw JSONB — 018 R2-A5) with the optimistic-concurrency version.

Never re-decodes (the suppression hook subtracts from the residual). σ_d comes
from the capture store the candidates came from — a different corpus's σ would
silently rescale ES.
"""

import logging
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from ..core.config import settings
from ..models.circuit_runs import CircuitCaptureRun, CircuitDiscoveryRun
from . import circuit_validation_math as vmath
from .circuit_capture_store import EventReader
from .circuit_intervention_hooks import make_suppression_hook

logger = logging.getLogger(__name__)

DEFAULT_K = 20
DEFAULT_PROMPTS_PER_EDGE = 8
DEFAULT_NULL_SAMPLES = 20
WINDOW = 48  # tokens around a firing


class InterventionConfigError(ValueError):
    """Invalid validation scope/config — surfaces as a 422."""


def sigma_d_from_store(reader: EventReader, feature_idx: int) -> float:
    """Downstream feature's activation SD from the SAME store (A.5: never a
    fresh corpus). Captured events are above-threshold, so this is the SD of
    the firing distribution — the scale ES is expressed in."""
    ev = reader.feature_events(feature_idx)
    acts = np.asarray(ev["act"], dtype=np.float64)
    if len(acts) < 2:
        return 1.0  # avoid div-by-zero; a lone firing can't scale ES
    sd = float(acts.std())
    return sd if sd > 0 else 1.0


class CircuitInterventionService:
    @staticmethod
    def create_scope(config: Dict[str, Any]) -> Dict[str, Any]:
        ordering = config.get("ordering", "coact")
        if ordering not in ("coact", "attr"):
            raise InterventionConfigError(
                f"ordering must be coact|attr, got {ordering!r}")
        k = int(config.get("k", DEFAULT_K))
        if k < 1:
            raise InterventionConfigError("k must be ≥ 1")
        return {
            "ordering": ordering, "k": k,
            "prompts_per_edge": int(config.get("prompts_per_edge",
                                               DEFAULT_PROMPTS_PER_EDGE)),
            "null_samples": int(config.get("null_samples", DEFAULT_NULL_SAMPLES)),
            "percentile": float(config.get("percentile",
                                           vmath.DEFAULT_NULL_PERCENTILE)),
            "sign_frac": float(config.get("sign_frac", vmath.DEFAULT_SIGN_FRAC)),
            "baseline": config.get("baseline", "zero"),  # zero | corpus_mean
            "seed": int(config.get("seed", 0)),
        }

    @staticmethod
    def top_k_edges(candidates: List[Dict[str, Any]], ordering: str,
                    k: int) -> List[Dict[str, Any]]:
        rank_key = "attr_rank" if ordering == "attr" else "coact_rank"

        def _rank(c):
            r = (c.get("orderings") or {}).get(rank_key)
            return r if r is not None else 10**9
        ordered = sorted(candidates, key=_rank)
        # attr ordering only meaningful once attribution ran
        return ordered[:k]

    # ── worker body (GPU) ────────────────────────────────────────────────

    @staticmethod
    def run(db, run_id: str, scope: Dict[str, Any], *,
            cancel_check: Optional[Callable] = None,
            progress_cb: Optional[Callable] = None) -> Dict[str, Any]:
        import torch
        from datasets import load_from_disk

        from ..ml.layer_discovery import (discover_transformer_structure,
                                          get_hookable_module)
        from ..ml.model_loader import load_model_from_hf
        from ..models.dataset_tokenization import DatasetTokenization
        from ..models.external_sae import ExternalSAE
        from ..models.model import Model, QuantizationFormat
        from .circuit_capture_service import _load_sae_sync, _pad_batch
        from .extraction_service import cleanup_gpu_memory
        from .steering_service import resolve_decoder_weight

        run = db.query(CircuitDiscoveryRun).filter(
            CircuitDiscoveryRun.id == run_id).first()
        if run is None:
            raise ValueError(f"Discovery run {run_id} not found")
        capture = db.query(CircuitCaptureRun).filter(
            CircuitCaptureRun.id == run.capture_run_id).first()
        if capture is None or not capture.store_path:
            raise InterventionConfigError("Capture store missing")
        manifest = capture.manifest or {}
        candidates = list(run.candidates or [])
        edges = CircuitInterventionService.top_k_edges(
            candidates, scope["ordering"], scope["k"])
        if not edges:
            raise InterventionConfigError("No candidates to validate")

        store_dir = settings.resolve_data_path(capture.store_path)
        readers = {e["layer"]: EventReader(store_dir, e["layer"])
                   for e in manifest.get("layers", [])}
        sae_by_layer = {e["layer"]: e["sae_id"]
                        for e in manifest.get("layers", [])}
        tokenization = db.query(DatasetTokenization).filter(
            DatasetTokenization.id == manifest["corpus"]["tokenization_id"]).first()
        dataset = load_from_disk(
            str(settings.resolve_data_path(tokenization.tokenized_path)))

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model_record = db.query(Model).filter(
            Model.id == manifest.get("model_id")).first()
        run.validation_status = "running"
        run.validation_progress = 0.0
        db.commit()

        model = None
        saes: Dict[int, Any] = {}
        t0 = time.monotonic()
        try:
            resolved = (settings.resolve_data_path(model_record.file_path)
                        if model_record.file_path else None)
            model, tokenizer, _c, _m = load_model_from_hf(
                repo_id=model_record.repo_id,
                quant_format=QuantizationFormat(model_record.quantization),
                cache_dir=resolved, device_map=device,
                local_files_only=bool(resolved and resolved.exists()))
            model.eval()
            structure = discover_transformer_structure(model)
            for entry in manifest.get("layers", []):
                L = entry["layer"]
                sae_rec = db.query(ExternalSAE).filter(
                    ExternalSAE.id == entry["sae_id"]).first()
                saes[L] = _load_sae_sync(sae_rec, device)

            edge_results = []
            for ei, cand in enumerate(edges):
                if cancel_check is not None and cancel_check():
                    run.validation_status = "cancelled"
                    db.commit()
                    return {"status": "cancelled"}
                up, down = cand["up"], cand["down"]
                if "feature_idx" not in up or "feature_idx" not in down:
                    continue  # cluster candidates: expanded elsewhere (v1 feature edges)
                result = CircuitInterventionService._validate_edge(
                    model, structure, get_hookable_module, saes, readers,
                    dataset, tokenizer, up, down, scope, sae_by_layer, device,
                    resolve_decoder_weight)
                edge_results.append(result)
                run.validation_progress = (ei + 1) / len(edges) * 100.0
                db.commit()
                if progress_cb is not None:
                    progress_cb(run.validation_progress)

            # survival + uplift (only meaningful with BOTH orderings — record
            # this ordering's survival; uplift filled when both tiers exist)
            survived = [r for r in edge_results if r["verdict"]["passed"]]
            survival = vmath.survival_rate(
                [r["verdict"]["passed"] for r in edge_results])

            payload = {
                "intervention": {"kind": "directional_suppression",
                                 "baseline": scope["baseline"]},
                "config": scope,
                "seeds": [scope["seed"]],
                "ordering": scope["ordering"], "k": scope["k"],
                "edges": edge_results,
                "survival": survival,
                "null_summary": {"samples": scope["null_samples"],
                                 "percentile": scope["percentile"],
                                 "kind": "shuffled_non_edge_support_matched"},
            }
            from .manifest_service import ManifestService  # sync-safe import
            manifest_row = CircuitInterventionService._persist_manifest(
                db, run_id, payload)

            # write rung-2 verdicts back onto the discovery candidates
            CircuitInterventionService._write_back(
                run, edge_results, scope["ordering"], manifest_row_id=manifest_row)

            run.validation_status = "completed"
            run.validation_progress = 100.0
            run.report = {**(run.report or {}),
                          "validation": {
                              "ordering": scope["ordering"], "k": scope["k"],
                              "survival": survival,
                              "passed": len(survived),
                              "manifest_id": manifest_row,
                              "wall_clock_seconds": round(time.monotonic() - t0, 1)}}
            db.commit()
            return {"status": "completed", "validated": len(edge_results),
                    "passed": len(survived), "survival": survival,
                    "manifest_id": manifest_row}
        finally:
            cleanup_gpu_memory(
                [m for m in [model, *saes.values()] if m is not None],
                context=f"circuit_validation:{run_id}")

    # ── per-edge (GPU) ───────────────────────────────────────────────────

    @staticmethod
    def _validate_edge(model, structure, get_hookable_module, saes, readers,
                       dataset, tokenizer, up, down, scope, sae_by_layer,
                       device, resolve_decoder_weight) -> Dict[str, Any]:
        import torch

        from .circuit_capture_service import _pad_batch

        up_L, up_i = up["layer"], up["feature_idx"]
        down_L, down_i = down["layer"], down["feature_idx"]
        up_reader = readers[up_L]
        down_reader = readers[down_L]
        sigma_d = sigma_d_from_store(down_reader, down_i)

        # prompt windows around u's strongest firings
        ev = up_reader.feature_events(up_i)
        if len(ev) == 0:
            return CircuitInterventionService._empty_result(up, down, "no firings")
        order = np.argsort(ev["act"])[::-1]
        doc_ids = list(dict.fromkeys(int(ev["doc_id"][j]) for j in order))
        doc_ids = doc_ids[:scope["prompts_per_edge"]]

        up_module = get_hookable_module(structure.layers_module[up_L],
                                        "residual", structure)
        down_module = structure.layers_module[down_L]
        W_dec_up = resolve_decoder_weight(saes[up_L])

        # a_base
        a_base = 0.0
        if scope["baseline"] == "corpus_mean":
            a_base = float(np.asarray(ev["act"], dtype=np.float64).mean())

        deltas = []
        for doc_id in doc_ids:
            if doc_id >= len(dataset):
                continue
            batch = dataset[doc_id:doc_id + 1]
            input_ids, mask, lengths = _pad_batch(batch, tokenizer)
            input_ids = input_ids.to(model.device)
            mask = mask.to(model.device)
            # clean-fire positions for u: where u fired in this doc
            fire_pos = set(int(ev["token_pos"][j]) for j in range(len(ev))
                           if int(ev["doc_id"][j]) == doc_id)
            if not fire_pos:
                continue
            a_d_clean = CircuitInterventionService._downstream_acts(
                model, down_module, saes[down_L], down_i, input_ids, mask)
            # intervened: suppress u at u's layer
            enc = CircuitInterventionService._encoder_for(saes[up_L], up_i)
            handle = up_module.register_forward_hook(
                make_suppression_hook(W_dec_up, up_i, a_base=a_base,
                                      encode_fn=enc))
            try:
                a_d_int = CircuitInterventionService._downstream_acts(
                    model, down_module, saes[down_L], down_i, input_ids, mask)
            finally:
                handle.remove()
            # Δ over clean-fire tokens
            L_i = lengths[0]
            pos = [p for p in fire_pos if p < L_i]
            if not pos:
                continue
            clean = a_d_clean[0, pos].mean().item()
            interv = a_d_int[0, pos].mean().item()
            deltas.append(clean - interv)

        # shuffled-non-edge null: random support-matched u' in up's layer
        null_es = CircuitInterventionService._null_effect_sizes(
            model, structure, get_hookable_module, saes, up_reader, down_reader,
            dataset, tokenizer, up_L, down_L, down_i, doc_ids, sigma_d, scope,
            resolve_decoder_weight, exclude=up_i)

        verdict = vmath.edge_verdict(
            deltas, sigma_d, null_es,
            percentile=scope["percentile"], sign_frac=scope["sign_frac"])
        return {
            "up": up, "down": down,
            "effect_size": round(verdict.effect_size, 5),
            "sign_consistency": round(verdict.sign_consistency, 3),
            "sigma_d": round(sigma_d, 5),
            "n_prompts": len(deltas),
            "null_percentile_value": round(verdict.null_percentile_value, 5),
            "verdict": {"passed": verdict.passed, "reason": verdict.reason},
            # rung 2 iff passed; else tested_and_failed history (018 ladder)
            "rung": 2 if verdict.passed else None,
            "tested_and_failed": (not verdict.passed),
        }

    @staticmethod
    def _downstream_acts(model, down_module, sae_d, feature_idx, input_ids, mask):
        """One forward, capture down_module residual, encode, return a_d [b,s]."""
        import torch

        captured = {}

        def cap(mod, inp, out):
            captured["h"] = (out[0] if isinstance(out, tuple) else out).detach()

        h = down_module.register_forward_hook(cap)
        try:
            with torch.no_grad():
                model(input_ids=input_ids, attention_mask=mask)
        finally:
            h.remove()
        hidden = captured["h"].float()
        z = sae_d.encode(hidden)
        if isinstance(z, tuple):
            z = z[0]
        return z[..., feature_idx]

    @staticmethod
    def _encoder_for(sae, feature_idx):
        """encode_fn(hidden)->a_u for the suppression hook (same-pass encode)."""
        def enc(hidden):
            z = sae.encode(hidden.float())
            if isinstance(z, tuple):
                z = z[0]
            return z[..., feature_idx]
        return enc

    @staticmethod
    def _null_effect_sizes(model, structure, get_hookable_module, saes,
                           up_reader, down_reader, dataset, tokenizer, up_L,
                           down_L, down_i, doc_ids, sigma_d, scope,
                           resolve_decoder_weight, exclude) -> List[float]:
        """Support-matched random non-edges: pick u' in up's layer with similar
        support, suppress it, measure Δ on d — the null ES distribution."""
        import torch

        rng = np.random.default_rng(scope["seed"] + 7919)
        candidates = [f for f in up_reader.feature_ids if f != exclude]
        if not candidates:
            return []
        picks = rng.choice(candidates,
                           size=min(scope["null_samples"], len(candidates)),
                           replace=False)
        up_module = get_hookable_module(structure.layers_module[up_L],
                                        "residual", structure)
        down_module = structure.layers_module[down_L]
        W_dec_up = resolve_decoder_weight(saes[up_L])
        null = []
        for uprime in picks:
            ev = up_reader.feature_events(int(uprime))
            deltas = []
            for doc_id in doc_ids[:2]:  # cheap: 2 prompts per null sample
                if doc_id >= len(dataset):
                    continue
                batch = dataset[doc_id:doc_id + 1]
                input_ids, mask, lengths = _pad_batch_local(batch, tokenizer)
                input_ids = input_ids.to(model.device)
                mask = mask.to(model.device)
                fire_pos = [int(ev["token_pos"][j]) for j in range(len(ev))
                            if int(ev["doc_id"][j]) == doc_id
                            and int(ev["token_pos"][j]) < lengths[0]]
                if not fire_pos:
                    continue
                clean = CircuitInterventionService._downstream_acts(
                    model, down_module, saes[down_L], down_i, input_ids, mask)
                enc = CircuitInterventionService._encoder_for(saes[up_L], int(uprime))
                handle = up_module.register_forward_hook(
                    make_suppression_hook(W_dec_up, int(uprime), encode_fn=enc))
                try:
                    interv = CircuitInterventionService._downstream_acts(
                        model, down_module, saes[down_L], down_i, input_ids, mask)
                finally:
                    handle.remove()
                deltas.append(clean[0, fire_pos].mean().item()
                              - interv[0, fire_pos].mean().item())
            if deltas:
                null.append(vmath.effect_size(deltas, sigma_d))
        return null

    @staticmethod
    def _empty_result(up, down, reason):
        return {"up": up, "down": down, "effect_size": 0.0,
                "sign_consistency": 0.0, "n_prompts": 0,
                "verdict": {"passed": False, "reason": reason},
                "rung": None, "tested_and_failed": True}

    @staticmethod
    def _persist_manifest(db, run_id, payload) -> str:
        """Sync-context manifest insert (worker uses a sync session)."""
        from ..models.validation_manifest import ValidationManifest
        from .manifest_service import validate_payload
        validate_payload("edge_batch", payload)
        m = ValidationManifest(kind="edge_batch", payload=payload,
                               discovery_run_id=run_id)
        db.add(m)
        db.commit()
        db.refresh(m)
        return m.id

    @staticmethod
    def _write_back(run, edge_results, ordering, manifest_row_id):
        """Write validation onto the discovery-run candidates (rung history +
        validation field). Promoted-circuit edge writes go through
        CircuitService.write_edge_validation at the endpoint/caller (async);
        here we annotate the discovery candidates (sync JSONB is the run's own
        working data, not a circuit's contract-governed edges)."""
        by_key = {}
        for r in edge_results:
            up, down = r["up"], r["down"]
            by_key[(up.get("layer"), up.get("feature_idx"),
                    down.get("layer"), down.get("feature_idx"))] = r
        cands = [dict(c) for c in (run.candidates or [])]
        for c in cands:
            k = (c["up"].get("layer"), c["up"].get("feature_idx"),
                 c["down"].get("layer"), c["down"].get("feature_idx"))
            if k in by_key:
                r = by_key[k]
                c["validation"] = {
                    "ordering": ordering,
                    "effect_size": r["effect_size"],
                    "passed": r["verdict"]["passed"],
                    "manifest_id": manifest_row_id,
                }
                if r["rung"] == 2:
                    c["validated_rung"] = 2
                if r["tested_and_failed"]:
                    hist = c.get("tested_and_failed_history", [])
                    hist.append({"ordering": ordering,
                                 "reason": r["verdict"]["reason"]})
                    c["tested_and_failed_history"] = hist
        run.candidates = cands


def _pad_batch_local(batch, tokenizer):
    from .circuit_capture_service import _pad_batch
    return _pad_batch(batch, tokenizer)
