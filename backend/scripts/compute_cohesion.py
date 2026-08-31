"""Compute embedding cohesion per feature and validate it against consensus.

Cohesion replaces an unstable model judgement with a computed number. It is
validated against the CONSENSUS set built by build_consensus_set.py — the
features where gemma-4-12B-it gave the same verdict across three perturbed runs
— because a single verdict is not ground truth: the model flips on 47% of
features when the passages are merely reordered.

The baseline is computed ONCE over random passages from the same extraction and
subtracted from every feature. Without it the score measures "this corpus
resembles itself" and a feature firing on "the" ranks near the top.

Usage, inside the mistudio backend pod:
    env PYTHONPATH=/app python3 /tmp/compute_cohesion.py [n_features]

Env:
    EMBED_MODEL      default Nemotron-3-Embed-8B-BF16
    CONSENSUS_IN     default /data/consensus_set.json
    COHESION_OUT     default /data/cohesion_scores.json
"""

import json
import os
import random
import sys
import time

import httpx
from sqlalchemy import text

from src.core.database import SyncSessionLocal
from src.services.feature_cohesion import (
    cohesion_score,
    mean_pairwise_cosine,
    summarise,
)

EXTRACTION = os.environ.get("EXTRACTION", "extr_20260828_080834_sae_sae_39cc_002")
ENDPOINT = os.environ.get(
    "MILLM_ENDPOINT", "http://millm-backend.millm.svc.cluster.local:8000/v1")
EMBED_MODEL = os.environ.get("EMBED_MODEL", "Nemotron-3-Embed-8B-BF16")
CONSENSUS_IN = os.environ.get("CONSENSUS_IN", "/data/consensus_set.json")
OUT = os.environ.get("COHESION_OUT", "/data/cohesion_scores.json")
N_PASSAGES = 10
N_BASELINE = 60
EMBED_BATCH = 16


def render(row):
    """Plain text, no markers, no activation values.

    Identical treatment for a feature's own passages and for the baseline
    passages — if the two were rendered differently, the contrast would measure
    the formatting difference rather than the feature.
    """
    def clean(t):
        return str(t).replace("▁", " ").replace("Ġ", " ").replace("##", "")
    pre, prime, suf = row[0] or [], row[1] or "", row[2] or []
    return " ".join(
        ("".join(clean(t) for t in pre) + clean(prime) +
         "".join(clean(t) for t in suf)).split()
    )


def embed(client, texts):
    """Embed a list of strings, batched. Returns list-of-vectors (None on failure)."""
    out = []
    for i in range(0, len(texts), EMBED_BATCH):
        chunk = texts[i:i + EMBED_BATCH]
        try:
            r = client.post(f"{ENDPOINT}/embeddings",
                            json={"model": EMBED_MODEL, "input": chunk})
            if r.status_code != 200:
                print(f"    embed HTTP {r.status_code}: {r.text[:160]}")
                out.extend([None] * len(chunk))
                continue
            data = r.json().get("data") or []
            # Order is not promised by the schema; sort by index like any
            # multi-item OpenAI response.
            data = sorted(data, key=lambda d: d.get("index", 0))
            vecs = [d.get("embedding") for d in data]
            if len(vecs) != len(chunk):
                print(f"    embed returned {len(vecs)} of {len(chunk)}")
                vecs += [None] * (len(chunk) - len(vecs))
            out.extend(vecs)
        except Exception as exc:
            print(f"    embed error: {type(exc).__name__}: {exc}")
            out.extend([None] * len(chunk))
    return out


def main():
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    db = SyncSessionLocal()

    consensus = None
    try:
        c = json.load(open(CONSENSUS_IN))
        if c.get("extraction") == EXTRACTION:
            consensus = c
        else:
            print(f"note: consensus set is for {c.get('extraction')}, not this "
                  f"extraction — running DISTRIBUTION ONLY (no AUC)")
    except Exception:
        print("note: no consensus set for this extraction — DISTRIBUTION ONLY "
              "(no AUC)")

    if consensus:
        feats = [f for f in consensus["features"] if f.get("unanimous")]
    else:
        # Even stride across the whole extraction, same sampling as the
        # consensus builder so the layers are compared like for like.
        total = db.execute(text(
            "SELECT count(*) FROM features WHERE extraction_job_id=:e"),
            {"e": EXTRACTION}).scalar()
        want = limit or 112          # match the L46 sample size by default
        stride = max(1, total // want)
        rows_f = db.execute(text("""
            SELECT id FROM features WHERE extraction_job_id = :e
              AND MOD(neuron_index, :s) = 0 ORDER BY neuron_index LIMIT :n
        """), {"e": EXTRACTION, "s": stride, "n": want}).fetchall()
        feats = [{"feature_id": r[0], "consensus": None, "unanimous": True}
                 for r in rows_f]
    if limit:
        feats = feats[:limit]
    print(f"embedder   : {EMBED_MODEL}")
    print(f"extraction : {EXTRACTION}")
    if consensus:
        print(f"consensus  : {len(feats)} unanimous features "
              f"({sum(1 for f in feats if f['consensus'])} refused / "
              f"{sum(1 for f in feats if f['consensus'] is False)} labeled)")
    else:
        print(f"sample     : {len(feats)} features (no ground truth available)")

    with httpx.Client(timeout=300.0) as client:
        # ---- corpus baseline, computed once -----------------------------
        # DISTINCT ON (sample_index): one passage per DOCUMENT.
        #
        # The first version ordered by MOD(sample_index, 977), which selects the
        # few documents where that expression is smallest and then takes many
        # features' activations on those same documents — measured: 180 rows
        # from 11 distinct documents, one contributing 28 passages. A "random"
        # baseline built from 11 documents is highly cohesive by construction,
        # which inflated the baseline and pushed every feature's score negative.
        # It did not affect AUC (a constant shift cancels in a rank metric) but
        # it made the absolute numbers meaningless.
        rows = db.execute(text("""
            SELECT DISTINCT ON (fa.sample_index)
                   fa.prefix_tokens, fa.prime_token, fa.suffix_tokens
            FROM feature_activations fa JOIN features f ON f.id = fa.feature_id
            WHERE f.extraction_job_id = :e
            ORDER BY fa.sample_index, fa.feature_id
            LIMIT :n
        """), {"e": EXTRACTION, "n": N_BASELINE * 3}).fetchall()
        pool = [render(r) for r in rows]
        random.Random(20260831).shuffle(pool)
        pool = [p for p in pool if p][:N_BASELINE]
        print(f"baseline   : {len(pool)} random passages from the same corpus",
              flush=True)
        t0 = time.time()
        base_vecs = embed(client, pool)
        baseline = mean_pairwise_cosine([v for v in base_vecs if v])
        if baseline is None:
            print("BASELINE FAILED — cannot score without a contrast.")
            return 1
        print(f"             baseline cohesion = {baseline:.4f} "
              f"({time.time()-t0:.0f}s)\n", flush=True)

        # ---- per feature -------------------------------------------------
        scores, own_raw = {}, {}
        for n, f in enumerate(feats, 1):
            ex = db.execute(text("""
                SELECT prefix_tokens, prime_token, suffix_tokens
                FROM feature_activations WHERE feature_id = :f
                ORDER BY max_activation DESC LIMIT :k
            """), {"f": f["feature_id"], "k": N_PASSAGES}).fetchall()
            texts = [t for t in (render(r) for r in ex) if t]
            vecs = embed(client, texts) if texts else []
            own = mean_pairwise_cosine([v for v in vecs if v])
            own_raw[f["feature_id"]] = own
            scores[f["feature_id"]] = cohesion_score(own, baseline)
            if n % 20 == 0:
                print(f"  {n}/{len(feats)} ({time.time()-t0:.0f}s)", flush=True)

    json.dump({"extraction": EXTRACTION, "embedder": EMBED_MODEL,
               "baseline": baseline,
               "scores": scores, "own_cohesion": own_raw},
              open(OUT, "w"), indent=1)
    print(f"\nwrote {OUT}")

    # ---- validation against consensus ------------------------------------
    ref = [scores[f["feature_id"]] for f in feats
           if f.get("consensus") is True and scores.get(f["feature_id"]) is not None]
    lab = [scores[f["feature_id"]] for f in feats
           if f.get("consensus") is False and scores.get(f["feature_id"]) is not None]

    print(f"\n{'='*74}")
    print("distribution of cohesion score (own - baseline):")
    print(f"  all      : {summarise(scores)}")

    def auc(pos, neg):
        if not pos or not neg:
            return None
        wins = sum(1.0 if a > b else (0.5 if a == b else 0.0)
                   for a in pos for b in neg)
        return wins / (len(pos) * len(neg))

    a = auc(lab, ref)
    print(f"\nSEPARATION against the consensus verdict")
    print(f"  consensus-LABELED  n={len(lab)}  mean={sum(lab)/max(len(lab),1):+.4f}")
    print(f"  consensus-REFUSED  n={len(ref)}  mean={sum(ref)/max(len(ref),1):+.4f}")
    if a is None:
        print("  AUC: not computable (a class is empty)")
    else:
        print(f"  AUC = {a:.3f}   "
              f"({'USEFUL' if a >= 0.70 else 'marginal' if a >= 0.65 else 'no better than chance'})")
        print("\n  (0.50 = noise, 0.65 = the gate this project uses, 0.70+ = useful)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
