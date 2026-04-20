#!/usr/bin/env python3
"""
fetch_feature.py — Pull activation examples for a single SAE feature from
the miStudio database and write a formatted text file ready for labeling.

Usage:
    python fetch_feature.py <neuron_index> [--sae <name_fragment>] [--max <N>] [--out <dir>]

Defaults:
    --sae   "LFM2.5-1.2B-Instruct"
    --max   100
    --out   . (same directory as this script)

Examples:
    python fetch_feature.py 9178
    python fetch_feature.py 10023 --max 50
    python fetch_feature.py 9178 --sae "L12-residual"
"""

import argparse
import json
import os
import sys

import psycopg2
import psycopg2.extras

DB_URL = "postgresql://mistudio:mistudio@localhost:5435/mistudio"

# ── helpers ──────────────────────────────────────────────────────────────────

def connect():
    return psycopg2.connect(DB_URL, cursor_factory=psycopg2.extras.RealDictCursor)


def find_sae(cur, name_fragment: str) -> dict:
    cur.execute(
        """
        SELECT id, name, model_name, layer, n_features
        FROM external_saes
        WHERE name ILIKE %s OR description ILIKE %s
        ORDER BY created_at DESC
        LIMIT 10
        """,
        (f"%{name_fragment}%", f"%{name_fragment}%"),
    )
    rows = cur.fetchall()
    if not rows:
        raise SystemExit(f"No SAE found matching '{name_fragment}'")
    if len(rows) == 1:
        return rows[0]
    print("Multiple SAEs matched — pick one:\n")
    for i, r in enumerate(rows):
        print(f"  [{i}] {r['id']}  {r['name']}  (layer {r['layer']}, {r['n_features']} features)")
    choice = int(input("\nEnter index: "))
    return rows[choice]


def find_feature(cur, sae_id: str, neuron_index: int) -> dict:
    cur.execute(
        """
        SELECT id, name, max_activation, activation_frequency, interpretability_score,
               label_source, nlp_analysis
        FROM features
        WHERE external_sae_id = %s AND neuron_index = %s
        """,
        (sae_id, neuron_index),
    )
    row = cur.fetchone()
    if not row:
        raise SystemExit(
            f"Feature neuron_index={neuron_index} not found for SAE {sae_id}"
        )
    return row


def fetch_activations(cur, feature_id: str, max_rows: int) -> list:
    cur.execute(
        """
        SELECT sample_index, max_activation,
               prefix_tokens, prime_token, suffix_tokens, prime_activation_index,
               tokens, activations
        FROM feature_activations
        WHERE feature_id = %s
        ORDER BY max_activation DESC
        LIMIT %s
        """,
        (feature_id, max_rows),
    )
    return cur.fetchall()


_BPE_CHAR_MAP = {
    "\u0120": " ",   # Ġ  → space (BPE space-prefix marker)
    "\u010a": "\n",  # Ċ  → newline
    "\u0109": "\t",  # ĉ  → tab
    # Mojibake sequences for Windows-1252 / Latin-1 smart punctuation stored
    # as raw UTF-8 bytes and then mis-decoded.  Cover the most common ones:
    "\u00e2\u0080\u0099": "'",   # â€™ → right single quote
    "\u00e2\u0080\u0098": "'",   # â€˜ → left single quote
    "\u00e2\u0080\u009c": '"',   # â€œ → left double quote
    "\u00e2\u0080\u009d": '"',   # â€  → right double quote
    "\u00e2\u0080\u0093": "–",   # â€" → en-dash
    "\u00e2\u0080\u0094": "—",   # â€" → em-dash
    "\u00e2\u0080\u00a6": "…",   # â€¦ → ellipsis
}

# Multi-char replacement sequences (need to be applied before single-char)
_MOJIBAKE = [
    ("âĢĻ", "'"),
    ("âĢĺ", "\u2018"),
    ("âĢĻ", "\u2019"),
    ("âĢľ", "\u201c"),
    ("âĢĿ", "\u201d"),
    ("âĢĶ", "\u2014"),
    ("âĢĵ", "\u2013"),
    ("âĢ¦", "\u2026"),
    ("âĢ¢", "\u2022"),
    ("Ã©", "é"),
    ("Ã¨", "è"),
    ("Ã ", "à"),
    ("Ã¼", "ü"),
    ("Ã¶", "ö"),
    ("Ã¤", "ä"),
    ("Ã", "Ã"),  # leave unknown Ã sequences as-is after others handled
]


def clean_token(t: str) -> str:
    """Replace BPE markers and common mojibake sequences with readable characters."""
    for bad, good in _MOJIBAKE:
        t = t.replace(bad, good)
    # Replace the Ġ space-prefix and other BPE control chars
    result = []
    for ch in t:
        result.append(_BPE_CHAR_MAP.get(ch, ch))
    return "".join(result)


def clean_tokens(tokens: list) -> list:
    return [clean_token(t) for t in tokens]


def reconstruct_context(row: dict) -> tuple[str, str, str]:
    """Return (prefix_text, prime_token, suffix_text) from enhanced or legacy format."""
    # Enhanced format
    if row["prime_token"] is not None:
        prefix = row["prefix_tokens"] or []
        if isinstance(prefix, str):
            prefix = json.loads(prefix)
        suffix = row["suffix_tokens"] or []
        if isinstance(suffix, str):
            suffix = json.loads(suffix)
        return (
            "".join(clean_tokens(prefix)),
            clean_token(row["prime_token"]),
            "".join(clean_tokens(suffix)),
        )

    # Legacy format — find max-activation token
    tokens = row["tokens"]
    activations_vals = row["activations"]
    if isinstance(tokens, str):
        tokens = json.loads(tokens)
    if isinstance(activations_vals, str):
        activations_vals = json.loads(activations_vals)

    if not tokens:
        return "", "?", ""

    peak_idx = max(range(len(activations_vals)), key=lambda i: activations_vals[i])
    prefix_tokens = tokens[max(0, peak_idx - 25): peak_idx]
    prime = tokens[peak_idx]
    suffix_tokens = tokens[peak_idx + 1: peak_idx + 26]
    return (
        "".join(clean_tokens(prefix_tokens)),
        clean_token(prime),
        "".join(clean_tokens(suffix_tokens)),
    )


# ── formatting ────────────────────────────────────────────────────────────────

def format_file(feature: dict, sae: dict, activations: list, neuron_index: int) -> str:
    max_act = feature["max_activation"]
    lines = [
        f"# Feature #{neuron_index:05d}",
        f"",
        f"**SAE:** {sae['name']}",
        f"**Neuron index:** {neuron_index}",
        f"**Feature ID:** {feature['id']}",
        f"**Max Activation:** `{max_act:.4f}`",
        f"**Activation frequency:** {feature['activation_frequency']:.4f}",
        f"**Interpretability score:** {feature['interpretability_score']:.4f}",
        f"**Current label:** {feature['name']}  (source: {feature['label_source']})",
        f"**Examples included:** {len(activations)}",
        f"",
        "---",
        "",
    ]

    for n, row in enumerate(activations, start=1):
        prefix, prime, suffix = reconstruct_context(row)
        act_val = row["max_activation"]
        sample = row["sample_index"] if row["sample_index"] is not None else "?"
        lines += [
            f"### Example {n} - Sample #{sample}",
            f"**Max Activation:** `{act_val:.4f}`",
            f"> {prefix}**[{prime.lstrip()}]**{suffix}",
            "",
        ]

    return "\n".join(lines)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("neuron_index", type=int, help="Feature neuron index")
    parser.add_argument("--sae", default="LFM2.5-1.2B-Instruct", help="SAE name fragment to match")
    parser.add_argument("--sae-id", dest="sae_id", default=None, help="Exact SAE ID (skips interactive selection)")
    parser.add_argument("--max", type=int, default=100, dest="max_examples", help="Max activation examples (default 100)")
    parser.add_argument("--out", default=os.path.dirname(os.path.abspath(__file__)), help="Output directory")
    args = parser.parse_args()

    conn = connect()
    cur = conn.cursor()

    if args.sae_id:
        cur.execute("SELECT id, name, layer, n_features FROM external_saes WHERE id = %s", (args.sae_id,))
        sae = cur.fetchone()
        if not sae:
            raise SystemExit(f"No SAE found with id '{args.sae_id}'")
        print(f"Using SAE: {sae['id']}  \"{sae['name']}\"")
    else:
        print(f"Searching for SAE matching '{args.sae}'...")
        sae = find_sae(cur, args.sae)
    print(f"  Found: {sae['id']}  \"{sae['name']}\"  ({sae['n_features']} features)")

    print(f"Looking up feature neuron_index={args.neuron_index}...")
    feature = find_feature(cur, sae["id"], args.neuron_index)
    print(f"  Found: {feature['id']}")
    print(f"  Max activation: {feature['max_activation']:.4f}")
    print(f"  Current label:  {feature['name']} ({feature['label_source']})")

    print(f"Fetching up to {args.max_examples} activation examples...")
    activations = fetch_activations(cur, feature["id"], args.max_examples)
    print(f"  Got {len(activations)} rows")

    conn.close()

    content = format_file(feature, sae, activations, args.neuron_index)

    out_path = os.path.join(args.out, f"feature_{args.neuron_index:05d}.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"Written → {out_path}")


if __name__ == "__main__":
    main()
