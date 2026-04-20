#!/usr/bin/env python3
"""
label_feature_llm.py — Auto-label a SAE feature using a two-pass LLM strategy.

Pass 1 — per-example:  For each of the top N examples, ask the LLM:
  "What is this token doing in this specific context? One sentence."

Pass 2 — synthesis:  Feed all per-example summaries back to the LLM and ask:
  "What is the unifying concept? Produce a structured label."

Intermediate summaries are saved to feature_XXXXX_summaries.txt for inspection.
Final label is written to feature_XXXXX_label.md (YAML frontmatter ready for apply_label.py).

Does NOT commit to the database. Review, then run:
    python apply_label.py feature_XXXXX_label.md --neuron XXXXX --sae-id <id> --commit

Usage:
    python label_feature_llm.py feature_09178.txt
    python label_feature_llm.py feature_09178.txt --examples 20
    python label_feature_llm.py feature_09178.txt --model gemma-4-E4B-it
"""

import argparse
import json
import os
import re
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.request import urlopen, Request
from urllib.error import URLError

MILLM_URL = "http://k8s-millm.hitsai.local/v1/chat/completions"
DEFAULT_MODEL = "gemma-4-E4B-it"
DEFAULT_TEMPERATURE = 0.1
DEFAULT_EXAMPLES = 20      # examples to summarize in pass 1
PER_EXAMPLE_MAX_TOKENS = 80
SYNTHESIS_MAX_TOKENS = 400


# ── .txt parser ───────────────────────────────────────────────────────────────

def parse_feature_txt(text: str) -> dict:
    lines = text.splitlines()

    neuron_index = None
    max_activation = None
    current_label = None

    for line in lines[:20]:
        m = re.search(r"Neuron index.*?(\d+)", line)
        if m:
            neuron_index = int(m.group(1))
        m = re.search(r"\*\*Max Activation:\*\*\s*`([\d.]+)`", line)
        if m and max_activation is None:
            max_activation = float(m.group(1))
        m = re.search(r"\*\*Current label:\*\*\s*(.+?)\s*\(", line)
        if m:
            current_label = m.group(1).strip()

    examples = []
    i = 0
    while i < len(lines):
        line = lines[i]
        m = re.match(r"^### Example (\d+) - Sample #(\S+)", line)
        if not m:
            i += 1
            continue
        n = int(m.group(1))
        sample = m.group(2)
        act_val = None
        context_line = None
        for j in range(i + 1, min(i + 5, len(lines))):
            al = lines[j]
            am = re.search(r"\*\*Max Activation:\*\*\s*`([\d.]+)`", al)
            if am:
                act_val = float(am.group(1))
            if al.startswith("> "):
                context_line = al[2:]
                break
        if context_line is not None:
            pm = re.search(r"\*\*\[(.+?)\]\*\*", context_line)
            prime = pm.group(1) if pm else ""
            prefix_part = context_line[: context_line.find("**[")]
            suffix_part = re.sub(r".*?\*\*\[.+?\]\*\*", "", context_line, count=1)
            # Reconstruct plain-text context for the LLM prompt
            plain_context = prefix_part + f"[{prime}]" + suffix_part
            examples.append({
                "n": n,
                "sample": sample,
                "activation": act_val,
                "prime": prime,
                "prefix": prefix_part,
                "suffix": suffix_part,
                "plain_context": plain_context,
            })
        i += 1

    return {
        "neuron_index": neuron_index,
        "max_activation": max_activation,
        "current_label": current_label,
        "examples": examples,
    }


# ── LLM call ─────────────────────────────────────────────────────────────────

def call_llm(prompt: str, model: str, temperature: float, max_tokens: int,
             retries: int = 3, backoff: float = 2.0) -> str:
    payload = json.dumps({
        "model": model,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": prompt}],
    }).encode()

    last_err = None
    for attempt in range(retries):
        try:
            req = Request(
                MILLM_URL,
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urlopen(req, timeout=90) as resp:
                data = json.loads(resp.read())
            return data["choices"][0]["message"]["content"].strip()
        except URLError as e:
            last_err = e
            wait = backoff * (attempt + 1)
            print(f"    [retry {attempt+1}/{retries} after {wait:.0f}s: {e}]", flush=True)
            time.sleep(wait)

    raise SystemExit(f"LLM request failed after {retries} retries: {last_err}")


# ── pass 1: per-example summarization ────────────────────────────────────────

def summarize_example(example: dict, model: str, temperature: float) -> str:
    ctx = example["plain_context"].strip()
    # Trim very long contexts to keep prompt small
    if len(ctx) > 300:
        # Keep 120 chars before and after the token
        token_pos = ctx.find(f"[{example['prime']}]")
        if token_pos > 0:
            start = max(0, token_pos - 120)
            end = min(len(ctx), token_pos + len(example["prime"]) + 2 + 120)
            ctx = ("..." if start > 0 else "") + ctx[start:end] + ("..." if end < len(ctx) else "")

    prompt = (
        f"A language-model feature fires on the token [{example['prime']}] "
        f"(activation strength: {example['activation']:.2f}) in this passage:\n\n"
        f"  {ctx}\n\n"
        f"In ONE sentence, describe what linguistic or semantic role [{example['prime']}] "
        f"is playing in this specific context. Be precise and concrete."
    )
    return call_llm(prompt, model, temperature, PER_EXAMPLE_MAX_TOKENS)


# ── pass 2: synthesis ─────────────────────────────────────────────────────────

def synthesize_label(feature: dict, summaries: list[tuple[dict, str]], model: str, temperature: float) -> tuple[dict, str]:
    examples = feature["examples"]
    counter = Counter(e["prime"] for e in examples if e["prime"])
    freq_lines = []
    for token, count in counter.most_common(15):
        pct = 100.0 * count / len(examples)
        freq_lines.append(f"  {count:3d}x ({pct:4.0f}%)  [{token}]")

    summary_block = "\n".join(
        f"{i+1:3d}. [act={ex['activation']:.2f}, token={ex['prime']!r}] {note}"
        for i, (ex, note) in enumerate(summaries)
    )

    prompt = (
        f"You are analyzing a sparse autoencoder feature from a language model.\n"
        f"The feature fires on specific tokens. You have examined {len(summaries)} examples "
        f"and written one-sentence observations for each.\n\n"
        f"PRIME TOKEN FREQUENCIES (across all {len(examples)} examples):\n"
        + "\n".join(freq_lines)
        + f"\n\nPER-EXAMPLE OBSERVATIONS:\n{summary_block}\n\n"
        f"Based on these observations, identify the single unifying concept this feature has learned.\n"
        f"Produce a JSON object with:\n"
        f'  "name"        : short snake_case slug (max 5 words)\n'
        f'  "category"    : broader snake_case category slug\n'
        f'  "description" : ONE precise sentence describing the firing pattern\n'
        f'  "confidence"  : "high", "medium", or "low"\n\n'
        f"Respond with ONLY the JSON object."
    )

    raw = call_llm(prompt, model, temperature, SYNTHESIS_MAX_TOKENS)
    label = extract_json(raw)
    return label, raw


# ── JSON extractor ────────────────────────────────────────────────────────────

def extract_json(text: str) -> dict:
    text = re.sub(r"^```(?:json)?\s*", "", text.strip(), flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text.strip())
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    result = {}
    for field in ("name", "category", "description", "confidence"):
        m = re.search(rf'"{field}"\s*:\s*"((?:[^"\\]|\\.)*)"', text)
        if m:
            result[field] = m.group(1).replace('\\"', '"')
    if not result:
        raise SystemExit(f"Could not parse JSON from LLM response:\n{text}")
    return result


# ── output writers ────────────────────────────────────────────────────────────

def write_summaries_txt(out_path: str, feature: dict, summaries: list[tuple[dict, str]]) -> None:
    lines = [
        f"# Feature #{feature['neuron_index']:05d} — Per-Example Summaries (Pass 1)",
        f"",
        f"Max activation : {feature['max_activation']:.4f}",
        f"Current label  : {feature['current_label']}",
        f"Examples used  : {len(summaries)}",
        f"",
        "---",
        "",
    ]
    for ex, note in summaries:
        lines += [
            f"### Example {ex['n']}  [act={ex['activation']:.2f}]",
            f"**Token:** `[{ex['prime']}]`",
            f"**Context:** ...{ex['prefix'][-80:]}[{ex['prime']}]{ex['suffix'][:80]}...",
            f"**Summary:** {note}",
            "",
        ]
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def _yaml_scalar(value: str) -> str:
    if any(c in value for c in ('"', "'", ":", "#", "\n")):
        return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'
    return value


def write_label_md(out_path: str, feature: dict, label: dict,
                   summaries: list[tuple[dict, str]], raw_synthesis: str) -> None:
    name = label.get("name", "").strip().replace(" ", "_").lower()
    category = label.get("category", "").strip().replace(" ", "_").lower()
    description = label.get("description", "").strip()
    confidence = label.get("confidence", "medium").strip().lower()
    neuron = feature["neuron_index"]

    content = f"""---
name: {_yaml_scalar(name)}
category: {_yaml_scalar(category)}
description: {_yaml_scalar(description)}
---

## Feature #{neuron:05d} Label  *(LLM two-pass — review before committing)*

**Proposed label:** {name}
**Confidence:** {confidence}
**Examples summarized:** {len(summaries)} of {len(feature['examples'])}

### Pass 2 — Synthesis (raw LLM response)

```
{raw_synthesis}
```

### Pass 1 — Per-example summaries

| Ex | Act | Token | Summary |
|----|-----|-------|---------|
"""
    for ex, note in summaries:
        token_cell = ex['prime'].replace("|", "\\|")
        note_cell = note.replace("|", "\\|")
        content += f"| {ex['n']:3d} | {ex['activation']:.2f} | `{token_cell}` | {note_cell} |\n"

    content += f"""
---

*Generated by label_feature_llm.py (two-pass) using {DEFAULT_MODEL}.*
*To commit: `python apply_label.py {os.path.basename(out_path)} --neuron {neuron} --sae-id <sae_id> --commit`*
"""

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(content)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("txt_file", help="Path to feature_XXXXX.txt file")
    parser.add_argument("--examples", type=int, default=DEFAULT_EXAMPLES,
                        help=f"Number of examples to summarize in pass 1 (default {DEFAULT_EXAMPLES})")
    parser.add_argument("--workers", type=int, default=DEFAULT_EXAMPLES,
                        help=f"ThreadPoolExecutor workers for pass 1 (default: same as --examples, i.e. fully parallel)")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--out", default=None, help="Output directory (default: same as input file)")
    args = parser.parse_args()

    if not os.path.exists(args.txt_file):
        raise SystemExit(f"File not found: {args.txt_file}")

    with open(args.txt_file, encoding="utf-8") as f:
        text = f.read()

    print(f"Parsing {args.txt_file}...")
    feature = parse_feature_txt(text)

    if feature["neuron_index"] is None:
        raise SystemExit("Could not parse neuron_index from .txt file")
    if not feature["examples"]:
        raise SystemExit("No activation examples found in .txt file")

    print(f"  Neuron : {feature['neuron_index']}  |  Max act: {feature['max_activation']}  |  {len(feature['examples'])} examples total")

    # Top N by activation (already sorted in the .txt)
    top_examples = feature["examples"][: args.examples]

    # ── Pass 1 — parallel per-example summarization ─────────────────────────
    print(f"\nPass 1 — summarizing {len(top_examples)} examples in parallel (workers={args.workers})...")
    t_pass1_start = time.perf_counter()

    # Preserve original ordering after parallel execution
    results: dict[int, str] = {}

    def _summarize(ex):
        note = summarize_example(ex, args.model, args.temperature)
        note = re.sub(r'^["\'`]+|["\'`]+$', "", note.strip())
        return ex, note

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_summarize, ex): ex for ex in top_examples}
        for future in as_completed(futures):
            ex, note = future.result()
            results[ex["n"]] = (ex, note)
            print(f"  Ex {ex['n']:3d} [{ex['activation']:.2f}] [{ex['prime']}]  →  {note[:100]}")

    # Re-sort by original example order
    summaries: list[tuple[dict, str]] = [results[ex["n"]] for ex in top_examples]

    t_pass1 = time.perf_counter() - t_pass1_start
    print(f"  Pass 1 done in {t_pass1:.1f}s")

    out_dir = args.out or os.path.dirname(os.path.abspath(args.txt_file))
    summaries_path = os.path.join(out_dir, f"feature_{feature['neuron_index']:05d}_summaries.txt")
    write_summaries_txt(summaries_path, feature, summaries)
    print(f"Summaries written → {summaries_path}")

    # ── Pass 2 ──────────────────────────────────────────────────────────────
    print(f"\nPass 2 — synthesizing label from {len(summaries)} summaries...")
    t_pass2_start = time.perf_counter()
    label, raw_synthesis = synthesize_label(feature, summaries, args.model, args.temperature)
    t_pass2 = time.perf_counter() - t_pass2_start
    t_total = t_pass1 + t_pass2

    print(f"  Pass 2 done in {t_pass2:.1f}s")
    print(f"\nTiming summary:")
    print(f"  Pass 1 (parallel, {len(top_examples)} examples, {args.workers} workers): {t_pass1:.1f}s")
    print(f"  Pass 2 (synthesis):                                    {t_pass2:.1f}s")
    print(f"  Total:                                                 {t_total:.1f}s")
    print(f"\nRaw synthesis response:\n{raw_synthesis}\n")
    print("Parsed label:")
    for k, v in label.items():
        print(f"  {k:12s}: {v!r}")

    out_name = f"feature_{feature['neuron_index']:05d}_label.md"
    out_path = os.path.join(out_dir, out_name)
    write_label_md(out_path, feature, label, summaries, raw_synthesis)
    print(f"\nLabel written → {out_path}")
    print(f"\nReview, then commit:")
    print(f"  python apply_label.py {out_name} --neuron {feature['neuron_index']} --sae-id <sae_id> --commit")


if __name__ == "__main__":
    main()
