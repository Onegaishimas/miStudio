#!/usr/bin/env python3
"""
apply_label.py — Write a label from a _label.md frontmatter block to the database.

Reads the YAML frontmatter of a label file and updates the matching feature row with:
  name, category, description, notes, label_source='user', labeled_at, updated_at

Dry-run by default. Pass --commit to write.

Usage:
    python apply_label.py feature_09178_label.md --neuron 9178 --sae-id sae_eb8374929894
    python apply_label.py feature_09178_label.md --neuron 9178 --sae-id sae_eb8374929894 --commit
"""

import argparse
import re
import sys
from datetime import datetime, timezone

import psycopg2
import psycopg2.extras

DB_URL = "postgresql://mistudio:mistudio@localhost:5435/mistudio"


# ── YAML frontmatter parser (no PyYAML dependency) ────────────────────────────

def parse_frontmatter(text: str) -> dict:
    """
    Extract the YAML frontmatter block between the first two '---' lines.
    Supports scalar fields and block scalars (| style, indented).
    Returns a dict with keys: name, category, description, notes.
    """
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        raise SystemExit("No YAML frontmatter found (file must start with '---')")

    # Collect frontmatter lines
    fm_lines = []
    for line in lines[1:]:
        if line.strip() == "---":
            break
        fm_lines.append(line)

    result = {}
    i = 0
    while i < len(fm_lines):
        line = fm_lines[i]
        # Match "key: value" or "key: |"
        m = re.match(r'^(\w+):\s*(.*)', line)
        if not m:
            i += 1
            continue
        key, value = m.group(1), m.group(2).strip()
        if value == "|":
            # Block scalar — collect indented lines that follow
            block = []
            i += 1
            while i < len(fm_lines):
                bline = fm_lines[i]
                if bline and not bline[0].isspace():
                    break  # Next top-level key
                block.append(bline.strip())
                i += 1
            # Strip trailing blank lines
            while block and not block[-1]:
                block.pop()
            result[key] = "\n".join(block)
        else:
            # Strip surrounding quotes if present
            value = value.strip('"').strip("'")
            result[key] = value
            i += 1

    return result


# ── DB helpers ────────────────────────────────────────────────────────────────

def connect():
    return psycopg2.connect(DB_URL, cursor_factory=psycopg2.extras.RealDictCursor)


def lookup_feature(cur, sae_id: str, neuron_index: int) -> dict:
    cur.execute(
        """
        SELECT id, name, category, description, notes, label_source
        FROM features
        WHERE external_sae_id = %s AND neuron_index = %s
        """,
        (sae_id, neuron_index),
    )
    row = cur.fetchone()
    if not row:
        raise SystemExit(f"Feature neuron_index={neuron_index} not found for SAE {sae_id}")
    return row


def apply_label(cur, feature_id: str, fm: dict, now: datetime) -> None:
    cur.execute(
        """
        UPDATE features
        SET name         = %s,
            category     = %s,
            description  = %s,
            notes        = %s,
            label_source = 'user',
            labeled_at   = %s,
            updated_at   = %s
        WHERE id = %s
        """,
        (
            fm["name"],
            fm.get("category"),
            fm.get("description"),
            fm.get("notes"),
            now,
            now,
            feature_id,
        ),
    )


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("label_file", help="Path to the _label.md file with YAML frontmatter")
    parser.add_argument("--neuron", type=int, required=True, help="Feature neuron index")
    parser.add_argument("--sae-id", dest="sae_id", required=True, help="SAE ID (e.g. sae_eb8374929894)")
    parser.add_argument("--commit", action="store_true", help="Write to database (default: dry run)")
    args = parser.parse_args()

    with open(args.label_file, encoding="utf-8") as f:
        text = f.read()

    fm = parse_frontmatter(text)

    required = ["name", "description"]
    missing = [k for k in required if not fm.get(k)]
    if missing:
        raise SystemExit(f"Frontmatter missing required fields: {missing}")

    conn = connect()
    cur = conn.cursor()
    feature = lookup_feature(cur, args.sae_id, args.neuron)

    print(f"\nFeature: {feature['id']}")
    print(f"  Current name        : {feature['name']!r}")
    print(f"  Current category    : {feature['category']!r}")
    print(f"  Current label_source: {feature['label_source']!r}")
    print()
    print(f"  → new name          : {fm['name']!r}")
    print(f"  → new category      : {fm.get('category')!r}")
    print(f"  → new description   : {fm.get('description', '')[:80]!r}{'...' if len(fm.get('description','')) > 80 else ''}")
    print(f"  → new notes         : {(fm.get('notes','')[:60] + '...') if fm.get('notes') else None!r}")
    print(f"  → label_source      : 'user'")

    if not args.commit:
        print("\n[DRY RUN] No changes written. Pass --commit to apply.")
        conn.close()
        return

    now = datetime.now(timezone.utc)
    apply_label(cur, feature["id"], fm, now)
    conn.commit()
    print(f"\n[COMMITTED] Feature {args.neuron} updated successfully.")
    conn.close()


if __name__ == "__main__":
    main()
