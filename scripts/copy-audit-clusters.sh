#!/usr/bin/env bash
# Feature 012 acceptance audit: no user-facing "Feature Group(s)" copy may remain.
#
# Checks (1) the BUILT frontend bundle and (2) manual page bodies.
# Allowlisting happens at CONTENT level, never on whole grep lines — a previous
# version filtered `grep -rn` output with `grep -v feature-groups`, which
# excluded every hit inside feature-groups.md via its own FILE PATH (masking
# real violations). The slug `feature-groups` and identifiers like
# `feature_groups`/`FeatureGroupsPanel` are legitimate internals; the spaced,
# human-facing phrase "feature group(s)" (any case) is not.
set -uo pipefail
cd "$(dirname "$0")/.."

fail=0

echo "== bundle audit (frontend/dist) =="
if ls frontend/dist/assets/*.js >/dev/null 2>&1; then
  # Case-insensitive: sentence templates may compose lowercase copy.
  hits=$(grep -lioE "feature groups?\b" frontend/dist/assets/*.js 2>/dev/null || true)
  if [ -n "$hits" ]; then
    echo "FAIL: user-facing 'feature group(s)' found in built bundle:"; echo "$hits"; fail=1
  else
    echo "OK: no 'feature group(s)' in built bundle"
  fi
else
  echo "SKIP: no build output (run npm run build first)"; fail=1
fi

echo "== manual audit =="
# -h drops filenames so path text can never mask or trigger matches; report
# per-file below. Strip legitimate slug/anchor tokens from CONTENT before
# matching so a line with both a slug link and forbidden copy is still caught.
manual_fail=0
while IFS= read -r -d '' f; do
  cleaned=$(sed -e 's/feature-groups//g' -e 's/feature_groups//g' "$f")
  if printf '%s' "$cleaned" | grep -qiE "feature groups?\b"; then
    echo "FAIL: $f still contains user-facing 'feature group(s)':"
    printf '%s\n' "$cleaned" | grep -inE "feature groups?\b" | head -5
    manual_fail=1
  fi
done < <(find manual/docs -name '*.md' -print0)
if [ "$manual_fail" -eq 0 ]; then echo "OK: manual clean"; else fail=1; fi

exit $fail
