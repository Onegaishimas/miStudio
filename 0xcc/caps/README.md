# Screen Captures

Reference screenshots of the miStudio UI panels, captured from the live K8s
deployment (`http://k8s-mistudio.hitsai.local/`) with the Playwright skill
(headless Chromium, 1600×1000 viewport).

Filenames are `{panel}_{YYYYMMDD}.png`. Re-run a capture set with the script in
"How to refresh" below; keep the newest date and prune older ones as the UI
evolves.

## 2026-07-14

| File | Panel | Notes |
|------|-------|-------|
| `models_20260714.png` | Models | Model download/quantization/architecture |
| `datasets_20260714.png` | Datasets | HF download + per-model tokenization |
| `training_20260714.png` | Training | SAE training jobs + metrics |
| `extractions_20260714.png` | Extractions | Feature/activation extraction jobs + browser |
| `labeling_20260714.png` | Labeling | Bulk auto-labeling jobs |
| `feature-groups_20260714.png` | Feature Groups | Cross-feature grouping (Feature 010) — cleaned tokens + pinned header/scrolling list |
| `saes_20260714.png` | SAEs | Trained + external SAE management |
| `steering_20260714.png` | Steering | Feature steering ("select an SAE to begin" default state) |
| `templates_20260714.png` | Templates | Training/extraction/labeling/steering presets |
| `monitor_20260714.png` | Monitor | Live GPU/CPU/RAM/disk/network (RTX 3090) |
| `settings_20260714.png` | Settings | Endpoints / API keys / labeling / display (PIN-gated) |

## Documentation screenshots (2026-07-14)

Higher-value captures driven to specific UI states, used to illustrate the
manual's previously image-less pages. Naming follows the manual's
`miStudio_{Area}_{View}` convention. **Deploy copies** of the embedded subset
also live in `manual/static/img/` — the manual site can only serve from there
(`0xcc/` is stripped from the public mirror), so `caps/` holds the canonical
originals and `manual/static/img/` holds the served copies.

| File (caps/) | State captured | Embedded in |
|--------------|----------------|-------------|
| `miStudio_FeatureGroups_Panel-Browse_*.jpg` | Groups list, pinned header + filter bar, cleaned tokens | `core-workflow/feature-groups.md` |
| `miStudio_FeatureGroups_Panel-ExpandedGroup_*.jpg` | Group expanded: member table, select-all checkbox, `*token*` snippets, sim scores | `core-workflow/feature-groups.md` |
| `miStudio_Monitor_Panel-Resources_*.jpg` | Live CPU/RAM/disk + per-GPU info (RTX 3090) | `advanced/multi-gpu.md` |
| `miStudio_SAE_Panel-DownloadAndBrowse_*.jpg` | Inline HF download form + 19-SAE list | archive (SAE page already illustrated) |
| `miStudio_Extraction_Panel-Browse_*.jpg` | Extraction job browser | archive (extraction page already illustrated) |

## How to refresh

Requires the `playwright-skill` plugin installed and set up (`npm run setup`).
Use the **LAN host** `k8s-mistudio.hitsai.local` — the public `mistudio.hitsai.net`
name sits behind a Cloudflare bot challenge that blocks headless browsers. This
host has no X server, so captures must run **headless**.

```bash
# 1. capture all panels to /tmp/caps (script: see 0xcc/caps/capture-panels.js)
SKILL=~/.claude/plugins/cache/playwright-skill/*/skills/playwright-skill
cd $SKILL && node run.js /path/to/capture-panels.js

# 2. copy into this folder with today's date
DATE=$(date +%Y%m%d)
for f in /tmp/caps/*.png; do
  cp "$f" "0xcc/caps/$(basename "$f" .png)_${DATE}.png"
done
```

The capture script lives alongside this README as `capture-panels.js`.
