# HuggingFace Progress API Gaps & miStudio Workarounds

## Summary

HuggingFace libraries provide no programmatic progress callback API for their long-running
operations. miStudio requires real-time WebSocket progress for every operation it runs.
This document records the three workarounds we built, the operations they cover, and
the improvement requests we would make to HuggingFace.

---

## Workarounds Built in miStudio

### 1. `DownloadProgressMonitor` — Model Downloads
**File:** `backend/src/workers/model_tasks.py`

`snapshot_download()` from `huggingface_hub` emits tqdm bars to stderr only; no
programmatic hook exists. `DownloadProgressMonitor` runs a background thread that
polls the HuggingFace cache directory size at regular intervals and infers download
percentage from filesystem growth.

**Limitations:**
- Percentage is an approximation, not an accurate byte count
- Progress is capped at 90% during download and bumped to 95% after `snapshot_download()`
  returns, because there is no signal for when the final file has fully flushed
- Will misreport if another download is concurrently writing to the same cache directory

---

### 2. `tqdm_websocket_bridge.py` — Dataset Download & Tokenization
**File:** `backend/src/workers/tqdm_websocket_bridge.py`

`load_dataset()` and `dataset.map()` use tqdm internally but expose no callback API.
`TqdmWebSocketCallback` subclasses tqdm and is monkey-patched into the library at two
locations:

- `datasets.utils.tqdm.tqdm` — the canonical reference
- `datasets.arrow_dataset.hf_tqdm` — a cached module-level alias that `arrow_dataset.py`
  captures at import time; patching only the first location silently misses all
  multiprocessing paths

Each tqdm tick is intercepted, throttled (fires only if ≥1% delta or ≥0.5 s has
elapsed), written to the database, and emitted via WebSocket.

**Limitations:**
- Relies on undocumented internal module structure; may break across `datasets` versions
- Requires patching two separate references; future refactors inside `datasets` could
  introduce a third cached reference without warning
- Multiprocessing workers each have their own tqdm instance; reported percentage can
  be non-monotonic when workers finish out of order

---

### 3. Hardcoded Checkpoint Emissions — Dataset Post-Processing Phases
**File:** `backend/src/workers/dataset_tasks.py`

After `load_dataset()` completes, HuggingFace enters a silent save-to-Arrow-cache phase
that can take significant time. tqdm goes quiet during this phase. `dataset_tasks.py`
manually emits fixed percentages at known transition points:

| Hardcoded value | Meaning |
|---|---|
| 10 % | Download starting |
| 70 % | Download tqdm complete; saving to cache begins |
| 90 % | Cache save complete |
| 100 % | Dataset fully ready |

**Limitations:**
- These values are guesses about where HuggingFace is internally; they are not derived
  from actual byte counts
- The save phase duration is unpredictable (proportional to dataset size); 70 % → 90 %
  can stall for minutes on large datasets

---

## Operations That Do NOT Require Workarounds

These operations are driven entirely by miStudio's own code; progress is emitted
naturally at each loop iteration:

| Operation | Why it's native |
|---|---|
| SAE training | miStudio's training loop; emits at each step |
| Activation extraction | miStudio iterates the dataset per batch |
| Feature labeling | miStudio calls the LLM API per sample |
| Neuronpedia push | miStudio controls each upload call |
| SAE upload / download (Hub) | miStudio calls `upload_file()` / `hf_hub_download()` per file |

---

## Recommended Improvements to HuggingFace

These are the improvement requests that would eliminate all three workarounds above.

### Request 1 — `progress_callback` on `load_dataset()` and `.map()`
A documented callback parameter, e.g.:

```python
load_dataset("...", progress_callback=fn)
dataset.map(fn, progress_callback=fn)
# callback signature: fn(current: int, total: int, description: str) -> None
```

This would eliminate both `tqdm_websocket_bridge.py` and the hardcoded checkpoint
emissions, replacing ~200 lines of fragile monkey-patching with a single callback
registration.

### Request 2 — Per-file progress events from `snapshot_download()`
A `progress_callback` or async iterator variant, e.g.:

```python
snapshot_download("...", progress_callback=fn)
# callback signature: fn(filename: str, bytes_done: int, bytes_total: int) -> None
```

This would eliminate `DownloadProgressMonitor` entirely. Even a simple total-bytes
variant would be sufficient.

### Request 3 — Central progress handler registry
A single entry point that all HuggingFace libraries route progress through:

```python
huggingface_hub.utils.set_progress_handler(fn)
```

Currently `transformers`, `datasets`, and `huggingface_hub` each import tqdm under
different aliases. A monkey-patch applied to one does not reach the others. A central
registry analogous to `logging.setLevel()` would make the three requests above
unnecessary as a stopgap.

### Request 4 — Aggregated multiprocessing progress in `.map()`
When `num_proc > 1`, each subprocess has its own tqdm instance. Progress events do not
aggregate back to the parent process. A `total_progress_callback` that fires in the
parent process with a monotonically increasing row count would fix the jumpy reporting
we see in multi-process tokenization.

---

*Document created: 2026-04-12*
*Relevant files: `backend/src/workers/tqdm_websocket_bridge.py`, `backend/src/workers/model_tasks.py`, `backend/src/workers/dataset_tasks.py`*
