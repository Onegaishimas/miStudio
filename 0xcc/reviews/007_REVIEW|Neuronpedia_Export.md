# Feature Review: 007 — Neuronpedia Export & Push

**Reviewed:** 2026-07-11
**Reviewer:** deep code review (doc↔code accuracy + bug/correctness + quality/tech-debt)
**Feature docs:** [FPRD v2.0](../prds/007_FPRD|Neuronpedia_Export.md) (2026-03-21), FTDD, FTID
**Verdict:** The **export half is the best-documented data-model/endpoint match in the review** (export job model, routes, logit-lens algorithm all accurate). The **push half (this session's Jan-2026 work) is under-documented and uses raw SQL with no ORM model (P2)** — a real consistency gap that already surfaced in the task_queue federation. No P0/P1. Zero test coverage.

---

## 1. Scope

**Backend read:** `models/neuronpedia_export.py` (122), `api/v1/endpoints/neuronpedia.py` (586, full route inventory), `services/logit_lens_service.py` (538, algorithm), `workers/neuronpedia_push_tasks.py` (294, full — this session's code), `services/neuronpedia_export_service.py` (707, surveyed), `services/neuronpedia_local_service.py` (1043, surveyed), `workers/neuronpedia_tasks.py` (296, surveyed). Confirmed `feature_dashboard.py` model exists.
**Tests present:** **NONE** (no `test_neuronpedia*`, `test_logit*`, or `test_export*`).

---

## 2. Doc ↔ Code Accuracy

**Best-in-review for the export half; under-documented for the push half.**

### 2.1 Key files — 1 wrong, push files accurate

| PRD claims (§8) | Reality |
|---|---|
| `workers/export_tasks.py` | **Does not exist.** The export task is in `workers/neuronpedia_tasks.py`. |
| (push files in §1.4) | `neuronpedia_local_service.py`, `neuronpedia_push_tasks.py`, `neuronpedia.py` — all **correct** ✅. |

### 2.2 Data model — export accurate, push is model-less

| PRD | Reality |
|---|---|
| `NeuronpediaExportJob` (§6.1) | **Accurate** ✅ — id UUID, sae_id, config, status enum (pending/computing/packaging/completed/failed), progress, current_stage, output_path, file_size_bytes, feature_count. Adds `source_type`, `error_details`. Best data-model match in the review. |
| `FeatureDashboardData` (§6.2) | Exists as `feature_dashboard.py` ✅. |
| **push job table** | **Undocumented + no ORM model.** `neuronpedia_pushes` is a **raw-SQL-only table** — `INSERT INTO neuronpedia_pushes ...` via `text()` in the endpoint ([neuronpedia.py:433](../../backend/src/api/v1/endpoints/neuronpedia.py#L433)) and `UPDATE neuronpedia_pushes ...` via `text()` in the task ([neuronpedia_push_tasks.py:120](../../backend/src/workers/neuronpedia_push_tasks.py#L120)). No SQLAlchemy model, not in the PRD data model. See F1. |

### 2.3 API endpoints — export accurate, push surface undocumented

| PRD (§5) | Reality |
|---|---|
| `/export`, `/export/{id}`, `/export/{id}/download`, `/export/{id}/cancel`, `/exports` | **All accurate** ✅ (though cancel is `POST /export/{id}/cancel` and there's also `DELETE /export/{id}`). |
| (not in PRD §5) | Push + dashboard surface: `POST /push-local`, `GET /push-local/{id}`, `GET /push-local` (list), `POST /compute-dashboard-data`, `GET /local-status`. Documented narratively in §1.4 but absent from the endpoint table. |

### 2.4 WebSocket channel — naming mismatch

PRD §1.4 says push channel `neuronpedia-push/{job_id}`; the endpoint docstring says `neuronpedia/push/{push_job_id}` ([neuronpedia.py:388](../../backend/src/api/v1/endpoints/neuronpedia.py#L388)). Verify which the emitter actually uses and align docs + code (the frontend must match exactly or progress silently fails — cf. 008's monitoring channel issue this session).

### 2.5 Logit-lens algorithm — accurate

§9.1 pseudocode (`W_dec @ W_U.T`, top ±k) matches `logit_lens_service.compute_logit_lens_for_sae` ([logit_lens_service.py:8-10,177](../../backend/src/services/logit_lens_service.py#L8)), including the `W_dec` transpose handling for the `[d_in, d_sae]` → `[d_sae, d_in]` convention. Research-critical code, correct.

**Recommendation:** FPRD 007 needs the push half promoted from §1.4 narrative into the §5 endpoint table and §6 data model — and the `neuronpedia_pushes` table needs an ORM model (F1) so it's schema-tracked and documentable like everything else.

---

## 3. Findings (severity-ranked)

### P2 — Consistency / maintainability

**F1. `neuronpedia_pushes` has no ORM model — raw SQL everywhere.**
The push-job table is created and mutated exclusively via `text()` raw SQL across three files (endpoint INSERT, task UPDATE, and the task_queue federation SELECT I added this session). Consequences:
  - No SQLAlchemy model → not visible to `Base.metadata`, not covered by the schema validator, not documentable in the standard data-model section, and the column set is only discoverable by reading SQL strings.
  - The task_queue `/active` federation (built this session) hand-writes a `SELECT ... FROM neuronpedia_pushes` — if a column is renamed in the INSERT, that federation breaks silently.
  - Every other entity in the app has a model; this is the lone exception.
**Fix:** add a `NeuronpediaPushJob` SQLAlchemy model mirroring the table, migrate the raw SQL to ORM, and document it in FPRD 007 §6. Low risk, high consistency payoff.

**F2. WS push channel naming inconsistency (F2.4 above).** Doc says `neuronpedia-push/{id}`, code docstring says `neuronpedia/push/{id}`. Given this session already found a monitoring-channel mismatch silently break the Monitor page, verify the emitter/frontend agree and fix the doc. **P2, not P3, because a channel mismatch = silent progress failure** (the exact class of bug fixed in the Celery review F1).

### P3 — Quality / tech-debt

**F3. Zero test coverage** — no export, push, or logit-lens tests. The logit-lens computation is research-critical (wrong W_U transpose = garbage exported to the community) and completely unverified. The push flow (FK ordering Model-before-Source, dashboard computation) is likewise untested. Priority: a logit-lens correctness unit test (known SAE → expected top tokens) and a push happy-path integration test.

**F4. Export cancel is `POST /export/{id}/cancel` but delete is `DELETE /export/{id}`** — mixed with the push side using different verbs; minor REST inconsistency (recurring across features, see cross-feature synthesis).

**F5. `neuronpedia_export_service.py` (707) + `neuronpedia_local_service.py` (1043)** duplicate some dashboard-data computation (logit lens, histograms) between the export-to-ZIP path and the push path. Worth checking whether both call the shared `logit_lens_service` or reimplement — if the latter, a divergence risk for exported vs pushed data.

---

## 4. Test Coverage Notes

Zero coverage (tied with steering for worst). Highest-value additions:
1. **Logit-lens correctness** — a deterministic SAE + model → assert expected promoted/suppressed tokens. This protects research validity of every export/push.
2. **Push happy-path** — INSERT → task → status transitions → completion (would also exercise the raw-SQL table that F1 flags).
3. **Export format compatibility** — validate the ZIP structure matches SAELens/Neuronpedia expectations (the PRD checklist claims this is tested, but no test file exists).

---

## 5. Summary — if you fix three things

1. **F1** — give `neuronpedia_pushes` a real ORM model (removes the lone raw-SQL table; protects the task_queue federation; makes it documentable).
2. **F2** — verify + fix the push WS channel naming (silent-progress-failure class).
3. **F3** — add a logit-lens correctness test (research-critical, currently unverified).

And **promote the push half of FPRD 007 from §1.4 narrative into the §5/§6 tables** so the endpoint + data model reflect reality.

**The good:** the export subsystem — job model, routes, package structure, and the logit-lens algorithm — is the most doc-accurate feature in the review. The push subsystem works but was bolted on with raw SQL and light documentation; bringing it up to the export half's standard is the main task here.
