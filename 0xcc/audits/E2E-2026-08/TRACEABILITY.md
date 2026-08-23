# TRACEABILITY — doc chain ↔ code ↔ test

**Produced:** P11, 2026-08-23 · Companion to `FINDINGS.md`

The framework's documented join key is each FTASKS file's `## Relevant Files`
section (`0xcc/instruct/008_process-task-list.md`). This matrix reports how well
that key actually joins.

## 1. Section presence

| | Count |
|---|---:|
| FTASKS files | 28 |
| With a `Relevant Files` section | **23** |
| Without | **5** — `024_`…`028_` (Dictionary_Annotation, Intervention_Engine, Claims_Discipline, Contracts_And_Conformance, Runtime_Handoff) |
| Ad-hoc task files with the section | **0 of 6** |

Two heading spellings coexist: `## Relevant Files Summary` (001–011) and
`## Relevant Files` (012–023).

## 2. Do the listed paths exist?

273 distinct paths extracted from the 23 sections and tested against disk.

| Outcome | Count | Assessment |
|---|---:|---|
| Resolve as written | 223 | ✅ |
| Relative-path style (`panels/X.tsx` → resolves under `frontend/src/`) | 21 | cosmetic |
| Explicitly labelled "To Create" in 009, never built (PPRD row 10 agrees) | 7 | consistent |
| **Genuinely dead** | **22** | **broken join** |

Dead paths by file — worst first:

| FTASKS | dead |
|---|---:|
| 003 SAE_Training · 004 Feature_Discovery · 008 System_Monitoring | 3 each |
| 002 Model_Management · 019 Circuit_Calibration | 2 each |
| 001, 005, 006, 010, 012, 014, 018, 021, 022 | 1 each |
| **007, 011, 013, 015, 016, 017, 020, 023** | **0** |

## 3. The dead paths were never written, not renamed

`git log --all --diff-filter=A` on 15 of the 22: **15 of 15 have zero add-commits in
the entire repository history.** Not renames, not deletions — they never existed.

`003_FTASKS:248-256` is the clearest case: four `[x]` boxes under *"Create
TrainingForm Component"* against a `TrainingForm.tsx` that has never existed. The
capability is real (inline SVG charts at `TrainingCard.tsx:989-1059`), so it is a
documentation defect — **except** Task 7.7's `- [x] Zoom and pan`, which has no
implementation anywhere: no recharts `Brush`, no wheel handler, nothing.

**Conclusion:** for features 001–008 the sections appear to have been authored from
the *design documents*, not from the implementation. The join key is least reliable
exactly where the docs claim 100% completion.

Where the practice is followed it works — 013, 020 and 023 have zero dead paths, and
013's 13 paths were all touched inside its own commit window.

## 4. PPRD §2.1 status vs reality

Applying the documented `+1` offset (PPRD row R ↔ file R−1):

| PPRD rows | §2.1 says | Reality |
|---|---|---|
| 16–20 | **all "Planned"** | CLAUDE.md records all five **CLOSED**; code present |
| 22, 23, 24 | Planned | shipped (FTASKS 97% / 0% / 100%, code present) |
| 25–29 | Planned | code present, FTASKS 68–100% |
| 21 | "Implemented" | shipped — but the only row using that word instead of "✅ Complete" |

**PPRD §2.1 marks "Planned" every feature shipped since 2026-07-19 — 13 rows.**
MIS-E2E-011 recorded 6; that was an undercount, now corrected in the register.

## 5. Task-box census

| | Unchecked |
|---|---:|
| The 28 numbered FTASKS | 155 |
| Six ad-hoc task files (none has `Relevant Files`) | **193** |
| **Total** | **348** |

`IMPL_Celery_Steering_Migration.md` alone is 61 unchecked / 0 done, while PADR IDL-13
and the session log both describe that migration as shipped.

## 6. IDL conformance — the ones sampled

| IDL | Verdict |
|---|---|
| IDL-1, IDL-12 (WebSocket conventions) | ✗ channels and event names both differ from the code (MIS-E2E-158) |
| IDL-5 (Celery Beat for monitoring) | ✗ **inverted** — it is an asyncio loop in the API process (MIS-E2E-156) |
| IDL-11 (Celery resilience) | ✗ no DLQ, no backoff; the exemplar task overrides `acks_late` (MIS-E2E-159) |
| IDL-13 (dynamic layer discovery) | ~ substantive claim holds; one named consumer doesn't use it (MIS-E2E-164) |
| IDL-16 (schema validation tooling) | ✗ all three claims false (MIS-E2E-157) |
| IDL-26 (MCP architecture) | ~ architecture clean; category enumeration wrong (MIS-E2E-161) |
| IDL-38 (one steering core) | ✗ there are two (MIS-E2E-076) |
| IDL-46 (filesystem is the registry) | ~ "no DB table" holds; the mount wiring is absent (may be unbuilt) |
| **IDL-19** (OpenAI SDK) | ✅ exact — retries, reasoning-model detection, `max_completion_tokens`, backoff |
| **IDL-25** (PIN) | ✅ backend exact — PBKDF2 600k, random salt, `compare_digest`; only the manual's *scope* claim is wrong |
| **IDL-39** (checkpoint retention) | ✅ all eight points hold, including `dry_run` failing to its default |
| **IDL-20, 22, 23, 42** | ✅ clean |

## 7. Manual conformance

**Checked and clean:** all ~140 documented endpoint rows across
`manual/docs/reference/api/*.md` were diffed against every `@router.*` decorator —
**no documented method+path is missing from the backend**. Omissions run the other
way (`/task-queue/failed/dismiss-all`, the template `/favorite`, `/duplicate`,
`/clone`, `/set-default` routes, and the sync `/steering/compare|sweep` are
undocumented). SAE delete semantics, aqua-star protection and `multi-gpu.md`'s
"Planned (Not Yet Implemented)" table all match the code. **The Stop/Finalize and
checkpoint-retention pages under `manual/docs/` are accurate** — the 2026-07-26
incident was properly fixed there.

**Not clean:** MIS-E2E-149 (the same Stop sentence, uncorrected, in `docs/`),
150 (MCP anonymous), 151 (dataset cancel), 152 (k8s `sed`), 160 (PIN scope),
161 (MCP categories), 162 (README startup), 164 (four smaller).

## 8. The structural cause

Three of this phase's findings share one mechanism: **a correction is applied to the
document under review and not propagated.**

- The `system_monitor_tasks.py` deletion was fixed in `008_FPRD` and left standing in
  README, CLAUDE.md, the PPRD, the FTDD and the FTASKS.
- The "Stop saves final checkpoint" sentence was fixed in `manual/docs/` and left
  standing in `docs/miStudio_Manual.md`.
- The instruction-file renumbering was never reflected in CLAUDE.md at all.

A cross-document grep of the corrected phrase, run at fix time, catches all three.
That is the cheapest durable remediation this phase produces.
