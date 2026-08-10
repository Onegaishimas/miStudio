# Test Engineer Agent

**Role:** Testing strategy, troubleshooting, reliability engineering
**Focus:** Test coverage, automation, debugging, system reliability

## Current Analysis

### Testing Health Assessment
**Last Assessed:** 2025-10-22
**Testing Maturity:** BASIC-GOOD (Recent improvements, significant gaps remain)

**Coverage Analysis:**
- **Unit Tests:** 40% coverage (Target: 70%)
- **Integration Tests:** 20% coverage (Target: 50%)
- **End-to-End Tests:** 10% coverage (Target: 30%)
- **Performance Tests:** PLANNED (TrainingMetric query performance)
- **Security Tests:** PLANNED (WebSocket authentication, rate limiting)

**Test Quality Metrics:**
- **Test Reliability:** MEDIUM (improving with recent additions)
- **Test Speed:** FAST (unit tests run quickly)
- **Maintenance Burden:** LOW (good patterns make tests easier to maintain)

### Testing Strategy
**Current Focus:**
- [x] Unit test expansion for core logic (TrainingPanel, trainingsStore completed)
- [ ] Integration test implementation (WebSocket emission flows needed)
- [ ] End-to-end workflow testing (complete training flow needed)
- [ ] Performance benchmarking (TrainingMetric queries needed)
- [ ] Security vulnerability testing (WebSocket auth needed)

**Test Automation Status:**
- **CI/CD Integration:** PARTIAL (tests exist, CI pipeline status unknown)
- **Automated Test Runs:** YES (pytest, vitest)
- **Test Reporting:** BASIC (console output, coverage reports)

### Risk Assessment
**High-Risk Areas:**
1. **WebSocket Connection Reliability (P0):** Silent disconnects could show stale progress to users. Need reconnection tests, message ordering tests, state synchronization tests.

2. **Progress Monitor Thread Safety (P1):** Background thread in model_tasks.py could have race conditions. Need concurrent update tests, verify no missed/duplicate progress updates.

3. **Polling Fallback Activation (P1):** Unclear when polling vs WebSocket is active. Could cause duplicate API calls. Need integration tests for fallback scenarios.

4. **TrainingMetric Table Growth (P2):** Could reach millions of rows, slow queries. Need performance tests with 1000+ training runs, load testing.

**Testing Gaps:**
1. No WebSocket reliability tests (connection, disconnection, reconnection, state sync)
2. No unit tests for progress calculation formulas (3-phase extraction, training steps)
3. No integration tests for error recovery flows (OOM retry, batch_size reduction)
4. No tests for concurrent operations (multiple jobs updating simultaneously)
5. No performance tests for system monitor polling efficiency
6. No security tests for WebSocket authentication/authorization

### Troubleshooting & Debugging
**Current Issues:**
- **P0 (Critical):** No visibility into WebSocket connection state (users can't tell if updates are stale)
- **P1 (High):** Test coverage insufficient for production deployment (40% vs 70% target)
- **P2 (Medium):** No progress history UI for debugging slow training runs

**Debugging Tools Status:**
- [x] Logging comprehensive and structured (console.log in stores, backend logging)
- [x] Error tracking implemented (TaskQueue captures failures)
- [x] Performance monitoring active (system monitor dashboard)
- [x] Test environment stable (recent tests passing)

### Celery/Monitor Review Findings (2026-07-10)
See `.claude/context/sessions/review_celery_monitor_operations_2026-07-10.md`.
Highest-risk untested paths identified:
1. BackgroundMonitor → `/api/internal/ws/emit` (403 regression went unnoticed — needs integration test asserting 200 with token wiring)
2. task_queue lifecycle fail → retry → success (would have caught the retry-ghost bug)
3. Training pause-during-log-interval race (simulate status write between loop check and progress write)
4. Frontend staleness fallback (WS connected but silent → polling resumes; fake-timer test)

### Session Context
**Current Testing Focus:** Progress tracking & resource monitoring architecture
**Test Development Status:** Planning test expansion for WebSocket reliability and progress calculations
**Next Testing Actions:**
1. Write unit tests for all progress calculation functions (training, extraction, model download)
2. Write integration tests for WebSocket emission flows (training, extraction, dataset)
3. Write WebSocket reconnection tests (simulate network failure, verify state recovery)
4. Write concurrent operations tests (multiple jobs updating simultaneously)
5. Write performance tests for TrainingMetric queries (1000+ training runs) 

---

## Usage with Claude Code

### Loading This Agent Context
```markdown
@.claude/context/agents/test_engineer.md
@CLAUDE.md
@0xcc/adrs/000_PADR|[project-name].md
```

### Integration with Commands
- Primary agent for `/analyze testing [scope]`
- Essential for comprehensive `/review` workflows
- Update during test development and debugging sessions
- Include in reliability-focused `/collaborate` sessions

### Agent Activation Phrase
"Load the Test Engineer agent context for testing strategy and reliability analysis"

### Best Used For
- Test coverage analysis and planning
- Debugging and troubleshooting issues
- Performance and reliability assessments
- Risk assessment and mitigation planning

### Integration with Development Workflow
- Works with test commands defined in CLAUDE.md
- Coordinates with QA Engineer agent on quality aspects
- References testing standards from ADR
- Tracks testing metrics and reliability trends

### Hardcore Debugging Mode
When complex issues arise, this agent can work with the hardcore-debugger agent:
```markdown
@.claude/agents/hardcore-debugger.md
@.claude/context/agents/test_engineer.md
```

---
*Update this context when working on testing and quality assurance*


## Findings — J-Lens enhancement arc (2026-08-10)

Full record: `.claude/context/sessions/review_jlens_enhancements_2026-08-10.md`

**9 findings in round 2, all found by mutating rather than reading — plus round
3, which found that BOTH of round 2's headline fixes were themselves
unprotected.**

The single most valuable output remains: *break line X, run the suite, it stays
green.*

**Failure modes that actually occurred here:**

1. **A regression test inheriting the trap it was written to catch.** The
   unit-norm fix could be deleted with 63 tests green, because the fixture's
   `W_U` was `torch.eye(...)` — already unit-norm, so both behaviours are
   identical on it. When writing a regression test, first ask what fixture makes
   the two behaviours *differ*.
2. **Source-scrape guards that fail OPEN.** A regex allowing one decorator
   between `@celery_app.task` and `def` matched NOTHING for a two-decorator
   task, so it never entered the checked list. `".replace(" in
   inspect.getsource(...)` passes on any unrelated `.replace(` and fails on
   `shutil.move`. Read the live registry; kill the operation and assert what
   survived.
3. **Reachability that only the panel can prove.** The layer-range picker could
   be unmounted with 143/143 green: its own test renders it directly, and the
   store test calls `setState`. The same shape as the 16 unregistered MCP tools.
4. **Mocked values pinning a rendering, not a computation.** Frontend tests
   appeared to cover `separation_attainable`; they mock it.
5. **MUTATION CONTROL comments that are false as written.** Three this round.
   The comment is a claim and must be executed, not asserted.

**Discipline reinforced:**

- **Confirm the edit LANDED** before concluding a mutation survived. Several
  candidates here silently did not apply.
- **Never mutate while another agent is reading the tree.** One reviewer
  reported an in-flight mutation as a committed defect.
- **Record equivalent mutants** rather than chasing them: `>=` for `>` in
  `separation_attainable` survives because the two Wilson bounds are never
  exactly equal at any n. That is not a test gap.

**Priority for the next review:** mutate the round-3 fixes. Every round so far
has found that the previous round's fix was under-tested.
