# Product Engineer Agent

**Role:** Deep context development and business alignment
**Focus:** Requirements, user stories, business logic, product vision

## Current Analysis

### Context Completeness
**Last Assessed:** 2025-11-06
**Status:** EXCELLENT - Core features + UI optimization complete

**Requirements Coverage:**
- [x] User stories defined with acceptance criteria
- [x] Business rules documented and understood
- [x] Edge cases identified and planned (retry logic, error handling)
- [x] Integration requirements mapped
- [x] Success metrics defined (progress tracking, real-time updates)

**Context Gaps Identified:**
1. Resource utilization not correlated with active jobs (users can't see which job is using which GPU)
2. No historical progress visualization for performance debugging
3. Missing comparative analysis across training runs

### Product Health Check
- **User Story Coverage:** 90% complete
- **Business Logic Clarity:** CLEAR
- **Integration Readiness:** READY

### Recent Accomplishments (2025-11-06)
✅ **UI Compression & Enhancement Work Completed:**
- Screen space utilization improved 10% (80% → 90% width)
- UI density increased 20-30% through systematic compression
- 5 user requests fully implemented (swirl clock, filter counts, favorites, etc.)
- 7 additional enhancements delivered beyond original scope
- Excellent code quality and consistency maintained

### Current Recommendations
**High Priority:**
1. Gather user feedback on new compact UI (do users find it comfortable?)
2. Integrate system resource monitoring with job progress displays (show "Training X using GPU 0 - 85%")
3. Add unified operations dashboard showing all active jobs + resource usage

**Medium Priority:**
1. Consider user preference toggle for "Compact" vs "Comfortable" UI density
2. Add progress history visualization for debugging slow training runs
3. Implement training run comparison features for optimization

**Low Priority:**
1. Add resource usage predictions based on job parameters
2. Implement job queuing recommendations based on available resources

### Celery/Monitor Review Findings (2026-07-10)
See `.claude/context/sessions/review_celery_monitor_operations_2026-07-10.md`.
- Monitor page's "Active Operations" shows only ~3 of 10 job types (task_queue rows created only on failure; only labeling/pushes unioned in). Running trainings/extractions/steering invisible — contradicts the "unified operations dashboard" roadmap item.
- "Failed Operations" misses failed labeling jobs, pushes, trainings, extractions.
- Recommendation adopted: federate the endpoints over the real job tables (read-only) rather than dual-writing task_queue.

### Session Context
**Working On:** User feedback collection for UI compression work
**Last Completed:** UI compression and enhancement work (12 tasks, 5 files, 100% success rate)
**Next Steps:**
1. Gather user feedback on new compact UI density
2. Design unified operations dashboard UX
3. Plan resource-job correlation UI patterns
4. Document UI compression patterns in design system

**Blockers:**
- None - current implementation meets core requirements excellently

**Notes:**
Recent comprehensive architecture review shows excellent user-facing progress tracking. UI compression work delivered 10% more screen space with 20-30% tighter spacing while maintaining excellent readability and code quality. All 5 user requests plus 7 additional enhancements implemented successfully. Main opportunity is better integration between resource monitoring and job execution visibility.


---

## Usage with Claude Code

### Loading This Agent Context
```markdown
@.claude/context/agents/product_engineer.md
@CLAUDE.md
```

### Integration with Commands
- Use with `/analyze product [scope]`
- Update after `/feature` command usage
- Reference in `/review` workflows
- Include in `/collaborate` sessions

### Agent Activation Phrase
"Load the Product Engineer agent context for requirements analysis"

---
*Update this context when working on product/requirements analysis*


## Findings — J-Lens enhancement arc (2026-08-10)

Full record: `.claude/context/sessions/review_jlens_enhancements_2026-08-10.md`

**8 findings, all claim-versus-behaviour.** The pattern: the code was often
right and the thing the user or agent READS about it was wrong.

- A **rung-1 badge** over a rung-2 measurement, with a module docstring still
  describing the lens-space displacement deleted in the rewrite.
- **`artifact_id` documented as "act in JACOBIAN lens space"** when the
  perturbation is always in the residual stream — it is provenance, not a
  different experiment. An agent believing it files a rung-2 record against a
  lens that played no part in the finding.
- **The user's stated goal was unreachable from the product.** "Experiment with
  them to determine how they can be used in causal influence" cannot produce a
  positive result when every UI path sends one prompt and separation needs four.
  Then the fix's own remedy — "add prompts" — pointed at a card with no such
  control.
- **The swap partner**, half the experiment and the token whose rank is scored,
  was picked silently from pin order and never shown, while being written into
  the exported recipe.

**Priority for the next review:**

1. **Walk the user's stated goal end to end and check it is reachable.** Not
   "does the feature exist" — can the user get the outcome they described?
2. **Every remedy string names a control that exists.** "Use X to do Y" is a
   testable claim.
3. **BR-002 remains clean** — no band constant anywhere in the new frontend.
   Keep checking; it is the rule most likely to be re-introduced by a port.
4. **What travels.** `interventions.json` goes to HuggingFace and into miLLM. A
   field that is wrong there is wrong in production for someone else.
