# TID: Runtime Handoff

**Document ID:** 028_FTID|Runtime_Handoff · **Version:** 1.0

## Pitfalls
1. **A threshold without its scoring definition is a different detector.** Required field.
2. **The awareness score is the DIFFERENCE.** Put the subtraction inside the function; a caller who
   must remember to subtract will eventually not.
3. **Record the control set**, or the score cannot be reproduced.
4. **An unestimated operation class RAISES.** A defaulted-cheap estimate invites the run it should
   have warned about.
5. **Label estimates order-of-magnitude.** False precision invites planning against a number nobody
   measured.
6. **Directions are artifact-specific.** A watchlist must reference the artifact it was built
   against, or its coordinates mean nothing elsewhere.
7. **This increment emits only.** Runtime evaluation is miLLM's plane.

## Testing
- A watchlist without a scoring definition is refused.
- The awareness score of an eval-heavy prompt EXCEEDS a neutral one, and the raw mean does not
  distinguish them (the reason the control exists).
- Every operation class has an estimate; an unknown class raises.
- Estimates are labelled as order-of-magnitude.

**Mutation controls:** default the scoring definition; drop the control subtraction; default an
unknown class to cheap; drop the artifact reference.
