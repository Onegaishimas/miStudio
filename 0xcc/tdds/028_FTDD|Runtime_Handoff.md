# TDD: Runtime Handoff

**Document ID:** 028_FTDD|Runtime_Handoff · **Version:** 1.0

## 1. A watchlist is a detector definition, not a list of words

```
concept        -> direction + threshold
scoring_definition -> HOW the score was computed
artifact_ref   -> WHICH lens the directions live in
```

All three or none. A direction without its artifact is meaningless (lens coordinates are
artifact-specific); a threshold without its scoring definition is a different detector at the far
end; and a list without either is a list of words.

## 2. The evaluation-awareness score is a DIFFERENCE

```
score = mean_layers( mean(logprob, eval_tokens) - mean(logprob, control_nouns) )
```

The subtraction is not a refinement, it is the measurement. Common tokens have high log-probability
everywhere, so the raw mean makes every prompt look evaluation-aware — a detector that fires
constantly and is therefore never consulted.

The control set is **unrelated concrete nouns**, and it is recorded so the score can be reproduced.

## 3. The cost envelope refuses to guess

Every operation class has an estimator. A class with no estimator raises rather than returning a
default, because a defaulted-cheap estimate is worse than none: it invites the run it should have
warned about.

Estimates are ORDER-OF-MAGNITUDE and labelled as such. Presenting a false precision would invite
planning against a number that was never measured.

## 4. Risks

| risk | mitigation |
|---|---|
| A watchlist ships without its scoring definition | required field; refused at construction |
| The awareness score is reported raw | the control subtraction is inside the function, not a caller's job |
| An unestimated class defaults to cheap | raises instead |
| An estimate is read as a measurement | labelled order-of-magnitude, asserted |
| MCP parity drifts as features land | the reachability harness enumerates tools |
