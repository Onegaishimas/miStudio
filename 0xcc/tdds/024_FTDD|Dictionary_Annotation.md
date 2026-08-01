# TDD: Dictionary Annotation & Weight-Space Readouts

**Document ID:** 024_FTDD|Dictionary_Annotation · **Version:** 1.0 · **Status:** Planned

## 1. The projection

A feature's decoder direction `w` is a d_model vector. Its lens readout is
`softmax(W_U · norm(J_ℓ w))` — the same transport the readout service already uses, applied to a
WEIGHT-SPACE vector rather than a residual.

That means `LensTransport` is reused unchanged. If annotation needed its own projection path, the
two could disagree, and a dictionary annotated by a different projection than the readout shows is
worse than no annotation.

## 2. Why two fields (BR-012)

```
geometric   = excess kurtosis of the projected vocabulary distribution
behavioural = motor vs workspace, from WHERE in the stack the direction reads strongly
```

Motor features are sharp — they commit to a token — so they score HIGH on kurtosis, exactly like
workspace features. Classifying on kurtosis alone therefore labels every motor feature a workspace
feature, and the error is invisible because the number is real.

The behavioural field separates them by depth profile: a motor direction reads strongly only near
the output end; a workspace direction reads across the middle.

**Without a band report there is no principled "middle".** So the behavioural field is ABSENT when
no band report exists for the model, rather than guessed. Same rule as everywhere else in this arc:
no band report, no band-dependent claim.

## 3. Disagreement (BR-013)

```
disagreement = the auto-label's tokens and the lens readout's top-k share
               no meaningful overlap
```

Stored as a queryable field so the queue is a filter over the existing feature list, not a new
screen. `disagreement_score` is sortable; `has_disagreement` is filterable.

Deliberately NOT auto-resolved. The lens is rung 0; it does not overrule a human or LLM label. It
raises a question.

## 4. The distributional check (BR-014)

A shape check on the OUTPUT of an annotation sweep:

- fraction J-aligned once motor features are excluded should be MODEST, not most;
- non-aligned features should be dominated by low-level syntactic/bookkeeping roles.

This catches the class of bug where a threshold is mis-scaled and everything lands on one side. It
is not a correctness proof and is reported as a shape observation, not as validation of the lens.

## 5. Weight-space readouts (BR-015)

| direction | why separately |
|---|---|
| SAE decoder | what the feature pushes toward |
| transcoder ENCODER | what it reads |
| transcoder DECODER | what it writes |
| attention Q/K/V/O | what the head moves |

Encoder and decoder combined into one readout hides the transformation — which is the only reason to
look at a transcoder at all.

## 6. Risks

| risk | mitigation |
|---|---|
| Motor features labelled workspace | two independent fields; a test with a motor-shaped fixture |
| Behavioural field guessed without bands | absent without a band report, asserted |
| Annotation projection drifts from the readout's | both use `LensTransport`; asserted identical on a fixture |
| Disagreement flagged but unusable | filterable AND sortable, asserted |
| A mis-scaled threshold labels everything workspace | the distributional check fails on that fixture |
