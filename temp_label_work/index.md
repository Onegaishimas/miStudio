# SAE Feature Label Index
**SAE:** SAE from LFM2.5-1.2B-Instruct (L12-residual)
**SAE ID:** sae_eb8374929894
**Layer:** 12 (residual stream)
**Total features:** 32,768

---

## Labeled Features

| Feature | Proposed Label | Confidence | Core Pattern |
|---------|---------------|------------|--------------|
| [#09178](feature_09178_label.md) | Spelled-out number words and Roman numerals | High | Tokens that are (or form part of) a number word: cardinals (thirty, twenty), ordinals and their BPE suffixes (-teenth, nineteenth), and Roman numerals (III, VIII). Fires across time durations, century designations, edition numbers, constitutional amendments, section labels. |
| [#10023](feature_10023_label.md) | Preposition "of" following "University" in institution names | High | The token `[Ġof]` in "University of [Name]". Fires at near-constant strength (7.11–6.25) across 100 examples, all in academic research attribution contexts. Maximally monosemantic. |

---

## Notes

**On #09178 (number words):**
- Particularly strong on ordinal BPE fragments: `-teenth`, `-eteenth` (Fourteenth, nineteenth) reach max activation
- Roman numerals activate slightly lower than spelled-out words but are clearly within the core pattern
- Low-activation tail includes probable noise: first-person pronoun "I" may exhibit weak polysemanticity with Roman numeral I
- The word `[Ġten]` inside "tenures" at Example 74 is a notable false positive worth flagging

**On #10023 ("University of"):**
- Existing user label "university_of" was directionally correct
- The interpretability score (0.2592) appears to be a false low — the automated system likely penalized it for activating on a common function word ("of") without recognizing the specificity of the context
- Open question: does the feature generalize to "College of", "School of", "Institute of"?

---

## Files in this directory

| File | Purpose |
|------|---------|
| `fetch_feature.py` | DB query script — pulls activations for any feature in this SAE |
| `feature_09178.txt` | Raw activation examples for feature 9178 |
| `feature_10023.txt` | Raw activation examples for feature 10023 |
| `feature_09178_label.md` | Full label report for feature 9178 |
| `feature_10023_label.md` | Full label report for feature 10023 |
| `index.md` | This file |

## How to add another feature

```bash
# Pull examples (tunnel must be open):
python fetch_feature.py <neuron_index> --sae-id sae_eb8374929894

# Then run labeling workflow manually on the generated .txt file
# Write output to feature_XXXXX_label.md
# Add a row to the table above
```

**DB tunnel (if not already open):**
```bash
~/.local/bin/sshpass -p "pass" ssh -o StrictHostKeyChecking=no -fN \
  -L 5435:10.152.183.147:5432 sean@192.168.244.61
```

**Python interpreter:** `/home/x-sean/app/miStudio/backend/venv/bin/python3`
