---
name: ordinal_temporal_quantifier
category: numerical_linguistic_feature
description: This feature fires on tokens that function as suffixes or full words denoting ordinal numbers, temporal durations, or fractional parts within numerical contexts.
---

## Feature #09178 Label  *(LLM two-pass — review before committing)*

**Proposed label:** ordinal_temporal_quantifier
**Confidence:** high
**Examples summarized:** 20 of 100

### Pass 2 — Synthesis (raw LLM response)

```
```json
{
  "name": "ordinal_temporal_quantifier",
  "category": "numerical_linguistic_feature",
  "description": "This feature fires on tokens that function as suffixes or full words denoting ordinal numbers, temporal durations, or fractional parts within numerical contexts.",
  "confidence": "high"
}
```
```

### Pass 1 — Per-example summaries

| Ex | Act | Token | Summary |
|----|-----|-------|---------|
|   1 | 7.97 | `teenth` | In this context, "[teenth]" functions as the suffix completing the word "Fourteenth," specifically denoting the ordinal number of the amendment being discussed.His decision to amend their state constitution to outlaw racial preferences in college admissions somehow violates the Four[teenth] Amendment's Equal Protection Clause. As Dave Barry would say, I'm not making this up: The court voted 8 |
|   2 | 7.87 | `-five` | In this context, [-five] functions as a semantic placeholder or a linguistic marker within the phrase "forty[-five]," suggesting an implied or unstated continuation of a numerical or comparative concept related to age or size.ференциальноthought
Here's a thinking process to arrive at the suggested answer:

1.  **Analyze the Request:**
    * |
|   3 | 7.34 | `thirty` | In this context, "[thirty]" functions as a **temporal quantifier**, specifically specifying the duration of time elapsed before an event occurred. |
|   4 | 6.91 | `-first` | In this context, [-first] functions as a linguistic marker indicating the beginning of a sequence or enumeration, specifically preceding the phrase "episode of."Equally, [-first] is functioning as a linguistic marker indicating the beginning of a sequence or enumeration, specifically preceding the phrase "episode of. |
|   5 | 6.45 | `teenth` | The token "[teenth]" is functioning as the suffix completing the word "Fourteenth," specifically indicating the ordinal number of the amendment mentioned in the text.Sanity Check: The prompt asks for a description of the linguistic or semantic role of "[teenth]" in the context. The context is: "...Equal Protection Clause of the Four[teenth] Amendment of the United States Constitution." The role |
|   6 | 6.33 | `twenty` | In this context, "[twenty]" functions as a numerical quantifier specifying the denomination of the dollar bill mentioned in the preceding phrase.Equally, the token "[twenty]" is functioning as a numerical quantifier specifying the denomination of the dollar bill mentioned in the preceding phrase. |
|   7 | 6.30 | `fifty` | In this context, "[fifty]" functions as a temporal quantifier specifying the duration of the grim prospects for the average American.Equally valid answers:

*   It functions as a temporal quantifier specifying the duration of the grim prospects.
*   It specifies the time frame—fifty years—over which the grim outlook is projected. |
|   8 | 6.24 | `eteenth` | The token "[eteenth]" functions as a temporal marker, specifically indicating the midpoint of the nineteenth century, to delineate a period of limited scientific understanding before a major conceptual shift.ੁthought
Here's a thinking process to arrive at the desired output:

1.  **Analyze the Request:**
    *   **Input:** A passage of text with a specific token  |
|   9 | 6.19 | `thirty` | In this context, "[thirty]" functions as a temporal quantifier specifying the duration of the speaker's experience as a dog fancier. |
|  10 | 6.18 | `III` | In this context, "[III]" functions as a placeholder reference to a specific section within the document, serving as a formal citation or cross-reference point. |
|  11 | 6.00 | `twenty` | In this context, "[twenty]" functions as a **temporal quantifier**, specifying the duration of time elapsed before AOL supposedly "solved" the problem.っき |
|  12 | 5.89 | `nineteenth` | In this context, "[nineteenth]" functions as a **qualifier** specifying the time period of the predecessors mentioned, thereby describing the origin of the beer gardens. |
|  13 | 5.66 | `III` | In this context, "[III]" functions as a specific proper noun suffix identifying a particular installment or iteration of the "Street Fighter" franchise.Equally precise and concrete:

"[III]" acts as a numeral suffix specifying the third iteration of the "Street Fighter" series mentioned previously.Equally precise and concrete:

"[III]" functions as a direct identifier, specifying the |
|  14 | 5.47 | `twentieth` | In this context, "[twentieth]" functions as a temporal quantifier specifying the century to which the survival of the downtown area's older version relates.Equally good answers:

*   It specifies the century of the earthquake and war that spared the downtown area.
*   It acts as a temporal marker defining the century of the destruction/survival event. |
|  15 | 5.31 | `ten` | In this context, "[ten]" functions as a temporal quantifier, specifying a duration of ten years into the future from the present moment.แ |
|  16 | 5.20 | `seventy` | In this context, "[seventy]" functions as a numerical quantifier specifying the quantity of different names the narrator called out to the innkeeper.هَا</tr>
</table> |
|  17 | 5.18 | `Thirty` | In this context, "[Thirty]" functions as a temporal quantifier, specifically indicating a duration of thirty years, contrasting the present situation with a past state.Sanity Check: The passage is incomplete, but the context strongly suggests a comparison between the present and the past, making "Thirty years ago" a clear temporal marker. |
|  18 | 5.16 | `VIII` | In this context, "[VIII]" functions as a specific ordinal numeral suffix, identifying the eighth installment of the *Star Wars* film series.Your response is excellent. It is precise, concrete, and directly addresses the linguistic/semantic role of the token in the passage.thought
Here's a thinking process that leads to the suggested response:

1.  ** |
|  19 | 5.11 | `enth` | In this context, "[enth]" functions as a suffix completing the numerical fraction "nine-tenths," specifically representing the tenth part of a whole.แ |
|  20 | 4.91 | `teenth` | The token [teenth] functions as a suffix completing the ordinal number "fourteenth," specifying the temporal extent of the European Renaissance mentioned in the sentence. |

---

*Generated by label_feature_llm.py (two-pass) using gemma-4-E4B-it.*
*To commit: `python apply_label.py feature_09178_label.md --neuron 9178 --sae-id <sae_id> --commit`*
