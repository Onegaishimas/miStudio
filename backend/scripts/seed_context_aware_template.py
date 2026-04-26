"""
Seed script: Context-Aware Labeling template.

This template shifts focus from the prime token to the full semantic context
of each activation example, producing labels that describe WHAT IS HAPPENING
across examples rather than WHAT TOKEN fired.

Run with:
  DATABASE_URL=... python scripts/seed_context_aware_template.py
"""

import asyncio
import sys
from pathlib import Path

backend_path = Path(__file__).parent.parent
sys.path.insert(0, str(backend_path))

from sqlalchemy import select
from src.core.database import AsyncSessionLocal
from src.models.labeling_prompt_template import LabelingPromptTemplate

TEMPLATE_ID = "lpt_context_aware_v1"

SYSTEM_MESSAGE = """\
You are an expert in mechanistic interpretability analyzing sparse autoencoder (SAE) features from language models.

Your task is to identify the semantic or functional PATTERN that a feature has learned — not to name the token it fires on.

Key principle: A feature's identity is the PATTERN OF CONTEXTS in which it activates, not the surface form of the highlighted token. Two features that both fire on the word "the" can encode completely different things (one might track "definite reference to a specific person", another "sentence-initial topic introduction"). The token is just the trigger location; the meaning lives in the surrounding text.

Your analysis process:
1. Read each example as a complete passage and understand what it is saying
2. Note what meaning, situation, grammatical role, or discourse function is present in THAT passage
3. Find what is SHARED across all examples at the semantic or functional level
4. Produce a label that names that shared pattern — even if the prime token varies across examples\
"""

USER_PROMPT_TEMPLATE = """\
Below are activation examples for SAE feature {feature_id}. In each example, the token with the highest activation is marked with << >>.

{examples_block}

---

Study the examples above. For each one, consider: what is semantically happening in the full passage? What meaning, grammatical role, situation, or discourse function is present?

Now identify the single unifying pattern that appears across ALL (or nearly all) examples. Ask yourself:
- If someone described all these passages to you without showing the marked token, what concept would come up every time?
- Is this about a syntactic construction, a semantic domain, a discourse function, a type of entity or event?
- Could examples with DIFFERENT marked tokens still share this feature? (If so, the feature is about context, not token.)

Respond with ONLY this JSON — no explanation outside it:
{{
  "category": "syntactic|semantic|positional|discourse|entity|mixed",
  "specific": "snake_case_label_max_5_words",
  "description": "One precise sentence describing the shared contextual pattern, grounded in what the passages have in common."
}}

Rules:
- specific: snake_case, 2–5 words, names the PATTERN not the token (e.g. "definite_known_entity_reference", "modal_epistemic_uncertainty", "list_enumeration_continuation")
- description: start with "This feature activates in contexts where..." and complete with the semantic pattern
- Do NOT simply name the prime token — if the label could be replaced by "fires on <token>", it is not good enough
- If examples cluster into two distinct patterns, name the dominant one and note the other in description\
"""


async def seed():
    print("Seeding Context-Aware Labeling template...")

    async with AsyncSessionLocal() as db:
        existing = await db.get(LabelingPromptTemplate, TEMPLATE_ID)
        if existing:
            print(f"Template '{TEMPLATE_ID}' already exists — updating prompts.")
            existing.system_message = SYSTEM_MESSAGE
            existing.user_prompt_template = USER_PROMPT_TEMPLATE
            await db.commit()
            print("Updated.")
            return

        template = LabelingPromptTemplate(
            id=TEMPLATE_ID,
            name="Context-Aware Labeling (Semantic Pattern)",
            description=(
                "Focuses on the full semantic context of each example rather than the prime token. "
                "Produces labels that describe WHAT IS HAPPENING across examples — the shared "
                "meaning, function, or discourse role — rather than naming the surface token. "
                "Recommended for OpenAI GPT-4o / GPT-5 class models."
            ),
            system_message=SYSTEM_MESSAGE,
            user_prompt_template=USER_PROMPT_TEMPLATE,
            template_type="mistudio_context",  # uses {examples_block} with full context windows
            temperature=0.2,
            max_tokens=200,
            top_p=0.95,
            prime_token_marker="<< >>",
            include_prefix=True,
            include_suffix=True,
            include_negative_examples=True,
            num_negative_examples=3,
            include_logit_effects=False,
            include_nlp_analysis=False,
            is_default=False,
            is_system=True,  # visible to all users, can't be deleted
            created_by=None,
        )
        db.add(template)
        await db.commit()
        print(f"Created template: '{template.name}' (id={template.id})")
        print("Select it in Labeling → Templates when starting a bulk job.")


if __name__ == "__main__":
    asyncio.run(seed())
