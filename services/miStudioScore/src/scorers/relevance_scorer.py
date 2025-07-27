# src/scorers/relevance_scorer.py
"""
Implementation of the Business-Driven Relevance Scorer.
"""
import logging
from typing import List, Dict, Any
from .base_scorer import BaseScorer

logger = logging.getLogger(__name__)

class RelevanceScorer(BaseScorer):
    """
    Scores features based on their correlation with user-defined keywords
    found in their top activating examples.
    """

    @property
    def name(self) -> str:
        return "relevance_scorer"

    def score(self, features: List[Dict[str, Any]], **kwargs) -> List[Dict[str, Any]]:
        """
        Calculates a relevance score for each feature.

        Args:
            features: The list of feature data.
            **kwargs: Must contain 'score_name', 'positive_keywords',
                      and 'negative_keywords'.

        Returns:
            The enriched list of features.
        """
        score_name = kwargs.get("name")
        positive_keywords = set(kwargs.get("positive_keywords", []))
        negative_keywords = set(kwargs.get("negative_keywords", []))

        if not score_name:
            raise ValueError("RelevanceScorer requires a 'name' parameter.")

        logger.info(f"Starting relevance scoring for '{score_name}'...")

        for feature in features:
            score = 0.0
            # 'top_activating_examples' is assumed to be a list of strings
            activating_texts = feature.get("top_activating_examples", [])
            if not activating_texts:
                feature[score_name] = 0.0
                continue

            all_text = " ".join(activating_texts).lower()
            
            # Simple scoring: +1 for each positive keyword, -1 for each negative
            pos_count = sum(1 for keyword in positive_keywords if keyword in all_text)
            neg_count = sum(1 for keyword in negative_keywords if keyword in all_text)

            # Normalize the score to be between -1 and 1
            total_keywords = len(positive_keywords) + len(negative_keywords)
            if total_keywords > 0:
                score = (pos_count - neg_count) / total_keywords
            
            feature[score_name] = round(score, 4)

        logger.info(f"Relevance scoring for '{score_name}' complete.")
        return features
