"""
Feature Prioritizer for miStudioExplain Service

Analyzes input data to identify the most salient features for explanation.
"""

import logging
import re
from collections import Counter
from math import log
from typing import List, Dict, Any, Tuple

from .input_manager import ExplanationRequest, FindResultInput, RawTextInput

logger = logging.getLogger(__name__)


class FeaturePrioritizer:
    """
    Identifies and ranks the most important features from the input data
    to focus the explanation generation process.
    """

    def __init__(self):
        logger.info("🔧 FeaturePrioritizer initialized.")

    def prioritize_features(self, request: ExplanationRequest) -> List[str]:
        """
        Main method to prioritize features based on the input data type.

        Args:
            request: The validated ExplanationRequest object.

        Returns:
            A list of prioritized features (strings).
        """
        logger.info(f"Prioritizing features for request {request.request_id}...")
        if isinstance(request.input_data, RawTextInput):
            corpus = request.input_data.text_corpus
            return self._prioritize_from_text(corpus)
        elif isinstance(request.input_data, FindResultInput):
            return self._prioritize_from_find_result(request.input_data)
        else:
            logger.warning(f"Unsupported data type for prioritization: {type(request.input_data)}")
            return []

    def _prioritize_from_text(self, corpus: str, top_n: int = 5) -> List[str]:
        """
        Uses a simple TF-IDF-like approach to find important keywords in raw text.
        """
        # Simple text normalization
        words = re.findall(r'\b\w+\b', corpus.lower())
        
        # For simplicity, we'll use a single document, so IDF is constant.
        # We effectively just find the most frequent non-trivial words.
        # A real implementation would use a pre-existing corpus for IDF scores.
        stop_words = set(["the", "a", "an", "in", "is", "it", "of", "and", "to", "for"])
        
        word_counts = Counter(word for word in words if word not in stop_words and len(word) > 2)
        
        if not word_counts:
            return []

        # Get the most common words as features
        prioritized = [word for word, count in word_counts.most_common(top_n)]
        logger.info(f"Prioritized text features: {prioritized}")
        return prioritized

    def _prioritize_from_find_result(self, find_result: FindResultInput, top_n: int = 3) -> List[str]:
        """
        Extracts the most important features from a structured miStudioFind result.
        
        This assumes the 'feature_analysis' dict has some quantifiable metrics.
        Example structure: {"feature_123": {"activation_count": 500, "anomaly_score": 0.9}}
        """
        analysis = find_result.feature_analysis
        if not analysis:
            return []

        scored_features: List[Tuple[str, float]] = []

        # Example scoring logic: combine activation and anomaly score
        for feature_id, metrics in analysis.items():
            if isinstance(metrics, dict):
                activation_count = metrics.get("activation_count", 0)
                anomaly_score = metrics.get("anomaly_score", 0.0)
                
                # Simple weighted score
                score = (activation_count * 0.4) + (anomaly_score * 0.6)
                scored_features.append((feature_id, score))

        # Sort by score in descending order and get the top N feature IDs
        scored_features.sort(key=lambda x: x[1], reverse=True)
        
        prioritized = [feature_id for feature_id, score in scored_features[:top_n]]
        logger.info(f"Prioritized features from FindResult: {prioritized}")
        return prioritized