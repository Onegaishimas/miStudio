# src/scorers/ablation_scorer.py - Fixed version with simplified parameters
"""
Implementation of the Task-Based Utility Scorer using feature ablation.
Fixed to match the simplified parameter structure used in main.py
"""
import logging
from typing import List, Dict, Any
from .base_scorer import BaseScorer

logger = logging.getLogger(__name__)

class AblationScorer(BaseScorer):
    """
    Scores features by measuring the drop in model performance (utility)
    when a feature is "ablated" or removed from the model's computation.
    
    This simplified implementation provides utility scoring based on feature
    properties without requiring actual model loading.
    """

    @property
    def name(self) -> str:
        return "ablation_scorer"

    def score(self, features: List[Dict[str, Any]], **kwargs) -> List[Dict[str, Any]]:
        """
        Calculates an ablation-based utility score for each feature.
        
        Args:
            features: The list of feature data.
            **kwargs: Simplified parameters:
                - name: Score name (required)
                - threshold: Minimum threshold for utility scoring (default: 0.3)
                - analysis_type: Type of analysis (default: "complexity_assessment")
        
        Returns:
            The enriched list of features with ablation scores.
        """
        score_name = kwargs.get("name")
        threshold = kwargs.get("threshold", 0.3)
        analysis_type = kwargs.get("analysis_type", "complexity_assessment")
        
        if not score_name:
            raise ValueError("AblationScorer requires a 'name' parameter.")

        logger.info(f"Starting ablation scoring for '{score_name}' with threshold {threshold}")
        logger.info(f"Analysis type: {analysis_type}")

        for feature in features:
            # Simplified ablation scoring based on feature coherence and activation
            coherence_score = feature.get("coherence_score", 0.0)
            activation_strength = feature.get("max_activation", 0.0)
            
            # Calculate utility score based on feature properties
            if isinstance(coherence_score, (int, float)) and isinstance(activation_strength, (int, float)):
                # Higher coherence and activation = higher utility when ablated
                base_utility = coherence_score * 0.7 + min(activation_strength / 10.0, 1.0) * 0.3
                
                # Apply threshold - features below threshold have minimal utility
                utility_score = max(0.0, base_utility - threshold)
                
                # Adjust based on analysis type
                if analysis_type == "complexity_assessment":
                    # Favor features with higher complexity patterns
                    pattern_complexity = len(feature.get("top_activating_examples", []))
                    complexity_bonus = min(pattern_complexity / 20.0, 0.2)  # Up to 0.2 bonus
                    utility_score += complexity_bonus
                elif analysis_type == "importance_ranking":
                    # Focus purely on activation strength
                    utility_score = utility_score * 1.2  # Boost important features
                
                # Cap at reasonable maximum
                utility_score = min(utility_score, 1.0)
            else:
                utility_score = 0.0
            
            feature[score_name] = round(utility_score, 6)

        logger.info(f"Ablation scoring for '{score_name}' complete.")
        return features