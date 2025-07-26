# core/feature_analyzer.py  
"""
Core feature analysis engine - Production version.
"""

import logging
import torch
import numpy as np
from typing import List, Dict, Any, Tuple

logger = logging.getLogger(__name__)


class FeatureAnalyzer:
    """Core engine for analyzing individual features and their activations."""
    
    def __init__(self, top_k: int = 20):
        """Initialize FeatureAnalyzer."""
        self.top_k = top_k
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    def analyze_feature(self, feature_id: int, feature_activations: torch.Tensor, texts: List[str]) -> Dict[str, Any]:
        """
        Analyze a single feature and find top activating texts.
        
        Args:
            feature_id: Feature identifier
            feature_activations: Tensor of activations [n_samples, n_features]
            texts: List of text snippets
            
        Returns:
            Analysis results for the feature
        """
        self.logger.debug(f"Analyzing feature {feature_id}")
        
        # Extract activations for this feature
        feature_values = feature_activations[:, feature_id]
        
        # Find top-K activating samples
        top_k_indices = torch.topk(feature_values, min(self.top_k, len(feature_values))).indices
        
        # Create results
        top_activations = []
        for rank, idx in enumerate(top_k_indices):
            idx = int(idx)
            activation_value = float(feature_values[idx])
            
            top_activations.append({
                "text": texts[idx],
                "activation_value": activation_value,
                "text_index": idx,
                "ranking": rank + 1
            })
        
        # Calculate basic statistics
        stats = {
            "mean_activation": float(torch.mean(feature_values)),
            "max_activation": float(torch.max(feature_values)),
            "std_activation": float(torch.std(feature_values)),
            "activation_frequency": float(torch.sum(feature_values > 0.01) / len(feature_values))
        }
        
        # Simple coherence assessment
        coherence_score = self._assess_coherence(top_activations)
        
        # Quality classification
        if coherence_score >= 0.7 and stats["activation_frequency"] >= 0.01:
            quality_level = "high"
        elif coherence_score >= 0.4:
            quality_level = "medium"
        else:
            quality_level = "low"
        
        return {
            "feature_id": feature_id,
            "top_activations": top_activations,
            "statistics": stats,
            "coherence_score": coherence_score,
            "quality_level": quality_level,
            "pattern_keywords": self._extract_keywords(top_activations)
        }
    
    def _assess_coherence(self, top_activations: List[Dict[str, Any]]) -> float:
        """Simple coherence assessment based on text similarity."""
        if len(top_activations) < 2:
            return 0.0
        
        texts = [act["text"] for act in top_activations]
        
        # Simple word overlap analysis
        all_words = []
        word_sets = []
        
        for text in texts:
            words = set(text.lower().split())
            word_sets.append(words)
            all_words.extend(words)
        
        if not all_words:
            return 0.0
        
        # Calculate average pairwise similarity
        similarities = []
        for i in range(len(word_sets)):
            for j in range(i + 1, len(word_sets)):
                intersection = len(word_sets[i] & word_sets[j])
                union = len(word_sets[i] | word_sets[j])
                similarity = intersection / union if union > 0 else 0.0
                similarities.append(similarity)
        
        return np.mean(similarities) if similarities else 0.0
    
    def _extract_keywords(self, top_activations: List[Dict[str, Any]]) -> List[str]:
        """Extract potential keywords from top activating texts."""
        if not top_activations:
            return []
        
        word_counts = {}
        total_texts = len(top_activations)
        
        for activation in top_activations:
            words = activation["text"].lower().split()
            unique_words = set(words)
            
            for word in unique_words:
                if len(word) >= 3:  # Filter short words
                    word_counts[word] = word_counts.get(word, 0) + 1
        
        # Get words that appear in multiple texts
        keywords = [word for word, count in word_counts.items() if count >= max(2, total_texts * 0.3)]
        
        # Sort by frequency and return top 5
        keywords.sort(key=lambda w: word_counts[w], reverse=True)
        return keywords[:5]

    def analyze_all_features(self, activation_data: Dict[str, Any], progress_callback=None) -> List[Dict[str, Any]]:
        """
        Analyze all features in the dataset.
        
        Args:
            activation_data: Complete activation data from miStudioTrain
            progress_callback: Optional callback for progress updates
            
        Returns:
            List of analysis results for all features
        """
        feature_activations = activation_data["feature_activations"]
        texts = activation_data["texts"]
        feature_count = activation_data["feature_count"]
        
        self.logger.info(f"Analyzing {feature_count} features...")
        
        results = []
        for feature_id in range(feature_count):
            try:
                result = self.analyze_feature(feature_id, feature_activations, texts)
                results.append(result)
                
                if progress_callback:
                    progress_callback(feature_id + 1, feature_count, feature_id)
                
                if (feature_id + 1) % 50 == 0:
                    self.logger.info(f"Processed {feature_id + 1}/{feature_count} features")
                    
            except Exception as e:
                self.logger.error(f"Failed to analyze feature {feature_id}: {e}")
                continue
        
        self.logger.info(f"Analysis complete: {len(results)}/{feature_count} features processed")
        return results
