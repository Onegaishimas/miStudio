# utils/statistics_engine.py
"""
Statistical analysis and quality metrics for miStudioFind service.

This module provides comprehensive statistical analysis functions for
feature quality assessment and coherence measurement.
"""

import numpy as np
import logging
from typing import List, Dict, Any, Tuple, Optional
from scipy import stats
from collections import Counter
import warnings

# Suppress scipy warnings for cleaner logs
warnings.filterwarnings("ignore", category=RuntimeWarning)

logger = logging.getLogger(__name__)


class StatisticsEngine:
    """Comprehensive statistical analysis engine for feature quality assessment."""

    def __init__(self):
        """Initialize StatisticsEngine."""
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    def compute_activation_distributions(
        self, activations: np.ndarray
    ) -> Dict[str, Any]:
        """
        Compute comprehensive activation distribution statistics.

        Args:
            activations: Array of activation values

        Returns:
            Dictionary containing distribution statistics
        """
        if len(activations) == 0:
            return {"error": "No activations provided"}

        # Basic statistics
        basic_stats = {
            "count": len(activations),
            "mean": float(np.mean(activations)),
            "std": float(np.std(activations)),
            "min": float(np.min(activations)),
            "max": float(np.max(activations)),
            "median": float(np.median(activations)),
        }

        # Percentiles
        percentiles = {
            "p25": float(np.percentile(activations, 25)),
            "p75": float(np.percentile(activations, 75)),
            "p90": float(np.percentile(activations, 90)),
            "p95": float(np.percentile(activations, 95)),
            "p99": float(np.percentile(activations, 99)),
        }

        # Distribution shape
        try:
            skewness = float(stats.skew(activations))
            kurtosis = float(stats.kurtosis(activations))
        except Exception:
            skewness = 0.0
            kurtosis = 0.0

        # Dispersion measures
        iqr = percentiles["p75"] - percentiles["p25"]
        coefficient_of_variation = basic_stats["std"] / (basic_stats["mean"] + 1e-10)

        return {
            "basic_statistics": basic_stats,
            "percentiles": percentiles,
            "shape_statistics": {
                "skewness": skewness,
                "kurtosis": kurtosis,
                "iqr": iqr,
                "coefficient_of_variation": coefficient_of_variation,
            },
        }

    def calculate_coherence_scores(
        self, texts: List[str], method: str = "comprehensive"
    ) -> Dict[str, float]:
        """
        Calculate various coherence scores for a set of texts.

        Args:
            texts: List of text snippets
            method: Coherence calculation method

        Returns:
            Dictionary containing different coherence scores
        """
        if len(texts) < 2:
            return {"error": "Need at least 2 texts for coherence calculation"}

        if method == "comprehensive":
            return self._calculate_comprehensive_coherence(texts)
        elif method == "lexical":
            return {"lexical_coherence": self._calculate_lexical_coherence(texts)}
        elif method == "semantic":
            return {"semantic_coherence": self._calculate_semantic_coherence(texts)}
        else:
            raise ValueError(f"Unknown coherence method: {method}")

    def _calculate_comprehensive_coherence(self, texts: List[str]) -> Dict[str, float]:
        """Calculate comprehensive coherence using multiple methods."""
        # Lexical coherence
        lexical_score = self._calculate_lexical_coherence(texts)

        # Semantic coherence
        semantic_score = self._calculate_semantic_coherence(texts)

        # Syntactic coherence
        syntactic_score = self._calculate_syntactic_coherence(texts)

        # Length coherence
        length_score = self._calculate_length_coherence(texts)

        # Combined score
        weights = {"lexical": 0.3, "semantic": 0.4, "syntactic": 0.2, "length": 0.1}
        combined_score = (
            weights["lexical"] * lexical_score
            + weights["semantic"] * semantic_score
            + weights["syntactic"] * syntactic_score
            + weights["length"] * length_score
        )

        return {
            "lexical_coherence": lexical_score,
            "semantic_coherence": semantic_score,
            "syntactic_coherence": syntactic_score,
            "length_coherence": length_score,
            "combined_coherence": combined_score,
        }

    def _calculate_lexical_coherence(self, texts: List[str]) -> float:
        """Calculate lexical coherence based on word overlap."""
        if len(texts) < 2:
            return 0.0

        # Tokenize and normalize
        word_sets = []
        for text in texts:
            words = set(text.lower().split())
            word_sets.append(words)

        # Calculate pairwise Jaccard similarities
        similarities = []
        for i in range(len(word_sets)):
            for j in range(i + 1, len(word_sets)):
                intersection = len(word_sets[i] & word_sets[j])
                union = len(word_sets[i] | word_sets[j])
                similarity = intersection / union if union > 0 else 0.0
                similarities.append(similarity)

        return np.mean(similarities) if similarities else 0.0

    def _calculate_semantic_coherence(self, texts: List[str]) -> float:
        """Calculate semantic coherence based on topic consistency."""
        # Simple semantic coherence based on content words
        content_words = set()
        text_word_sets = []

        # Common stop words to exclude
        stop_words = {
            "the",
            "and",
            "or",
            "but",
            "in",
            "on",
            "at",
            "to",
            "for",
            "of",
            "with",
            "by",
            "a",
            "an",
            "is",
            "are",
            "was",
            "were",
            "be",
            "been",
            "have",
            "has",
            "had",
        }

        for text in texts:
            words = set(word.lower() for word in text.split() if len(word) > 2)
            content_words_in_text = words - stop_words
            text_word_sets.append(content_words_in_text)
            content_words.update(content_words_in_text)

        if not content_words:
            return 0.0

        # Calculate semantic overlap
        overlaps = []
        for i in range(len(text_word_sets)):
            for j in range(i + 1, len(text_word_sets)):
                overlap = len(text_word_sets[i] & text_word_sets[j])
                total_unique = len(text_word_sets[i] | text_word_sets[j])
                overlap_ratio = overlap / total_unique if total_unique > 0 else 0.0
                overlaps.append(overlap_ratio)

        return np.mean(overlaps) if overlaps else 0.0

    def _calculate_syntactic_coherence(self, texts: List[str]) -> float:
        """Calculate syntactic coherence based on structural similarity."""
        if len(texts) < 2:
            return 0.0

        # Extract syntactic features
        features = []
        for text in texts:
            feature_vector = {
                "word_count": len(text.split()),
                "char_count": len(text),
                "punct_count": sum(1 for char in text if char in ".,!?;:"),
                "capital_count": sum(1 for char in text if char.isupper()),
                "question_marks": text.count("?"),
                "exclamation_marks": text.count("!"),
            }
            features.append(feature_vector)

        # Calculate consistency across features
        consistencies = []
        for feature_name in features[0].keys():
            values = [f[feature_name] for f in features]
            if len(values) > 1 and np.std(values) > 0:
                # Coefficient of variation (normalized standard deviation)
                cv = np.std(values) / (np.mean(values) + 1e-10)
                consistency = 1.0 / (1.0 + cv)  # Convert to consistency measure
                consistencies.append(consistency)

        return np.mean(consistencies) if consistencies else 0.0

    def _calculate_length_coherence(self, texts: List[str]) -> float:
        """Calculate coherence based on text length consistency."""
        if len(texts) < 2:
            return 1.0

        lengths = [len(text.split()) for text in texts]
        mean_length = np.mean(lengths)
        std_length = np.std(lengths)

        # Coefficient of variation for length
        cv = std_length / (mean_length + 1e-10)

        # Convert to coherence score (lower CV = higher coherence)
        coherence = 1.0 / (1.0 + cv)

        return coherence

    def generate_quality_metrics(
        self, activations: np.ndarray, texts: List[str], feature_id: int
    ) -> Dict[str, Any]:
        """
        Generate comprehensive quality metrics for a feature.

        Args:
            activations: Feature activation values
            texts: Corresponding text snippets
            feature_id: Feature identifier

        Returns:
            Dictionary containing quality metrics
        """
        metrics = {
            "feature_id": feature_id,
            "sample_count": len(activations),
            "text_count": len(texts),
        }

        # Activation distribution analysis
        if len(activations) > 0:
            distribution_stats = self.compute_activation_distributions(activations)
            metrics["activation_distribution"] = distribution_stats

        # Text coherence analysis
        if len(texts) >= 2:
            coherence_scores = self.calculate_coherence_scores(texts)
            metrics["text_coherence"] = coherence_scores

        # Quality assessment
        overall_quality = self._assess_overall_quality(metrics)
        metrics["overall_quality"] = overall_quality

        return metrics

    def _assess_overall_quality(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Assess overall quality based on multiple metrics."""
        quality_factors = {}

        # Activation quality
        if "activation_distribution" in metrics:
            dist_stats = metrics["activation_distribution"]
            if "basic_statistics" in dist_stats:
                basic = dist_stats["basic_statistics"]

                # Good activation patterns: reasonable mean, not too sparse
                activation_quality = min(
                    1.0, basic["mean"] / 1.0
                )  # Normalize to reasonable range
                quality_factors["activation_strength"] = activation_quality

                # Consistency: lower CV is better
                shape = dist_stats.get("shape_statistics", {})
                cv = shape.get("coefficient_of_variation", 1.0)
                consistency_quality = 1.0 / (1.0 + cv)
                quality_factors["activation_consistency"] = consistency_quality

        # Text coherence quality
        if "text_coherence" in metrics:
            coherence = metrics["text_coherence"]
            if "combined_coherence" in coherence:
                quality_factors["text_coherence"] = coherence["combined_coherence"]

        # Overall quality score
        if quality_factors:
            overall_score = np.mean(list(quality_factors.values()))
        else:
            overall_score = 0.0

        # Quality classification
        if overall_score >= 0.8:
            quality_class = "excellent"
        elif overall_score >= 0.6:
            quality_class = "good"
        elif overall_score >= 0.4:
            quality_class = "fair"
        else:
            quality_class = "poor"

        return {
            "quality_factors": quality_factors,
            "overall_score": overall_score,
            "quality_classification": quality_class,
            "interpretable": overall_score >= 0.5,
        }

    def perform_outlier_analysis(
        self, values: np.ndarray, method: str = "iqr"
    ) -> Dict[str, Any]:
        """
        Perform outlier detection analysis.

        Args:
            values: Array of values to analyze
            method: Outlier detection method ("iqr", "zscore", "isolation")

        Returns:
            Dictionary containing outlier analysis results
        """
        if len(values) == 0:
            return {"error": "No values provided for outlier analysis"}

        outlier_indices = []
        outlier_values = []

        if method == "iqr":
            # Interquartile Range method
            q25, q75 = np.percentile(values, [25, 75])
            iqr = q75 - q25
            lower_bound = q25 - 1.5 * iqr
            upper_bound = q75 + 1.5 * iqr

            for i, value in enumerate(values):
                if value < lower_bound or value > upper_bound:
                    outlier_indices.append(i)
                    outlier_values.append(value)

        elif method == "zscore":
            # Z-score method
            z_scores = np.abs(stats.zscore(values))
            threshold = 2.5

            for i, z_score in enumerate(z_scores):
                if z_score > threshold:
                    outlier_indices.append(i)
                    outlier_values.append(values[i])

        else:
            raise ValueError(f"Unknown outlier detection method: {method}")

        outlier_ratio = len(outlier_indices) / len(values)

        return {
            "method": method,
            "outlier_indices": outlier_indices,
            "outlier_values": outlier_values,
            "outlier_count": len(outlier_indices),
            "outlier_ratio": outlier_ratio,
            "total_samples": len(values),
            "outlier_threshold_acceptable": outlier_ratio
            <= 0.1,  # Less than 10% outliers
        }
