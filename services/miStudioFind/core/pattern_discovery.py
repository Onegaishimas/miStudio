# core/pattern_discovery.py
"""
Advanced pattern discovery and quality assessment for miStudioFind service.

This module implements sophisticated pattern analysis, coherence assessment,
outlier detection, and diversity measurement for feature interpretability.
"""

import logging
import re
import numpy as np
from typing import List, Dict, Any, Set, Tuple, Optional
from collections import Counter, defaultdict
from dataclasses import dataclass

from models.analysis_models import FeatureAnalysisResult
from config.find_config import config

logger = logging.getLogger(__name__)


@dataclass
class PatternSignature:
    """Signature representing a discovered pattern in feature activations."""

    pattern_type: str
    confidence: float
    keywords: List[str]
    linguistic_features: Dict[str, Any]
    semantic_category: str


class PatternDiscovery:
    """Advanced pattern discovery and quality assessment engine."""

    def __init__(self):
        """Initialize PatternDiscovery with linguistic and semantic analysis tools."""
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

        # Common linguistic patterns
        self.linguistic_patterns = {
            "technical_terms": re.compile(
                r"\b[A-Z]{2,}|\b\w*(?:API|HTTP|JSON|XML|SQL|CPU|GPU|RAM|URL)\w*\b",
                re.IGNORECASE,
            ),
            "legal_language": re.compile(
                r"\b(?:shall|pursuant|thereof|hereby|whereas|contract|agreement|liability|compliance)\b",
                re.IGNORECASE,
            ),
            "medical_terms": re.compile(
                r"\b(?:diagnosis|treatment|patient|medical|clinical|therapy|syndrome|disease|symptom)\b",
                re.IGNORECASE,
            ),
            "financial_terms": re.compile(
                r"\b(?:revenue|profit|loss|investment|portfolio|market|stock|bond|equity|dividend)\b",
                re.IGNORECASE,
            ),
            "academic_language": re.compile(
                r"\b(?:research|study|analysis|hypothesis|methodology|conclusion|evidence|data)\b",
                re.IGNORECASE,
            ),
            "emotional_language": re.compile(
                r"\b(?:happy|sad|angry|excited|frustrated|disappointed|amazed|grateful|worried|confident)\b",
                re.IGNORECASE,
            ),
        }

        # Semantic categories for classification
        self.semantic_categories = {
            "technical": ["technical_terms", "programming", "system"],
            "legal": ["legal_language", "regulatory", "compliance"],
            "medical": ["medical_terms", "health", "clinical"],
            "financial": ["financial_terms", "business", "economic"],
            "academic": ["academic_language", "research", "scientific"],
            "emotional": ["emotional_language", "sentiment", "feeling"],
            "general": ["common", "everyday", "mixed"],
        }

    def detect_pattern_coherence(self, feature_result: FeatureAnalysisResult) -> float:
        """
        Detect pattern coherence in feature activations using advanced linguistic analysis.

        Args:
            feature_result: Feature analysis result containing top activations

        Returns:
            Enhanced coherence score between 0.0 and 1.0
        """
        if (
            not feature_result.top_activations
            or len(feature_result.top_activations) < 2
        ):
            return 0.0

        texts = [activation["text"] for activation in feature_result.top_activations]

        # Multiple coherence measures
        lexical_coherence = self._calculate_lexical_coherence(texts)
        syntactic_coherence = self._calculate_syntactic_coherence(texts)
        semantic_coherence = self._calculate_semantic_coherence(texts)
        pattern_coherence = self._calculate_pattern_coherence(texts)

        # Weighted combination of coherence measures
        weights = {
            "lexical": 0.25,
            "syntactic": 0.20,
            "semantic": 0.35,
            "pattern": 0.20,
        }

        total_coherence = (
            weights["lexical"] * lexical_coherence
            + weights["syntactic"] * syntactic_coherence
            + weights["semantic"] * semantic_coherence
            + weights["pattern"] * pattern_coherence
        )

        self.logger.debug(
            f"Coherence analysis for feature {feature_result.feature_id}: "
            f"lexical={lexical_coherence:.3f}, syntactic={syntactic_coherence:.3f}, "
            f"semantic={semantic_coherence:.3f}, pattern={pattern_coherence:.3f}, "
            f"total={total_coherence:.3f}"
        )

        return min(1.0, max(0.0, total_coherence))

    def _calculate_lexical_coherence(self, texts: List[str]) -> float:
        """Calculate coherence based on lexical overlap and vocabulary consistency."""
        if len(texts) < 2:
            return 0.0

        # Tokenize and normalize
        word_sets = []
        all_words = set()

        for text in texts:
            words = set(self._normalize_text(text).split())
            word_sets.append(words)
            all_words.update(words)

        if len(all_words) == 0:
            return 0.0

        # Calculate pairwise Jaccard similarity
        similarities = []
        for i in range(len(word_sets)):
            for j in range(i + 1, len(word_sets)):
                intersection = len(word_sets[i] & word_sets[j])
                union = len(word_sets[i] | word_sets[j])
                similarity = intersection / union if union > 0 else 0.0
                similarities.append(similarity)

        return np.mean(similarities) if similarities else 0.0

    def _calculate_syntactic_coherence(self, texts: List[str]) -> float:
        """Calculate coherence based on syntactic patterns and sentence structure."""
        if len(texts) < 2:
            return 0.0

        # Simple syntactic features
        features = []
        for text in texts:
            feature_vector = {
                "avg_word_length": (
                    np.mean([len(word) for word in text.split()]) if text.split() else 0
                ),
                "sentence_length": len(text.split()),
                "punct_density": (
                    sum(1 for char in text if char in ".,!?;:") / len(text)
                    if text
                    else 0
                ),
                "capital_density": (
                    sum(1 for char in text if char.isupper()) / len(text) if text else 0
                ),
                "has_question": "?" in text,
                "has_exclamation": "!" in text,
            }
            features.append(feature_vector)

        # Calculate consistency across syntactic features
        numeric_features = [
            "avg_word_length",
            "sentence_length",
            "punct_density",
            "capital_density",
        ]
        consistencies = []

        for feature_name in numeric_features:
            values = [f[feature_name] for f in features]
            if len(values) > 1 and np.std(values) > 0:
                # Normalize by mean to get coefficient of variation
                consistency = 1.0 - min(1.0, np.std(values) / (np.mean(values) + 1e-6))
                consistencies.append(consistency)

        return np.mean(consistencies) if consistencies else 0.0

    def _calculate_semantic_coherence(self, texts: List[str]) -> float:
        """Calculate coherence based on semantic similarity and topic consistency."""
        # Identify dominant linguistic patterns
        pattern_scores = defaultdict(list)

        for text in texts:
            for pattern_name, pattern_regex in self.linguistic_patterns.items():
                matches = len(pattern_regex.findall(text))
                pattern_scores[pattern_name].append(matches)

        # Find patterns that are consistently present
        coherent_patterns = 0
        total_patterns = len(self.linguistic_patterns)

        for pattern_name, scores in pattern_scores.items():
            # Pattern is coherent if it appears in most texts
            presence_ratio = sum(1 for score in scores if score > 0) / len(scores)
            if presence_ratio >= 0.5:  # Present in at least 50% of texts
                coherent_patterns += 1

        pattern_coherence = (
            coherent_patterns / total_patterns if total_patterns > 0 else 0.0
        )

        # Add topic consistency based on word overlap in meaningful words
        meaningful_words = self._extract_meaningful_words(texts)
        topic_coherence = self._calculate_topic_consistency(meaningful_words)

        return 0.6 * pattern_coherence + 0.4 * topic_coherence

    def _calculate_pattern_coherence(self, texts: List[str]) -> float:
        """Calculate coherence based on specific textual patterns and regularities."""
        if len(texts) < 2:
            return 0.0

        # Look for common patterns in text structure
        patterns = {
            "starts_with_capital": [
                text[0].isupper() if text else False for text in texts
            ],
            "ends_with_period": [
                text.endswith(".") if text else False for text in texts
            ],
            "contains_numbers": [bool(re.search(r"\d", text)) for text in texts],
            "contains_quotes": ['"' in text or "'" in text for text in texts],
            "has_parentheses": ["(" in text and ")" in text for text in texts],
        }

        # Calculate consistency for each pattern
        consistencies = []
        for pattern_name, values in patterns.items():
            if values:
                # Measure how consistent the pattern is across texts
                true_count = sum(values)
                consistency = max(true_count, len(values) - true_count) / len(values)
                consistencies.append(consistency)

        return np.mean(consistencies) if consistencies else 0.0

    def identify_outliers(self, feature_result: FeatureAnalysisResult) -> List[int]:
        """
        Identify outlier activations that may be noise or anomalous.

        Args:
            feature_result: Feature analysis result

        Returns:
            List of indices of outlier activations
        """
        if len(feature_result.top_activations) < 3:
            return []  # Need at least 3 samples for outlier detection

        activation_values = [
            act["activation_value"] for act in feature_result.top_activations
        ]
        texts = [act["text"] for act in feature_result.top_activations]

        outliers = []

        # Statistical outliers based on activation values
        values_array = np.array(activation_values)
        q75, q25 = np.percentile(values_array, [75, 25])
        iqr = q75 - q25
        lower_bound = q25 - (config.outlier_detection_threshold * iqr)
        upper_bound = q75 + (config.outlier_detection_threshold * iqr)

        statistical_outliers = set()
        for i, value in enumerate(activation_values):
            if value < lower_bound or value > upper_bound:
                statistical_outliers.add(i)

        # Textual outliers based on length and content
        text_lengths = [len(text.split()) for text in texts]
        length_mean = np.mean(text_lengths)
        length_std = np.std(text_lengths)

        textual_outliers = set()
        for i, length in enumerate(text_lengths):
            # Text significantly longer or shorter than average
            if abs(length - length_mean) > 2 * length_std:
                textual_outliers.add(i)

        # Semantic outliers (texts that don't match the dominant pattern)
        dominant_pattern = self._identify_dominant_pattern(texts)
        if dominant_pattern:
            semantic_outliers = set()
            for i, text in enumerate(texts):
                if not self._matches_pattern(text, dominant_pattern):
                    semantic_outliers.add(i)
        else:
            semantic_outliers = set()

        # Combine outlier detection methods
        # An activation is considered an outlier if it's flagged by multiple methods
        for i in range(len(feature_result.top_activations)):
            outlier_count = sum(
                [
                    i in statistical_outliers,
                    i in textual_outliers,
                    i in semantic_outliers,
                ]
            )

            if outlier_count >= 2:  # Flagged by at least 2 methods
                outliers.append(i)

        self.logger.debug(
            f"Identified {len(outliers)} outliers out of {len(feature_result.top_activations)} activations"
        )
        return outliers

    def measure_diversity_score(self, feature_result: FeatureAnalysisResult) -> float:
        """
        Measure diversity of activations to ensure the feature captures varied examples.

        Args:
            feature_result: Feature analysis result

        Returns:
            Diversity score between 0.0 and 1.0
        """
        if len(feature_result.top_activations) < 2:
            return 0.0

        texts = [act["text"] for act in feature_result.top_activations]

        # Lexical diversity
        all_words = []
        unique_words = set()
        for text in texts:
            words = self._normalize_text(text).split()
            all_words.extend(words)
            unique_words.update(words)

        lexical_diversity = len(unique_words) / len(all_words) if all_words else 0.0

        # Length diversity
        lengths = [len(text.split()) for text in texts]
        length_diversity = (
            np.std(lengths) / (np.mean(lengths) + 1e-6) if lengths else 0.0
        )
        length_diversity = min(1.0, length_diversity)  # Normalize

        # Content diversity (different topics/patterns)
        content_diversity = self._calculate_content_diversity(texts)

        # Combine diversity measures
        diversity_score = (
            0.4 * lexical_diversity + 0.3 * length_diversity + 0.3 * content_diversity
        )
        return min(1.0, max(0.0, diversity_score))

    def validate_feature_quality(
        self, feature_result: FeatureAnalysisResult
    ) -> Dict[str, Any]:
        """
        Comprehensive quality validation for a feature.

        Args:
            feature_result: Feature analysis result

        Returns:
            Dictionary containing detailed quality assessment
        """
        # Enhanced coherence analysis
        enhanced_coherence = self.detect_pattern_coherence(feature_result)

        # Outlier detection
        outliers = self.identify_outliers(feature_result)
        outlier_ratio = (
            len(outliers) / len(feature_result.top_activations)
            if feature_result.top_activations
            else 0.0
        )

        # Diversity measurement
        diversity_score = self.measure_diversity_score(feature_result)

        # Pattern identification
        texts = [act["text"] for act in feature_result.top_activations]
        pattern_signature = self._create_pattern_signature(texts)

        # Overall quality score
        quality_factors = {
            "coherence": enhanced_coherence,
            "low_outlier_ratio": 1.0 - outlier_ratio,
            "diversity": diversity_score,
            "pattern_confidence": pattern_signature.confidence,
        }

        overall_quality = np.mean(list(quality_factors.values()))

        # Quality classification
        if overall_quality >= 0.8:
            quality_class = "excellent"
        elif overall_quality >= 0.6:
            quality_class = "good"
        elif overall_quality >= 0.4:
            quality_class = "fair"
        else:
            quality_class = "poor"

        return {
            "enhanced_coherence_score": enhanced_coherence,
            "outlier_count": len(outliers),
            "outlier_ratio": outlier_ratio,
            "diversity_score": diversity_score,
            "pattern_signature": {
                "type": pattern_signature.pattern_type,
                "confidence": pattern_signature.confidence,
                "keywords": pattern_signature.keywords,
                "semantic_category": pattern_signature.semantic_category,
            },
            "quality_factors": quality_factors,
            "overall_quality_score": overall_quality,
            "quality_classification": quality_class,
            "interpretability_ready": overall_quality >= config.coherence_threshold,
        }

    # Helper methods

    def _normalize_text(self, text: str) -> str:
        """Normalize text for analysis."""
        # Convert to lowercase and remove extra whitespace
        text = re.sub(r"\s+", " ", text.lower().strip())
        # Remove punctuation but keep word boundaries
        text = re.sub(r"[^\w\s]", " ", text)
        return text

    def _extract_meaningful_words(self, texts: List[str]) -> List[Set[str]]:
        """Extract meaningful words (excluding stop words) from texts."""
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
            "do",
            "does",
            "did",
            "will",
            "would",
            "could",
            "should",
            "may",
            "might",
            "this",
            "that",
            "these",
            "those",
            "i",
            "you",
            "he",
            "she",
            "it",
            "we",
            "they",
        }

        meaningful_word_sets = []
        for text in texts:
            words = set(self._normalize_text(text).split())
            meaningful_words = words - stop_words
            # Filter out very short words
            meaningful_words = {word for word in meaningful_words if len(word) >= 3}
            meaningful_word_sets.append(meaningful_words)

        return meaningful_word_sets

    def _calculate_topic_consistency(self, word_sets: List[Set[str]]) -> float:
        """Calculate topic consistency based on word overlap."""
        if len(word_sets) < 2:
            return 0.0

        # Find words that appear in multiple texts
        all_words = set()
        for word_set in word_sets:
            all_words.update(word_set)

        if not all_words:
            return 0.0

        # Calculate average overlap
        overlaps = []
        for i in range(len(word_sets)):
            for j in range(i + 1, len(word_sets)):
                intersection = len(word_sets[i] & word_sets[j])
                union = len(word_sets[i] | word_sets[j])
                overlap = intersection / union if union > 0 else 0.0
                overlaps.append(overlap)

        return np.mean(overlaps) if overlaps else 0.0

    def _identify_dominant_pattern(self, texts: List[str]) -> Optional[str]:
        """Identify the dominant linguistic pattern in a set of texts."""
        if not texts:
            return None

        pattern_counts = defaultdict(int)

        for text in texts:
            for pattern_name, pattern_regex in self.linguistic_patterns.items():
                if pattern_regex.search(text):
                    pattern_counts[pattern_name] += 1

        if not pattern_counts:
            return None

        # Find pattern that appears in majority of texts
        dominant_pattern = max(pattern_counts.items(), key=lambda x: x[1])
        if dominant_pattern[1] >= len(texts) * 0.5:  # At least 50% of texts
            return dominant_pattern[0]

        return None

    def _matches_pattern(self, text: str, pattern_name: str) -> bool:
        """Check if text matches a specific linguistic pattern."""
        if pattern_name not in self.linguistic_patterns:
            return False

        pattern_regex = self.linguistic_patterns[pattern_name]
        return bool(pattern_regex.search(text))

    def _calculate_content_diversity(self, texts: List[str]) -> float:
        """Calculate diversity based on content patterns and topics."""
        if len(texts) < 2:
            return 0.0

        # Check diversity across different linguistic patterns
        pattern_presence = defaultdict(set)

        for i, text in enumerate(texts):
            for pattern_name, pattern_regex in self.linguistic_patterns.items():
                if pattern_regex.search(text):
                    pattern_presence[pattern_name].add(i)

        # Calculate how evenly distributed the patterns are
        if not pattern_presence:
            return 0.0

        # Shannon entropy-based diversity
        total_texts = len(texts)
        entropy = 0.0

        for pattern_name, text_indices in pattern_presence.items():
            if text_indices:
                proportion = len(text_indices) / total_texts
                entropy -= proportion * np.log2(proportion + 1e-10)

        # Normalize entropy by maximum possible entropy
        max_entropy = np.log2(len(self.linguistic_patterns))
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0

        return min(1.0, normalized_entropy)

    def _create_pattern_signature(self, texts: List[str]) -> PatternSignature:
        """Create a pattern signature for a set of texts."""
        if not texts:
            return PatternSignature("empty", 0.0, [], {}, "general")

        # Analyze linguistic patterns
        pattern_scores = {}
        for pattern_name, pattern_regex in self.linguistic_patterns.items():
            matches = sum(1 for text in texts if pattern_regex.search(text))
            pattern_scores[pattern_name] = matches / len(texts)

        # Find dominant pattern type
        dominant_pattern = max(pattern_scores.items(), key=lambda x: x[1])
        pattern_type = dominant_pattern[0]
        confidence = dominant_pattern[1]

        # Extract keywords
        all_words = []
        for text in texts:
            all_words.extend(self._normalize_text(text).split())

        word_freq = Counter(all_words)
        # Get most frequent non-stop words
        keywords = [
            word
            for word, count in word_freq.most_common(10)
            if len(word) >= 3 and count >= 2
        ]

        # Determine semantic category
        semantic_category = "general"
        for category, patterns in self.semantic_categories.items():
            if pattern_type in patterns or any(p in pattern_type for p in patterns):
                semantic_category = category
                break

        # Calculate linguistic features
        linguistic_features = {
            "avg_length": np.mean([len(text.split()) for text in texts]),
            "vocab_diversity": (
                len(set(all_words)) / len(all_words) if all_words else 0.0
            ),
            "pattern_distribution": pattern_scores,
            "dominant_pattern_strength": confidence,
        }

        return PatternSignature(
            pattern_type=pattern_type,
            confidence=confidence,
            keywords=keywords[:5],  # Top 5 keywords
            linguistic_features=linguistic_features,
            semantic_category=semantic_category,
        )
