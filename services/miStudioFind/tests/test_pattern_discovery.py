# tests/test_pattern_discovery.py
"""
Unit tests for PatternDiscovery module.

Tests advanced pattern analysis, coherence assessment, and quality validation.
"""

import pytest
import numpy as np

from core.pattern_discovery import PatternDiscovery, PatternSignature
from models.analysis_models import FeatureAnalysisResult


class TestPatternDiscovery:
    """Test suite for PatternDiscovery functionality."""

    @pytest.fixture
    def pattern_discovery(self):
        """Create PatternDiscovery instance for testing."""
        return PatternDiscovery()

    @pytest.fixture
    def sample_feature_result(self):
        """Create sample feature analysis result for testing."""
        return FeatureAnalysisResult(
            feature_id=42,
            top_activations=[
                {
                    "text": "The medical diagnosis requires careful analysis",
                    "activation_value": 4.5,
                    "text_index": 10,
                    "ranking": 1,
                },
                {
                    "text": "Medical treatment involves diagnostic procedures",
                    "activation_value": 4.2,
                    "text_index": 25,
                    "ranking": 2,
                },
                {
                    "text": "Clinical diagnosis is essential for patient care",
                    "activation_value": 3.8,
                    "text_index": 33,
                    "ranking": 3,
                },
            ],
            activation_statistics={
                "mean_activation": 2.1,
                "max_activation": 4.5,
                "activation_frequency": 0.15,
            },
            coherence_score=0.7,
            quality_level="medium",
            pattern_keywords=["medical", "diagnosis", "clinical"],
        )

    def test_pattern_discovery_initialization(self, pattern_discovery):
        """Test PatternDiscovery initialization."""
        assert hasattr(pattern_discovery, "linguistic_patterns")
        assert hasattr(pattern_discovery, "semantic_categories")
        assert len(pattern_discovery.linguistic_patterns) > 0
        assert len(pattern_discovery.semantic_categories) > 0

    def test_detect_pattern_coherence(self, pattern_discovery, sample_feature_result):
        """Test pattern coherence detection."""
        coherence_score = pattern_discovery.detect_pattern_coherence(
            sample_feature_result
        )

        assert 0.0 <= coherence_score <= 1.0
        assert isinstance(coherence_score, float)

    def test_detect_pattern_coherence_empty_activations(self, pattern_discovery):
        """Test pattern coherence with empty activations."""
        empty_result = FeatureAnalysisResult(
            feature_id=0,
            top_activations=[],
            activation_statistics={},
            coherence_score=0.0,
            quality_level="low",
            pattern_keywords=[],
        )

        coherence_score = pattern_discovery.detect_pattern_coherence(empty_result)
        assert coherence_score == 0.0

    def test_identify_outliers(self, pattern_discovery, sample_feature_result):
        """Test outlier identification."""
        outliers = pattern_discovery.identify_outliers(sample_feature_result)

        assert isinstance(outliers, list)
        assert all(isinstance(idx, int) for idx in outliers)
        assert all(
            0 <= idx < len(sample_feature_result.top_activations) for idx in outliers
        )

    def test_measure_diversity_score(self, pattern_discovery, sample_feature_result):
        """Test diversity score measurement."""
        diversity_score = pattern_discovery.measure_diversity_score(
            sample_feature_result
        )

        assert 0.0 <= diversity_score <= 1.0
        assert isinstance(diversity_score, float)

    def test_measure_diversity_score_single_activation(self, pattern_discovery):
        """Test diversity score with single activation."""
        single_result = FeatureAnalysisResult(
            feature_id=0,
            top_activations=[
                {
                    "text": "Single text sample",
                    "activation_value": 1.0,
                    "text_index": 0,
                    "ranking": 1,
                }
            ],
            activation_statistics={},
            coherence_score=0.0,
            quality_level="low",
            pattern_keywords=[],
        )

        diversity_score = pattern_discovery.measure_diversity_score(single_result)
        assert diversity_score == 0.0

    def test_validate_feature_quality_comprehensive(
        self, pattern_discovery, sample_feature_result
    ):
        """Test comprehensive feature quality validation."""
        quality_assessment = pattern_discovery.validate_feature_quality(
            sample_feature_result
        )

        required_keys = [
            "enhanced_coherence_score",
            "outlier_count",
            "outlier_ratio",
            "diversity_score",
            "pattern_signature",
            "quality_factors",
            "overall_quality_score",
            "quality_classification",
            "interpretability_ready",
        ]

        for key in required_keys:
            assert key in quality_assessment

        # Verify value ranges
        assert 0.0 <= quality_assessment["enhanced_coherence_score"] <= 1.0
        assert quality_assessment["outlier_count"] >= 0
        assert 0.0 <= quality_assessment["outlier_ratio"] <= 1.0
        assert 0.0 <= quality_assessment["diversity_score"] <= 1.0
        assert 0.0 <= quality_assessment["overall_quality_score"] <= 1.0
        assert quality_assessment["quality_classification"] in [
            "excellent",
            "good",
            "fair",
            "poor",
        ]
        assert isinstance(quality_assessment["interpretability_ready"], bool)

    def test_linguistic_pattern_detection(self, pattern_discovery):
        """Test linguistic pattern detection."""
        # Test medical terminology detection
        medical_text = "The patient requires immediate medical diagnosis and treatment"
        medical_pattern = pattern_discovery.linguistic_patterns["medical_terms"]
        matches = medical_pattern.findall(medical_text)

        assert len(matches) > 0
        assert any(
            "medical" in match.lower() or "patient" in match.lower()
            for match in matches
        )

        # Test legal language detection
        legal_text = "The contract hereby establishes the terms and conditions pursuant to applicable law"
        legal_pattern = pattern_discovery.linguistic_patterns["legal_language"]
        matches = legal_pattern.findall(legal_text)

        assert len(matches) > 0

    def test_create_pattern_signature(self, pattern_discovery):
        """Test pattern signature creation."""
        texts = [
            "Medical diagnosis requires expertise",
            "Clinical diagnosis involves testing",
            "Patient diagnosis and treatment planning",
        ]

        signature = pattern_discovery._create_pattern_signature(texts)

        assert isinstance(signature, PatternSignature)
        assert 0.0 <= signature.confidence <= 1.0
        assert isinstance(signature.keywords, list)
        assert (
            signature.semantic_category in pattern_discovery.semantic_categories.keys()
        )
        assert isinstance(signature.linguistic_features, dict)
