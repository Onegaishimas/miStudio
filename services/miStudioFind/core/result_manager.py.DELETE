# core/result_manager.py
"""
Result management and output structuring for miStudioFind service.

This module handles organizing, validating, and storing analysis results
in formats suitable for downstream services and human inspection.
"""

import json
import logging
import torch
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

from models.analysis_models import FeatureAnalysisResult
from models.api_models import QualityDistribution, FeaturePreview
from config.find_config import config

logger = logging.getLogger(__name__)


class ResultManager:
    """Manages organization, validation, and storage of analysis results."""

    def __init__(self, data_path: str = None):
        """
        Initialize ResultManager.

        Args:
            data_path: Base path for data storage. Defaults to config.data_path.
        """
        self.data_path = Path(data_path or config.data_path)
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    def structure_feature_results(
        self, results: List[FeatureAnalysisResult], processing_metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Structure feature analysis results into organized format.

        Args:
            results: List of feature analysis results
            processing_metadata: Metadata about the processing job

        Returns:
            Structured results dictionary ready for serialization
        """
        self.logger.info(f"Structuring results for {len(results)} features")

        # Organize feature mappings
        feature_mappings = {}
        for result in results:
            feature_mappings[f"feature_{result.feature_id}"] = {
                "top_activations": result.top_activations,
                "statistics": result.activation_statistics,
                "coherence_score": result.coherence_score,
                "quality_level": result.quality_level,
                "pattern_keywords": result.pattern_keywords,
            }

        # Calculate processing statistics
        processing_stats = self._calculate_processing_statistics(
            results, processing_metadata
        )

        structured_results = {
            "feature_mappings": feature_mappings,
            "processing_metadata": processing_stats,
            "quality_summary": self._generate_quality_summary(results),
            "feature_count": len(results),
            "timestamp": datetime.now().isoformat(),
            "service_version": config.service_version,
        }

        self.logger.info("Feature results structured successfully")
        return structured_results

    def generate_summary_statistics(
        self, results: List[FeatureAnalysisResult]
    ) -> Dict[str, Any]:
        """
        Generate comprehensive summary statistics for the analysis.

        Args:
            results: List of feature analysis results

        Returns:
            Dictionary containing summary statistics
        """
        if not results:
            return {"error": "No results to summarize"}

        # Quality distribution
        quality_counts = {"high": 0, "medium": 0, "low": 0}
        coherence_scores = []
        activation_frequencies = []
        pattern_types = []

        for result in results:
            quality_counts[result.quality_level] += 1
            coherence_scores.append(result.coherence_score)

            if "activation_frequency" in result.activation_statistics:
                activation_frequencies.append(
                    result.activation_statistics["activation_frequency"]
                )

            if result.pattern_keywords:
                pattern_types.extend(result.pattern_keywords)

        # Statistical summaries
        coherence_stats = {
            "mean": float(np.mean(coherence_scores)) if coherence_scores else 0.0,
            "std": float(np.std(coherence_scores)) if coherence_scores else 0.0,
            "min": float(np.min(coherence_scores)) if coherence_scores else 0.0,
            "max": float(np.max(coherence_scores)) if coherence_scores else 0.0,
            "median": float(np.median(coherence_scores)) if coherence_scores else 0.0,
        }

        activation_stats = {}
        if activation_frequencies:
            activation_stats = {
                "mean": float(np.mean(activation_frequencies)),
                "std": float(np.std(activation_frequencies)),
                "min": float(np.min(activation_frequencies)),
                "max": float(np.max(activation_frequencies)),
            }

        # Most common pattern keywords
        from collections import Counter

        common_patterns = Counter(pattern_types).most_common(20)

        return {
            "total_features": len(results),
            "quality_distribution": quality_counts,
            "quality_percentages": {
                level: (count / len(results)) * 100
                for level, count in quality_counts.items()
            },
            "coherence_statistics": coherence_stats,
            "activation_statistics": activation_stats,
            "common_patterns": [
                {"keyword": keyword, "frequency": freq}
                for keyword, freq in common_patterns
            ],
            "features_above_threshold": sum(
                1 for score in coherence_scores if score >= config.coherence_threshold
            ),
            "interpretability_rate": (
                sum(
                    1
                    for score in coherence_scores
                    if score >= config.coherence_threshold
                )
                / len(results)
            )
            * 100,
        }

    def create_feature_preview(
        self, results: List[FeatureAnalysisResult], top_n: int = None
    ) -> List[FeaturePreview]:
        """
        Create human-readable preview of top coherent features.

        Args:
            results: List of feature analysis results
            top_n: Number of top features to include in preview

        Returns:
            List of FeaturePreview objects for top features
        """
        top_n = top_n or config.feature_preview_count

        # Sort results by coherence score
        sorted_results = sorted(results, key=lambda r: r.coherence_score, reverse=True)

        previews = []
        for result in sorted_results[:top_n]:
            # Generate pattern description
            pattern_description = self._generate_pattern_description(result)

            # Extract example texts (top 3-5 activations)
            example_texts = [
                (
                    activation["text"][:100] + "..."
                    if len(activation["text"]) > 100
                    else activation["text"]
                )
                for activation in result.top_activations[:5]
            ]

            preview = FeaturePreview(
                feature_id=result.feature_id,
                coherence_score=result.coherence_score,
                pattern_description=pattern_description,
                example_texts=example_texts,
            )
            previews.append(preview)

        self.logger.info(f"Created preview for top {len(previews)} features")
        return previews

    def save_analysis_artifacts(
        self,
        job_id: str,
        structured_results: Dict[str, Any],
        summary_stats: Dict[str, Any],
        feature_preview: List[FeaturePreview],
        source_metadata: Dict[str, Any],
    ) -> Dict[str, str]:
        """
        Save all analysis artifacts to persistent storage.

        Args:
            job_id: Unique job identifier
            structured_results: Main analysis results
            summary_stats: Summary statistics
            feature_preview: Human-readable feature preview
            source_metadata: Metadata from source training job

        Returns:
            Dictionary mapping artifact names to file paths
        """
        self.logger.info(f"Saving analysis artifacts for job: {job_id}")

        # Create output directory
        output_dir = self.data_path / "analysis" / job_id
        output_dir.mkdir(parents=True, exist_ok=True)

        saved_files = {}

        # Save main analysis results (PyTorch format for efficiency)
        analysis_path = output_dir / "feature_analysis.pt"
        torch.save(structured_results, analysis_path)
        saved_files["feature_analysis"] = str(analysis_path)

        # Save comprehensive metadata
        metadata = self._create_comprehensive_metadata(
            job_id, summary_stats, source_metadata
        )
        metadata_path = output_dir / "find_metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2, default=str)
        saved_files["metadata"] = str(metadata_path)

        # Save human-readable feature examples
        preview_data = {
            "job_id": job_id,
            "top_coherent_features": [
                preview.model_dump() for preview in feature_preview
            ],
            "total_features": len(structured_results.get("feature_mappings", {})),
            "quality_summary": summary_stats.get("quality_distribution", {}),
            "generation_timestamp": datetime.now().isoformat(),
            "interpretability_summary": {
                "high_quality_count": summary_stats.get("quality_distribution", {}).get(
                    "high", 0
                ),
                "interpretability_rate": summary_stats.get(
                    "interpretability_rate", 0.0
                ),
                "mean_coherence": summary_stats.get("coherence_statistics", {}).get(
                    "mean", 0.0
                ),
            },
        }

        examples_path = output_dir / "feature_examples.json"
        with open(examples_path, "w") as f:
            json.dump(preview_data, f, indent=2, default=str)
        saved_files["feature_examples"] = str(examples_path)

        # Save processing log for debugging
        log_path = output_dir / "processing_log.json"
        processing_log = {
            "job_id": job_id,
            "processing_start": structured_results.get("processing_metadata", {}).get(
                "start_time"
            ),
            "processing_duration": structured_results.get(
                "processing_metadata", {}
            ).get("total_time_seconds"),
            "features_processed": len(structured_results.get("feature_mappings", {})),
            "configuration_used": {
                "top_k_selections": config.top_k_selections,
                "coherence_threshold": config.coherence_threshold,
                "memory_optimization": config.memory_optimization,
            },
            "quality_breakdown": summary_stats.get("quality_distribution", {}),
            "common_patterns": summary_stats.get("common_patterns", [])[:10],
        }

        with open(log_path, "w") as f:
            json.dump(processing_log, f, indent=2, default=str)
        saved_files["processing_log"] = str(log_path)

        self.logger.info(f"Saved {len(saved_files)} artifact files for job {job_id}")
        return saved_files

    def validate_output_quality(
        self, results: List[FeatureAnalysisResult]
    ) -> Dict[str, Any]:
        """
        Validate the overall quality of analysis outputs.

        Args:
            results: List of feature analysis results

        Returns:
            Dictionary containing quality validation results
        """
        if not results:
            return {
                "valid": False,
                "error": "No results to validate",
                "quality_score": 0.0,
            }

        # Calculate quality metrics
        total_features = len(results)
        high_quality_features = sum(1 for r in results if r.quality_level == "high")
        interpretable_features = sum(
            1 for r in results if r.coherence_score >= config.coherence_threshold
        )

        quality_rate = high_quality_features / total_features
        interpretability_rate = interpretable_features / total_features

        # Check if results meet minimum quality standards
        min_interpretability_rate = 0.3  # At least 30% should be interpretable
        min_quality_rate = 0.1  # At least 10% should be high quality

        validation_checks = {
            "sufficient_interpretable_features": interpretability_rate
            >= min_interpretability_rate,
            "sufficient_high_quality_features": quality_rate >= min_quality_rate,
            "no_empty_results": all(r.top_activations for r in results),
            "coherence_scores_valid": all(
                0.0 <= r.coherence_score <= 1.0 for r in results
            ),
        }

        # Overall validation
        all_checks_passed = all(validation_checks.values())
        overall_quality_score = (quality_rate + interpretability_rate) / 2.0

        return {
            "valid": all_checks_passed,
            "overall_quality_score": overall_quality_score,
            "interpretability_rate": interpretability_rate,
            "high_quality_rate": quality_rate,
            "validation_checks": validation_checks,
            "ready_for_explain_service": all_checks_passed
            and interpretability_rate >= 0.5,
            "total_features": total_features,
            "interpretable_features": interpretable_features,
            "high_quality_features": high_quality_features,
        }

    def prepare_for_explain_service(
        self,
        job_id: str,
        results: List[FeatureAnalysisResult],
        saved_files: Dict[str, str],
    ) -> Dict[str, Any]:
        """
        Prepare results specifically for miStudioExplain service consumption.

        Args:
            job_id: Job identifier
            results: Analysis results
            saved_files: Dictionary of saved file paths

        Returns:
            Dictionary containing explain service preparation info
        """
        # Filter for high-quality, interpretable features
        interpretable_results = [
            r
            for r in results
            if r.coherence_score >= config.coherence_threshold
            and r.quality_level in ["high", "medium"]
        ]

        # Create prioritized feature list for explanation
        prioritized_features = sorted(
            interpretable_results,
            key=lambda r: (r.coherence_score, len(r.pattern_keywords)),
            reverse=True,
        )

        # Generate explanation readiness report
        readiness_info = {
            "job_id": job_id,
            "total_features_analyzed": len(results),
            "features_ready_for_explanation": len(interpretable_results),
            "readiness_percentage": (
                (len(interpretable_results) / len(results)) * 100 if results else 0.0
            ),
            "prioritized_feature_ids": [r.feature_id for r in prioritized_features],
            "high_priority_count": len(
                [r for r in prioritized_features if r.quality_level == "high"]
            ),
            "medium_priority_count": len(
                [r for r in prioritized_features if r.quality_level == "medium"]
            ),
            "input_files_for_explain": saved_files,
            "recommended_explanation_order": [
                {
                    "feature_id": r.feature_id,
                    "coherence_score": r.coherence_score,
                    "pattern_keywords": r.pattern_keywords[:3],  # Top 3 keywords
                    "quality_level": r.quality_level,
                }
                for r in prioritized_features[:50]  # Top 50 for explanation
            ],
        }

        return readiness_info

    # Helper methods

    def _calculate_processing_statistics(
        self, results: List[FeatureAnalysisResult], metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Calculate processing performance statistics."""
        import numpy as np

        processing_times = []
        total_activations = sum(len(r.top_activations) for r in results)

        return {
            "total_features_processed": len(results),
            "total_activations_found": total_activations,
            "average_activations_per_feature": (
                total_activations / len(results) if results else 0
            ),
            "start_time": metadata.get("start_time"),
            "total_time_seconds": metadata.get("processing_time", 0),
            "average_time_per_feature": (
                metadata.get("processing_time", 0) / len(results) if results else 0
            ),
            "top_k_setting": config.top_k_selections,
            "coherence_threshold_used": config.coherence_threshold,
        }

    def _generate_quality_summary(
        self, results: List[FeatureAnalysisResult]
    ) -> QualityDistribution:
        """Generate quality distribution summary."""
        quality_counts = {"high": 0, "medium": 0, "low": 0}
        coherence_scores = []

        for result in results:
            quality_counts[result.quality_level] += 1
            coherence_scores.append(result.coherence_score)

        mean_coherence = float(np.mean(coherence_scores)) if coherence_scores else 0.0

        return QualityDistribution(
            high_quality_features=quality_counts["high"],
            medium_quality_features=quality_counts["medium"],
            low_quality_features=quality_counts["low"],
            mean_coherence_score=mean_coherence,
            coherence_scores=coherence_scores,
        )

    def _generate_pattern_description(self, result: FeatureAnalysisResult) -> str:
        """Generate human-readable pattern description for a feature."""
        if not result.pattern_keywords:
            return f"Feature {result.feature_id} (quality: {result.quality_level})"

        keywords_str = ", ".join(result.pattern_keywords[:3])
        quality_desc = {
            "high": "highly coherent",
            "medium": "moderately coherent",
            "low": "low coherence",
        }.get(result.quality_level, "unknown quality")

        return f"Pattern related to: {keywords_str} ({quality_desc})"

    def _create_comprehensive_metadata(
        self,
        job_id: str,
        summary_stats: Dict[str, Any],
        source_metadata: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Create comprehensive metadata for the analysis job."""
        return {
            "job_id": job_id,
            "source_training_job": source_metadata.get("source_job_id"),
            "service": "miStudioFind",
            "version": config.service_version,
            "timestamp": datetime.now().isoformat(),
            "processing_results": {
                "features_processed": summary_stats.get("total_features", 0),
                "high_quality_features": summary_stats.get(
                    "quality_distribution", {}
                ).get("high", 0),
                "medium_quality_features": summary_stats.get(
                    "quality_distribution", {}
                ).get("medium", 0),
                "low_quality_features": summary_stats.get(
                    "quality_distribution", {}
                ).get("low", 0),
                "processing_time_seconds": summary_stats.get(
                    "coherence_statistics", {}
                ).get("mean", 0),
            },
            "quality_distribution": summary_stats.get("quality_distribution", {}),
            "coherence_statistics": summary_stats.get("coherence_statistics", {}),
            "configuration_used": {
                "top_k_selections": config.top_k_selections,
                "coherence_threshold": config.coherence_threshold,
                "memory_optimization": config.memory_optimization,
                "batch_size": config.batch_size,
            },
            "source_model_info": source_metadata.get("model_info", {}),
            "source_sae_config": source_metadata.get("sae_config", {}),
            "ready_for_explain_service": summary_stats.get("interpretability_rate", 0)
            >= 30.0,
        }
