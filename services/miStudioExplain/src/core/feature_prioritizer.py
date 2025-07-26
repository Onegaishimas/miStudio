"""
Feature Prioritizer for miStudioExplain Service

Intelligent feature selection and ranking for explanation generation with
comprehensive priority scoring and batch optimization.
"""

import logging
import math
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass, field
from enum import Enum
import statistics

from .input_manager import FeatureData

logger = logging.getLogger(__name__)


class FeatureComplexity(Enum):
    """Feature complexity levels based on activation patterns and context."""
    SIMPLE = "simple"        # Clear, single-concept patterns
    MEDIUM = "medium"        # Multi-faceted but coherent patterns  
    COMPLEX = "complex"      # Abstract or highly technical patterns


class PatternCategory(Enum):
    """Pattern categories from miStudioFind analysis."""
    TECHNICAL = "technical"           # Code, API, system patterns
    CONVERSATIONAL = "conversational" # Chat, dialogue patterns
    BEHAVIORAL = "behavioral"         # Decision-making patterns
    TEMPORAL = "temporal"            # Time-related patterns
    DOMAIN_SPECIFIC = "domain_specific" # Medical, legal, etc.
    REASONING = "reasoning"          # Logical connections
    EMOTIONAL = "emotional"          # Sentiment, feelings
    GENERAL = "general"              # Uncategorized patterns


class BusinessRelevance(Enum):
    """Business relevance classification for explanations."""
    CRITICAL = "critical"    # Safety, compliance, core functionality
    HIGH = "high"           # Important behavioral patterns
    MEDIUM = "medium"       # Useful insights
    LOW = "low"            # Interesting but not actionable


@dataclass
class PriorityFeature:
    """Feature with comprehensive priority scoring and metadata."""
    feature_id: int
    original_coherence: float
    priority_score: float
    complexity: FeatureComplexity
    category: PatternCategory
    business_relevance: BusinessRelevance
    processing_model: str
    
    # Additional scoring factors
    activation_diversity: float = 0.0
    pattern_clarity: float = 0.0
    safety_relevance: float = 0.0
    novelty_score: float = 0.0
    
    # Processing metadata
    estimated_processing_time: float = 0.0
    recommended_batch: int = 0
    
    def __post_init__(self):
        """Validate priority feature data."""
        if not isinstance(self.feature_id, int) or self.feature_id < 0:
            raise ValueError(f"Invalid feature_id: {self.feature_id}")
        
        if not (0.0 <= self.original_coherence <= 1.0):
            raise ValueError(f"Invalid coherence score: {self.original_coherence}")
        
        if not (0.0 <= self.priority_score <= 10.0):
            raise ValueError(f"Invalid priority score: {self.priority_score}")


@dataclass
class ProcessingBatch:
    """Batch of features for optimized processing."""
    batch_id: int
    features: List[PriorityFeature]
    recommended_model: str
    estimated_total_time: float
    complexity_level: FeatureComplexity
    average_priority: float
    
    def __post_init__(self):
        """Calculate batch statistics."""
        if self.features:
            self.average_priority = statistics.mean(f.priority_score for f in self.features)
        else:
            self.average_priority = 0.0


class FeaturePrioritizer:
    """
    Manages feature selection and ranking for explanation processing.
    
    Implements intelligent prioritization based on multiple factors including
    coherence, business relevance, safety implications, and processing efficiency.
    """
    
    # Quality thresholds
    MIN_COHERENCE_THRESHOLD = 0.4
    HIGH_QUALITY_THRESHOLD = 0.6
    EXCELLENT_THRESHOLD = 0.8
    
    # Priority weights
    WEIGHTS = {
        'coherence': 0.35,
        'business_relevance': 0.25,
        'safety_relevance': 0.20,
        'activation_diversity': 0.10,
        'novelty': 0.10
    }
    
    # Processing time estimates (seconds per feature)
    PROCESSING_TIME_ESTIMATES = {
        FeatureComplexity.SIMPLE: 15,
        FeatureComplexity.MEDIUM: 30,
        FeatureComplexity.COMPLEX: 60
    }
    
    def __init__(self, quality_threshold: float = MIN_COHERENCE_THRESHOLD):
        """
        Initialize FeaturePrioritizer.
        
        Args:
            quality_threshold: Minimum coherence score for processing
            
        Raises:
            ValueError: If quality_threshold is invalid
        """
        if not (0.0 <= quality_threshold <= 1.0):
            raise ValueError("quality_threshold must be between 0.0 and 1.0")
        
        self.quality_threshold = quality_threshold
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        self.logger.info(f"FeaturePrioritizer initialized with threshold: {quality_threshold}")
    
    def filter_quality_features(self, features: List[FeatureData]) -> List[FeatureData]:
        """
        Filter features by quality threshold and basic validation.
        
        Args:
            features: List of features from InputManager
            
        Returns:
            List of features meeting quality criteria
            
        Raises:
            ValueError: If features list is empty or invalid
            
        Example:
            >>> prioritizer = FeaturePrioritizer(0.5)
            >>> quality_features = prioritizer.filter_quality_features(all_features)
            >>> print(f"Filtered to {len(quality_features)} quality features")
        """
        if not features:
            raise ValueError("Features list cannot be empty")
        
        if not isinstance(features, list):
            raise ValueError("Features must be a list")
        
        self.logger.info(f"Filtering {len(features)} features with threshold {self.quality_threshold}")
        
        filtered_features = []
        
        for feature in features:
            try:
                # Basic validation
                if not isinstance(feature, FeatureData):
                    self.logger.warning(f"Skipping non-FeatureData object: {type(feature)}")
                    continue
                
                # Coherence threshold check
                if feature.coherence_score < self.quality_threshold:
                    continue
                
                # Additional quality checks
                if not feature.top_activations:
                    self.logger.debug(f"Skipping feature {feature.feature_id}: no activations")
                    continue
                
                if not feature.pattern_keywords:
                    self.logger.debug(f"Skipping feature {feature.feature_id}: no keywords")
                    continue
                
                filtered_features.append(feature)
                
            except Exception as e:
                self.logger.warning(f"Error processing feature {getattr(feature, 'feature_id', 'unknown')}: {e}")
                continue
        
        filter_rate = len(filtered_features) / len(features) * 100
        self.logger.info(
            f"Quality filtering complete: {len(filtered_features)}/{len(features)} "
            f"features passed ({filter_rate:.1f}%)"
        )
        
        return filtered_features
    
    def calculate_priority_score(self, feature: FeatureData) -> float:
        """
        Calculate comprehensive priority score for a feature.
        
        Args:
            feature: Feature to score
            
        Returns:
            Priority score between 0.0 and 10.0
            
        Example:
            >>> score = prioritizer.calculate_priority_score(feature)
            >>> print(f"Feature {feature.feature_id} priority: {score:.2f}")
        """
        try:
            # Base coherence score (0-10 scale)
            coherence_component = feature.coherence_score * 10
            
            # Business relevance score
            business_component = self._calculate_business_relevance(feature)
            
            # Safety relevance score
            safety_component = self._calculate_safety_relevance(feature)
            
            # Activation diversity score
            diversity_component = self._calculate_activation_diversity(feature)
            
            # Novelty score (based on pattern uniqueness)
            novelty_component = self._calculate_novelty_score(feature)
            
            # Weighted combination
            priority_score = (
                coherence_component * self.WEIGHTS['coherence'] +
                business_component * self.WEIGHTS['business_relevance'] +
                safety_component * self.WEIGHTS['safety_relevance'] +
                diversity_component * self.WEIGHTS['activation_diversity'] +
                novelty_component * self.WEIGHTS['novelty']
            )
            
            # Ensure score is within bounds
            priority_score = max(0.0, min(10.0, priority_score))
            
            self.logger.debug(
                f"Feature {feature.feature_id} priority: {priority_score:.2f} "
                f"(coherence: {coherence_component:.1f}, business: {business_component:.1f}, "
                f"safety: {safety_component:.1f}, diversity: {diversity_component:.1f}, "
                f"novelty: {novelty_component:.1f})"
            )
            
            return priority_score
            
        except Exception as e:
            self.logger.error(f"Error calculating priority for feature {feature.feature_id}: {e}")
            return feature.coherence_score * 5.0  # Fallback to simple scoring
    
    def rank_features(self, features: List[FeatureData]) -> List[PriorityFeature]:
        """
        Rank features by priority for processing.
        
        Args:
            features: List of quality-filtered features
            
        Returns:
            List of PriorityFeature objects sorted by priority (highest first)
            
        Example:
            >>> ranked = prioritizer.rank_features(quality_features)
            >>> top_feature = ranked[0]
            >>> print(f"Top priority: Feature {top_feature.feature_id} ({top_feature.priority_score:.2f})")
        """
        if not features:
            self.logger.warning("No features to rank")
            return []
        
        self.logger.info(f"Ranking {len(features)} features by priority...")
        
        priority_features = []
        
        for feature in features:
            try:
                # Calculate priority score
                priority_score = self.calculate_priority_score(feature)
                
                # Determine complexity
                complexity = self._determine_complexity(feature)
                
                # Determine category
                category = self._map_pattern_category(feature.pattern_category)
                
                # Determine business relevance
                business_relevance = self._determine_business_relevance(feature)
                
                # Select optimal processing model
                processing_model = self._select_processing_model(complexity, category)
                
                # Create priority feature
                priority_feature = PriorityFeature(
                    feature_id=feature.feature_id,
                    original_coherence=feature.coherence_score,
                    priority_score=priority_score,
                    complexity=complexity,
                    category=category,
                    business_relevance=business_relevance,
                    processing_model=processing_model,
                    activation_diversity=self._calculate_activation_diversity(feature),
                    pattern_clarity=self._calculate_pattern_clarity(feature),
                    safety_relevance=self._calculate_safety_relevance(feature),
                    novelty_score=self._calculate_novelty_score(feature),
                    estimated_processing_time=self.PROCESSING_TIME_ESTIMATES[complexity]
                )
                
                priority_features.append(priority_feature)
                
            except Exception as e:
                self.logger.error(f"Error ranking feature {feature.feature_id}: {e}")
                continue
        
        # Sort by priority score (descending)
        priority_features.sort(key=lambda f: f.priority_score, reverse=True)
        
        self.logger.info(
            f"Ranking complete: {len(priority_features)} features ranked. "
            f"Top score: {priority_features[0].priority_score:.2f}, "
            f"Bottom score: {priority_features[-1].priority_score:.2f}"
        )
        
        return priority_features
    
    def batch_features_for_processing(self, features: List[PriorityFeature], 
                                    max_batch_size: int = 10) -> List[ProcessingBatch]:
        """
        Group features into optimized processing batches.
        
        Args:
            features: Ranked priority features
            max_batch_size: Maximum features per batch
            
        Returns:
            List of ProcessingBatch objects optimized for parallel processing
            
        Example:
            >>> batches = prioritizer.batch_features_for_processing(ranked_features)
            >>> for batch in batches:
            ...     print(f"Batch {batch.batch_id}: {len(batch.features)} features")
        """
        if not features:
            return []
        
        if max_batch_size < 1:
            raise ValueError("max_batch_size must be at least 1")
        
        self.logger.info(f"Creating batches for {len(features)} features (max size: {max_batch_size})")
        
        batches = []
        current_batch = []
        batch_id = 0
        
        # Group by complexity and model for efficiency
        features_by_model = {}
        for feature in features:
            model = feature.processing_model
            if model not in features_by_model:
                features_by_model[model] = []
            features_by_model[model].append(feature)
        
        # Create batches within each model group
        for model, model_features in features_by_model.items():
            model_features.sort(key=lambda f: f.complexity.value)  # Group by complexity
            
            for feature in model_features:
                current_batch.append(feature)
                
                # Create batch when max size reached or complexity changes significantly
                if (len(current_batch) >= max_batch_size or 
                    (len(current_batch) > 1 and 
                     current_batch[-1].complexity != current_batch[-2].complexity)):
                    
                    if current_batch:
                        batch = self._create_processing_batch(batch_id, current_batch, model)
                        batches.append(batch)
                        
                        # Update feature batch assignments
                        for f in current_batch:
                            f.recommended_batch = batch_id
                        
                        batch_id += 1
                        current_batch = []
        
        # Handle remaining features
        if current_batch:
            model = current_batch[0].processing_model
            batch = self._create_processing_batch(batch_id, current_batch, model)
            batches.append(batch)
            
            for f in current_batch:
                f.recommended_batch = batch_id
        
        total_time = sum(batch.estimated_total_time for batch in batches)
        self.logger.info(
            f"Created {len(batches)} processing batches. "
            f"Estimated total processing time: {total_time:.1f} seconds ({total_time/60:.1f} minutes)"
        )
        
        return batches
    
    def estimate_processing_time(self, features: List[PriorityFeature]) -> Dict[str, float]:
        """
        Estimate processing time and resource requirements.
        
        Args:
            features: List of priority features to process
            
        Returns:
            Dictionary with time estimates and resource requirements
            
        Example:
            >>> estimates = prioritizer.estimate_processing_time(priority_features)
            >>> print(f"Total time: {estimates['total_time_minutes']:.1f} minutes")
        """
        if not features:
            return {
                'total_time_seconds': 0.0,
                'total_time_minutes': 0.0,
                'by_complexity': {},
                'by_model': {},
                'parallel_time_estimate': 0.0
            }
        
        # Time by complexity
        complexity_times = {}
        for complexity in FeatureComplexity:
            complexity_features = [f for f in features if f.complexity == complexity]
            complexity_times[complexity.value] = {
                'count': len(complexity_features),
                'time_seconds': len(complexity_features) * self.PROCESSING_TIME_ESTIMATES[complexity]
            }
        
        # Time by model
        model_times = {}
        for feature in features:
            model = feature.processing_model
            if model not in model_times:
                model_times[model] = {'count': 0, 'time_seconds': 0.0}
            model_times[model]['count'] += 1
            model_times[model]['time_seconds'] += feature.estimated_processing_time
        
        # Total sequential time
        total_time_seconds = sum(f.estimated_processing_time for f in features)
        
        # Estimate parallel processing time (assuming 4 concurrent workers)
        parallel_workers = 4
        parallel_time = total_time_seconds / parallel_workers
        
        estimates = {
            'total_time_seconds': total_time_seconds,
            'total_time_minutes': total_time_seconds / 60,
            'by_complexity': complexity_times,
            'by_model': model_times,
            'parallel_time_estimate': parallel_time,
            'parallel_time_minutes': parallel_time / 60,
            'recommended_workers': min(parallel_workers, len(features))
        }
        
        self.logger.info(
            f"Processing estimates: {total_time_seconds:.1f}s sequential, "
            f"{parallel_time:.1f}s parallel ({parallel_workers} workers)"
        )
        
        return estimates
    
    def _calculate_business_relevance(self, feature: FeatureData) -> float:
        """Calculate business relevance score (0-10)."""
        score = 5.0  # Base score
        
        # Boost for technical patterns (more interpretable)
        if feature.pattern_category.lower() in ['technical', 'behavioral']:
            score += 2.0
        
        # Boost for features with clear keywords
        if len(feature.pattern_keywords) >= 3:
            score += 1.0
        
        # Boost for high activation frequency
        stats = feature.activation_statistics
        if stats.get('frequency', 0) > 0.01:  # Activates frequently
            score += 1.0
        
        return min(10.0, score)
    
    def _calculate_safety_relevance(self, feature: FeatureData) -> float:
        """Calculate safety relevance score (0-10)."""
        score = 0.0
        
        # Check for safety-related keywords
        safety_keywords = [
            'harmful', 'dangerous', 'illegal', 'unsafe', 'toxic', 'bias',
            'discriminat', 'violence', 'threat', 'attack', 'exploit'
        ]
        
        keywords_text = ' '.join(feature.pattern_keywords).lower()
        safety_matches = sum(1 for keyword in safety_keywords if keyword in keywords_text)
        
        if safety_matches > 0:
            score = min(10.0, safety_matches * 3.0)
        
        # Check activation samples for concerning patterns
        activations_text = ' '.join([
            str(act.get('text', '')) for act in feature.top_activations
        ]).lower()
        
        safety_activation_matches = sum(1 for keyword in safety_keywords if keyword in activations_text)
        if safety_activation_matches > 0:
            score = max(score, min(10.0, safety_activation_matches * 2.0))
        
        return score
    
    def _calculate_activation_diversity(self, feature: FeatureData) -> float:
        """Calculate activation diversity score (0-10)."""
        if not feature.top_activations:
            return 0.0
        
        # Simple diversity based on text length variation and uniqueness
        texts = [str(act.get('text', '')) for act in feature.top_activations]
        lengths = [len(text) for text in texts]
        
        if not lengths:
            return 0.0
        
        # Length diversity
        length_std = statistics.stdev(lengths) if len(lengths) > 1 else 0
        length_diversity = min(5.0, length_std / 20)  # Normalize
        
        # Text uniqueness (simple overlap check)
        unique_words = set()
        total_words = 0
        for text in texts:
            words = text.lower().split()
            unique_words.update(words)
            total_words += len(words)
        
        uniqueness = len(unique_words) / max(1, total_words) * 10
        
        return min(10.0, (length_diversity + uniqueness) / 2)
    
    def _calculate_novelty_score(self, feature: FeatureData) -> float:
        """Calculate novelty/uniqueness score (0-10)."""
        # For now, use inverse frequency as novelty indicator
        frequency = feature.activation_statistics.get('frequency', 0.1)
        
        # Less frequent = more novel
        novelty = (1.0 - min(0.9, frequency)) * 10
        
        return novelty
    
    def _calculate_pattern_clarity(self, feature: FeatureData) -> float:
        """Calculate pattern clarity score (0-10)."""
        score = feature.coherence_score * 10
        
        # Boost for clear keywords
        if len(feature.pattern_keywords) >= 2:
            score += 1.0
        
        # Boost for consistent activation strengths
        activations = feature.top_activations
        if activations and len(activations) > 1:
            strengths = [act.get('activation_strength', 0) for act in activations]
            if strengths and len(set(strengths)) > 1:
                strength_std = statistics.stdev(strengths)
                if strength_std < 0.1:  # Consistent activations
                    score += 1.0
        
        return min(10.0, score)
    
    def _determine_complexity(self, feature: FeatureData) -> FeatureComplexity:
        """Determine feature complexity based on patterns."""
        # Simple heuristics for complexity classification
        keyword_count = len(feature.pattern_keywords)
        activation_count = len(feature.top_activations)
        
        # Technical or domain-specific patterns tend to be more complex
        complex_categories = ['technical', 'domain_specific', 'reasoning']
        is_complex_category = feature.pattern_category.lower() in complex_categories
        
        if is_complex_category and keyword_count >= 5:
            return FeatureComplexity.COMPLEX
        elif keyword_count >= 3 or activation_count >= 5:
            return FeatureComplexity.MEDIUM
        else:
            return FeatureComplexity.SIMPLE
    
    def _map_pattern_category(self, category_str: str) -> PatternCategory:
        """Map string category to PatternCategory enum."""
        category_lower = category_str.lower()
        
        mapping = {
            'technical': PatternCategory.TECHNICAL,
            'conversational': PatternCategory.CONVERSATIONAL,
            'behavioral': PatternCategory.BEHAVIORAL,
            'temporal': PatternCategory.TEMPORAL,
            'domain_specific': PatternCategory.DOMAIN_SPECIFIC,
            'reasoning': PatternCategory.REASONING,
            'emotional': PatternCategory.EMOTIONAL
        }
        
        return mapping.get(category_lower, PatternCategory.GENERAL)
    
    def _determine_business_relevance(self, feature: FeatureData) -> BusinessRelevance:
        """Determine business relevance classification."""
        safety_score = self._calculate_safety_relevance(feature)
        business_score = self._calculate_business_relevance(feature)
        
        if safety_score >= 7.0:
            return BusinessRelevance.CRITICAL
        elif business_score >= 8.0 or feature.coherence_score >= 0.8:
            return BusinessRelevance.HIGH
        elif business_score >= 6.0 or feature.coherence_score >= 0.6:
            return BusinessRelevance.MEDIUM
        else:
            return BusinessRelevance.LOW
    
    def _select_processing_model(self, complexity: FeatureComplexity, 
                               category: PatternCategory) -> str:
        """Select optimal LLM model for processing."""
        # Simple model selection logic
        if complexity == FeatureComplexity.COMPLEX:
            return "llama3.1:70b"  # Use larger model for complex patterns
        elif category in [PatternCategory.TECHNICAL, PatternCategory.REASONING]:
            return "llama3.1:8b"   # Good for technical content
        else:
            return "llama3.1:8b"   # Default model
    
    def _create_processing_batch(self, batch_id: int, features: List[PriorityFeature], 
                               model: str) -> ProcessingBatch:
        """Create a processing batch from a list of features."""
        if not features:
            raise ValueError("Cannot create batch with no features")
        
        # Calculate batch statistics
        total_time = sum(f.estimated_processing_time for f in features)
        avg_priority = statistics.mean(f.priority_score for f in features)
        
        # Determine dominant complexity
        complexity_counts = {}
        for feature in features:
            complexity = feature.complexity
            complexity_counts[complexity] = complexity_counts.get(complexity, 0) + 1
        
        dominant_complexity = max(complexity_counts.keys(), key=lambda k: complexity_counts[k])
        
        return ProcessingBatch(
            batch_id=batch_id,
            features=features.copy(),
            recommended_model=model,
            estimated_total_time=total_time,
            complexity_level=dominant_complexity,
            average_priority=avg_priority
        )