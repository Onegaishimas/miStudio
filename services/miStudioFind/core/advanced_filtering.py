# core/advanced_filtering.py
"""
Advanced filtering and categorization system for miStudioFind.
Provides sophisticated pattern analysis and multi-dimensional filtering.
"""

import re
import numpy as np
from typing import List, Dict, Any, Set, Optional
from collections import defaultdict, Counter
from dataclasses import dataclass
from enum import Enum

class PatternCategory(Enum):
    """Enumeration of pattern categories."""
    TECHNICAL = "technical"
    CONVERSATIONAL = "conversational"  
    DOMAIN_SPECIFIC = "domain_specific"
    BEHAVIORAL = "behavioral"
    LINGUISTIC = "linguistic"
    FACTUAL = "factual"
    REASONING = "reasoning"
    EMOTIONAL = "emotional"
    UNKNOWN = "unknown"

class QualityTier(Enum):
    """Quality tier classification."""
    EXCELLENT = "excellent"      # >0.8 coherence
    GOOD = "good"               # 0.6-0.8 coherence  
    FAIR = "fair"               # 0.4-0.6 coherence
    POOR = "poor"               # <0.4 coherence

@dataclass
class FeatureClassification:
    """Complete classification of a feature."""
    feature_id: int
    primary_category: PatternCategory
    secondary_categories: List[PatternCategory]
    quality_tier: QualityTier
    confidence_score: float
    semantic_tags: List[str]
    behavioral_indicators: List[str]
    complexity_score: float

class AdvancedFeatureFilter:
    """Advanced filtering and categorization engine."""
    
    def __init__(self):
        self.pattern_detectors = self._initialize_pattern_detectors()
        self.semantic_analyzers = self._initialize_semantic_analyzers()
        
    def _initialize_pattern_detectors(self) -> Dict[PatternCategory, Dict[str, Any]]:
        """Initialize pattern detection rules."""
        return {
            PatternCategory.TECHNICAL: {
                'keywords': ['API', 'function', 'code', 'algorithm', 'data', 'system', 'method', 'process'],
                'patterns': [
                    re.compile(r'\b[A-Z]{2,}\b'),  # Acronyms
                    re.compile(r'\b\w+\(\)\b'),    # Function calls
                    re.compile(r'\b\d+\.\d+\b'),   # Version numbers
                ],
                'indicators': ['technical terminology', 'structured format', 'precise specifications']
            },
            
            PatternCategory.CONVERSATIONAL: {
                'keywords': ['hello', 'please', 'thank', 'help', 'question', 'answer', 'chat', 'talk'],
                'patterns': [
                    re.compile(r'\b[Hh]ello\b'),
                    re.compile(r'\b[Pp]lease\b'),
                    re.compile(r'\b[Tt]hank\s+you\b'),
                    re.compile(r'\?$'),  # Questions
                ],
                'indicators': ['politeness markers', 'question format', 'casual language']
            },
            
            PatternCategory.DOMAIN_SPECIFIC: {
                'medical': ['patient', 'diagnosis', 'treatment', 'medical', 'clinical', 'therapy'],
                'legal': ['contract', 'law', 'legal', 'court', 'agreement', 'liability'],
                'financial': ['money', 'investment', 'profit', 'market', 'finance', 'economic'],
                'scientific': ['research', 'study', 'experiment', 'hypothesis', 'analysis', 'data'],
                'indicators': ['domain expertise', 'specialized vocabulary', 'professional context']
            },
            
            PatternCategory.BEHAVIORAL: {
                'keywords': ['decision', 'choice', 'prefer', 'select', 'recommend', 'suggest', 'avoid'],
                'patterns': [
                    re.compile(r'\b[Ii] (would|will|can|should)\b'),
                    re.compile(r'\b[Ll]et me\b'),
                    re.compile(r'\b[Ii] recommend\b'),
                ],
                'indicators': ['decision making', 'preference expression', 'action planning']
            },
            
            PatternCategory.REASONING: {
                'keywords': ['because', 'therefore', 'however', 'although', 'consequently', 'thus', 'since'],
                'patterns': [
                    re.compile(r'\bbecause\b'),
                    re.compile(r'\btherefore\b'),
                    re.compile(r'\bif\s+\w+\s+then\b'),
                ],
                'indicators': ['logical connections', 'causal reasoning', 'argumentation']
            },
            
            PatternCategory.EMOTIONAL: {
                'keywords': ['happy', 'sad', 'excited', 'worried', 'confident', 'frustrated', 'grateful'],
                'patterns': [
                    re.compile(r'\b(very|extremely|quite)\s+\w+ed\b'),
                    re.compile(r'\bfeel\s+\w+\b'),
                ],
                'indicators': ['emotional expression', 'sentiment indicators', 'affective language']
            }
        }
    
    def _initialize_semantic_analyzers(self) -> Dict[str, Any]:
        """Initialize semantic analysis components."""
        return {
            'complexity_indicators': [
                'multi-step process', 'conditional logic', 'abstract concepts',
                'technical depth', 'domain expertise', 'nuanced reasoning'
            ],
            'behavioral_patterns': [
                'tool usage', 'information seeking', 'decision making',
                'problem solving', 'communication style', 'preference expression'
            ],
            'quality_indicators': [
                'consistent vocabulary', 'coherent topics', 'clear patterns',
                'semantic unity', 'contextual relevance'
            ]
        }
    
    def classify_feature(self, feature_data: Dict[str, Any]) -> FeatureClassification:
        """Perform comprehensive classification of a feature."""
        feature_id = feature_data.get('feature_id', 0)
        top_activations = feature_data.get('top_activations', [])
        coherence_score = feature_data.get('coherence_score', 0.0)
        pattern_keywords = feature_data.get('pattern_keywords', [])
        
        # Extract all text for analysis
        texts = [act.get('text', '') for act in top_activations]
        combined_text = ' '.join(texts).lower()
        
        # Primary category detection
        category_scores = self._calculate_category_scores(texts, pattern_keywords)
        primary_category = max(category_scores.items(), key=lambda x: x[1])[0]
        
        # Secondary categories (>0.3 score)
        secondary_categories = [
            cat for cat, score in category_scores.items() 
            if score > 0.3 and cat != primary_category
        ]
        
        # Quality tier
        quality_tier = self._determine_quality_tier(coherence_score, category_scores[primary_category])
        
        # Semantic tags
        semantic_tags = self._extract_semantic_tags(texts, pattern_keywords)
        
        # Behavioral indicators
        behavioral_indicators = self._identify_behavioral_patterns(texts)
        
        # Complexity score
        complexity_score = self._calculate_complexity_score(texts, category_scores)
        
        # Overall confidence
        confidence_score = min(1.0, (coherence_score + category_scores[primary_category]) / 2.0)
        
        return FeatureClassification(
            feature_id=feature_id,
            primary_category=primary_category,
            secondary_categories=secondary_categories,
            quality_tier=quality_tier,
            confidence_score=confidence_score,
            semantic_tags=semantic_tags,
            behavioral_indicators=behavioral_indicators,
            complexity_score=complexity_score
        )
    
    def _calculate_category_scores(self, texts: List[str], keywords: List[str]) -> Dict[PatternCategory, float]:
        """Calculate scores for each pattern category."""
        combined_text = ' '.join(texts).lower()
        scores = {}
        
        for category, detectors in self.pattern_detectors.items():
            score = 0.0
            
            if category == PatternCategory.DOMAIN_SPECIFIC:
                # Special handling for domain-specific categories
                domain_scores = []
                for domain, domain_keywords in detectors.items():
                    if domain != 'indicators':
                        domain_score = sum(1 for kw in domain_keywords if kw.lower() in combined_text)
                        domain_scores.append(domain_score)
                score = max(domain_scores) / 10.0 if domain_scores else 0.0
            else:
                # Keyword matching
                if 'keywords' in detectors:
                    keyword_matches = sum(1 for kw in detectors['keywords'] if kw.lower() in combined_text)
                    score += keyword_matches / len(detectors['keywords'])
                
                # Pattern matching
                if 'patterns' in detectors:
                    pattern_matches = sum(1 for pattern in detectors['patterns'] if pattern.search(combined_text))
                    score += pattern_matches / len(detectors['patterns'])
                
                # Keyword overlap
                keyword_overlap = len(set(kw.lower() for kw in keywords) & set(detectors.get('keywords', [])))
                score += keyword_overlap / max(1, len(keywords))
            
            scores[category] = min(1.0, score)
        
        # Ensure at least one category has a reasonable score
        if max(scores.values()) < 0.1:
            scores[PatternCategory.UNKNOWN] = 0.5
        
        return scores
    
    def _determine_quality_tier(self, coherence_score: float, category_confidence: float) -> QualityTier:
        """Determine quality tier based on multiple factors."""
        combined_score = (coherence_score + category_confidence) / 2.0
        
        if combined_score >= 0.8:
            return QualityTier.EXCELLENT
        elif combined_score >= 0.6:
            return QualityTier.GOOD
        elif combined_score >= 0.4:
            return QualityTier.FAIR
        else:
            return QualityTier.POOR
    
    def _extract_semantic_tags(self, texts: List[str], keywords: List[str]) -> List[str]:
        """Extract semantic tags from texts and keywords."""
        tags = set()
        combined_text = ' '.join(texts).lower()
        
        # Add keywords as tags
        tags.update(kw.lower() for kw in keywords if len(kw) > 2)
        
        # Topic-based tags
        topic_indicators = {
            'tool_usage': ['tool', 'function', 'capability', 'available', 'use'],
            'information_seeking': ['information', 'details', 'explain', 'describe', 'tell'],
            'decision_making': ['decide', 'choice', 'option', 'select', 'prefer'],
            'problem_solving': ['solve', 'solution', 'problem', 'issue', 'fix'],
            'communication': ['say', 'respond', 'answer', 'reply', 'communicate'],
            'analysis': ['analyze', 'examine', 'evaluate', 'assess', 'review']
        }
        
        for topic, indicators in topic_indicators.items():
            if any(indicator in combined_text for indicator in indicators):
                tags.add(topic)
        
        return sorted(list(tags))[:10]  # Limit to top 10 tags
    
    def _identify_behavioral_patterns(self, texts: List[str]) -> List[str]:
        """Identify behavioral patterns in the texts."""
        patterns = []
        combined_text = ' '.join(texts).lower()
        
        behavioral_signatures = {
            'capability_assessment': ['can', 'able', 'capability', 'available', 'support'],
            'polite_refusal': ['sorry', 'cannot', 'unable', 'not able', 'unfortunately'],
            'information_provision': ['here', 'information', 'details', 'explanation'],
            'tool_recommendation': ['recommend', 'suggest', 'use', 'try', 'consider'],
            'uncertainty_expression': ['might', 'perhaps', 'possibly', 'maybe', 'could'],
            'confidence_expression': ['definitely', 'certainly', 'sure', 'confident']
        }
        
        for pattern_name, indicators in behavioral_signatures.items():
            if any(indicator in combined_text for indicator in indicators):
                patterns.append(pattern_name)
        
        return patterns
    
    def _calculate_complexity_score(self, texts: List[str], category_scores: Dict[PatternCategory, float]) -> float:
        """Calculate complexity score based on various factors."""
        complexity_factors = []
        
        # Text length complexity
        avg_length = np.mean([len(text.split()) for text in texts])
        length_complexity = min(1.0, avg_length / 20.0)  # Normalize to 20 words
        complexity_factors.append(length_complexity)
        
        # Vocabulary diversity
        all_words = ' '.join(texts).lower().split()
        unique_words = set(all_words)
        diversity = len(unique_words) / max(1, len(all_words))
        complexity_factors.append(diversity)
        
        # Category mixing (multiple strong categories = higher complexity)
        strong_categories = sum(1 for score in category_scores.values() if score > 0.5)
        category_complexity = min(1.0, strong_categories / 3.0)
        complexity_factors.append(category_complexity)
        
        # Technical indicator presence
        technical_indicators = ['system', 'process', 'method', 'analysis', 'function']
        tech_presence = sum(1 for indicator in technical_indicators 
                          if indicator in ' '.join(texts).lower()) / len(technical_indicators)
        complexity_factors.append(tech_presence)
        
        return np.mean(complexity_factors)
    
    def filter_features(self, features: List[Dict[str, Any]], filters: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Apply advanced filters to feature list."""
        filtered_features = []
        
        for feature in features:
            classification = self.classify_feature(feature)
            
            # Apply filters
            if self._passes_filters(classification, feature, filters):
                # Add classification data to feature
                feature['classification'] = {
                    'primary_category': classification.primary_category.value,
                    'secondary_categories': [cat.value for cat in classification.secondary_categories],
                    'quality_tier': classification.quality_tier.value,
                    'confidence_score': classification.confidence_score,
                    'semantic_tags': classification.semantic_tags,
                    'behavioral_indicators': classification.behavioral_indicators,
                    'complexity_score': classification.complexity_score
                }
                filtered_features.append(feature)
        
        return filtered_features
    
    def _passes_filters(self, classification: FeatureClassification, feature: Dict[str, Any], filters: Dict[str, Any]) -> bool:
        """Check if feature passes all specified filters."""
        
        # Category filter
        if 'categories' in filters:
            allowed_categories = [PatternCategory(cat) for cat in filters['categories']]
            if classification.primary_category not in allowed_categories:
                if not any(cat in allowed_categories for cat in classification.secondary_categories):
                    return False
        
        # Quality tier filter
        if 'quality_tiers' in filters:
            allowed_tiers = [QualityTier(tier) for tier in filters['quality_tiers']]
            if classification.quality_tier not in allowed_tiers:
                return False
        
        # Minimum coherence filter
        if 'min_coherence' in filters:
            if feature.get('coherence_score', 0) < filters['min_coherence']:
                return False
        
        # Minimum confidence filter
        if 'min_confidence' in filters:
            if classification.confidence_score < filters['min_confidence']:
                return False
        
        # Semantic tag filter
        if 'required_tags' in filters:
            required_tags = set(filters['required_tags'])
            feature_tags = set(classification.semantic_tags)
            if not required_tags.issubset(feature_tags):
                return False
        
        # Behavioral pattern filter
        if 'behavioral_patterns' in filters:
            required_patterns = set(filters['behavioral_patterns'])
            feature_patterns = set(classification.behavioral_indicators)
            if not required_patterns.issubset(feature_patterns):
                return False
        
        # Complexity range filter
        if 'complexity_range' in filters:
            min_complexity, max_complexity = filters['complexity_range']
            if not (min_complexity <= classification.complexity_score <= max_complexity):
                return False
        
        return True
    
    def generate_category_summary(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate summary of feature categorization."""
        classifications = [self.classify_feature(feature) for feature in features]
        
        # Category distribution
        category_counts = Counter(c.primary_category.value for c in classifications)
        
        # Quality distribution
        quality_counts = Counter(c.quality_tier.value for c in classifications)
        
        # Semantic tag frequency
        all_tags = []
        for c in classifications:
            all_tags.extend(c.semantic_tags)
        tag_frequency = Counter(all_tags)
        
        # Behavioral pattern frequency
        all_behaviors = []
        for c in classifications:
            all_behaviors.extend(c.behavioral_indicators)
        behavior_frequency = Counter(all_behaviors)
        
        # Complexity distribution
        complexity_scores = [c.complexity_score for c in classifications]
        
        return {
            'total_features': len(features),
            'category_distribution': dict(category_counts),
            'quality_distribution': dict(quality_counts),
            'top_semantic_tags': dict(tag_frequency.most_common(20)),
            'top_behavioral_patterns': dict(behavior_frequency.most_common(15)),
            'complexity_statistics': {
                'mean': np.mean(complexity_scores),
                'std': np.std(complexity_scores),
                'min': np.min(complexity_scores),
                'max': np.max(complexity_scores),
                'median': np.median(complexity_scores)
            },
            'confidence_statistics': {
                'mean': np.mean([c.confidence_score for c in classifications]),
                'high_confidence_count': sum(1 for c in classifications if c.confidence_score > 0.8)
            }
        }