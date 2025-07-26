"""
Context Builder for miStudioExplain Service

Prepares rich context for LLM prompt generation with intelligent text sampling,
pattern extraction, and model-specific optimization.
"""

import logging
import re
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from collections import Counter
import statistics

from .input_manager import FeatureData
from .feature_prioritizer import PriorityFeature, FeatureComplexity, PatternCategory

logger = logging.getLogger(__name__)


@dataclass
class FeatureContext:
    """Rich context for a single feature with extracted patterns and metadata."""
    feature_id: int
    coherence_score: float
    pattern_category: PatternCategory
    pattern_keywords: List[str]
    top_activations: List[str]
    activation_context: Dict[str, Any]
    complexity_indicators: Dict[str, Any]
    
    # Enhanced context fields
    representative_samples: List[str] = field(default_factory=list)
    extracted_patterns: List[str] = field(default_factory=list)
    semantic_themes: List[str] = field(default_factory=list)
    statistical_summary: Dict[str, float] = field(default_factory=dict)
    quality_indicators: Dict[str, Any] = field(default_factory=dict)


@dataclass 
class LLMPromptContext:
    """Complete context for LLM prompt generation with model-specific formatting."""
    feature_context: FeatureContext
    model_specific_formatting: Dict[str, str]
    prompt_template: str
    expected_response_format: str
    
    # Prompt optimization
    token_count_estimate: int = 0
    context_priority_order: List[str] = field(default_factory=list)
    fallback_context: Optional[str] = None


class ContextBuilder:
    """
    Builds comprehensive context for LLM explanation generation.
    
    Extracts meaningful patterns, creates representative samples, and optimizes
    context for different LLM models and complexity levels.
    """
    
    # Context limits and parameters
    MAX_ACTIVATION_SAMPLES = 8
    MAX_TEXT_LENGTH = 200
    MIN_SAMPLE_DIVERSITY = 0.3
    TOKEN_ESTIMATE_RATIO = 0.75  # ~0.75 tokens per character
    
    # Model-specific token limits
    MODEL_TOKEN_LIMITS = {
        "llama3.1:8b": 8192,
        "llama3.1:70b": 8192,
        "gpt-4": 8192,
        "claude-3": 8192
    }
    
    # Prompt templates for different complexity levels
    PROMPT_TEMPLATES = {
        FeatureComplexity.SIMPLE: """
Analyze this AI feature that activates on the following text patterns:

**Pattern Keywords**: {keywords}
**Representative Examples**:
{examples}

**Statistical Summary**: 
- Coherence Score: {coherence:.3f}
- Activation Frequency: {frequency:.4f}
- Pattern Category: {category}

Generate a clear, concise explanation of what this feature detects. Focus on:
1. What specific concept or pattern the feature recognizes
2. Why this pattern is significant for the AI model
3. How this affects the model's behavior or understanding

Keep the explanation accessible but precise.
""",
        
        FeatureComplexity.MEDIUM: """
Analyze this moderately complex AI feature with the following activation patterns:

**Pattern Information**:
- Category: {category}
- Keywords: {keywords}
- Coherence Score: {coherence:.3f}
- Complexity Indicators: {complexity_indicators}

**Representative Activation Examples**:
{examples}

**Pattern Analysis**:
{pattern_analysis}

**Statistical Context**:
{statistical_summary}

Provide a detailed explanation covering:
1. The specific concept, behavior, or linguistic pattern this feature detects
2. The underlying mechanisms or reasoning the AI uses
3. Real-world implications and use cases
4. Any potential limitations or edge cases

Balance technical accuracy with clarity.
""",
        
        FeatureComplexity.COMPLEX: """
Analyze this complex AI feature with sophisticated activation patterns:

**Feature Profile**:
- ID: {feature_id}
- Category: {category} 
- Coherence: {coherence:.3f}
- Complexity Level: Complex

**Pattern Keywords and Themes**:
{keywords_and_themes}

**Diverse Activation Examples**:
{examples}

**Advanced Pattern Analysis**:
{advanced_patterns}

**Contextual Factors**:
{contextual_factors}

**Statistical Profile**:
{statistical_summary}

Generate a comprehensive explanation that addresses:
1. The sophisticated concept or multi-faceted pattern this feature captures
2. How the AI model integrates this understanding into its reasoning
3. The feature's role in broader AI capabilities or behaviors
4. Technical insights about the model's learned representations
5. Potential safety, bias, or interpretability considerations

Provide expert-level analysis while maintaining clarity for technical stakeholders.
"""
    }
    
    def __init__(self):
        """Initialize ContextBuilder with default parameters."""
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.pattern_extractors = self._initialize_pattern_extractors()
        self.semantic_analyzers = self._initialize_semantic_analyzers()
        
        self.logger.info("ContextBuilder initialized with pattern extraction capabilities")
    
    def build_feature_context(self, feature_data: FeatureData, 
                            priority_info: Optional[PriorityFeature] = None) -> FeatureContext:
        """
        Build rich context for a single feature with comprehensive analysis.
        
        Args:
            feature_data: Raw feature data from InputManager
            priority_info: Optional priority information from FeaturePrioritizer
            
        Returns:
            FeatureContext with extracted patterns and enriched metadata
            
        Raises:
            ValueError: If feature_data is invalid
            
        Example:
            >>> context = builder.build_feature_context(feature_data)
            >>> print(f"Extracted {len(context.extracted_patterns)} patterns")
        """
        if not isinstance(feature_data, FeatureData):
            raise ValueError("feature_data must be a FeatureData instance")
        
        self.logger.debug(f"Building context for feature {feature_data.feature_id}")
        
        try:
            # Extract representative samples
            representative_samples = self._select_representative_samples(feature_data)
            
            # Extract patterns from activations
            extracted_patterns = self.extract_key_patterns(feature_data)
            
            # Identify semantic themes
            semantic_themes = self._identify_semantic_themes(feature_data)
            
            # Create statistical summary
            statistical_summary = self._create_statistical_summary(feature_data)
            
            # Analyze quality indicators
            quality_indicators = self._analyze_quality_indicators(feature_data)
            
            # Determine complexity indicators
            complexity_indicators = self._analyze_complexity_indicators(feature_data, priority_info)
            
            # Map category
            pattern_category = self._map_to_pattern_category(feature_data.pattern_category)
            
            # Create activation context
            activation_context = self._build_activation_context(feature_data)
            
            feature_context = FeatureContext(
                feature_id=feature_data.feature_id,
                coherence_score=feature_data.coherence_score,
                pattern_category=pattern_category,
                pattern_keywords=feature_data.pattern_keywords.copy(),
                top_activations=[str(act.get('text', '')) for act in feature_data.top_activations],
                activation_context=activation_context,
                complexity_indicators=complexity_indicators,
                representative_samples=representative_samples,
                extracted_patterns=extracted_patterns,
                semantic_themes=semantic_themes,
                statistical_summary=statistical_summary,
                quality_indicators=quality_indicators
            )
            
            self.logger.debug(
                f"Context built for feature {feature_data.feature_id}: "
                f"{len(representative_samples)} samples, {len(extracted_patterns)} patterns"
            )
            
            return feature_context
            
        except Exception as e:
            self.logger.error(f"Failed to build context for feature {feature_data.feature_id}: {e}")
            raise
    
    def extract_key_patterns(self, feature_data: FeatureData) -> List[str]:
        """
        Extract key patterns from feature activation examples.
        
        Args:
            feature_data: Feature with activation examples
            
        Returns:
            List of identified patterns and linguistic structures
            
        Example:
            >>> patterns = builder.extract_key_patterns(feature_data)
            >>> print(f"Key patterns: {patterns}")
        """
        patterns = []
        
        # Collect all activation texts
        texts = [str(act.get('text', '')) for act in feature_data.top_activations]
        combined_text = ' '.join(texts).lower()
        
        # Apply pattern extractors
        for pattern_name, extractor in self.pattern_extractors.items():
            try:
                found_patterns = extractor(texts, combined_text)
                if found_patterns:
                    patterns.extend([f"{pattern_name}: {p}" for p in found_patterns])
            except Exception as e:
                self.logger.warning(f"Pattern extractor {pattern_name} failed: {e}")
        
        # Remove duplicates and limit results
        unique_patterns = list(set(patterns))[:10]
        
        self.logger.debug(f"Extracted {len(unique_patterns)} unique patterns")
        return unique_patterns
    
    def generate_prompt_context(self, feature_context: FeatureContext, 
                              model_name: str) -> LLMPromptContext:
        """
        Generate model-specific prompt context optimized for the target LLM.
        
        Args:
            feature_context: Rich feature context
            model_name: Target LLM model identifier
            
        Returns:
            LLMPromptContext with optimized prompt and formatting
            
        Example:
            >>> prompt_context = builder.generate_prompt_context(context, "llama3.1:8b")
            >>> print(f"Prompt length: {len(prompt_context.prompt_template)}")
        """
        # Determine complexity for template selection
        complexity = self._infer_complexity_from_context(feature_context)
        
        # Get appropriate template
        template = self.PROMPT_TEMPLATES.get(complexity, self.PROMPT_TEMPLATES[FeatureComplexity.MEDIUM])
        
        # Prepare template variables
        template_vars = self._prepare_template_variables(feature_context, complexity)
        
        # Format the prompt
        try:
            formatted_prompt = template.format(**template_vars)
        except KeyError as e:
            self.logger.warning(f"Template formatting error: {e}. Using fallback.")
            formatted_prompt = self._create_fallback_prompt(feature_context)
        
        # Optimize for model-specific constraints
        optimized_prompt, model_formatting = self._optimize_for_model(formatted_prompt, model_name)
        
        # Estimate token count
        token_estimate = self._estimate_token_count(optimized_prompt)
        
        # Create context priority order
        priority_order = self._determine_context_priority(feature_context, complexity)
        
        # Generate expected response format
        response_format = self._generate_response_format(complexity)
        
        # Create fallback if needed
        fallback_context = None
        if token_estimate > self.MODEL_TOKEN_LIMITS.get(model_name, 8192):
            fallback_context = self._create_compressed_context(feature_context)
        
        prompt_context = LLMPromptContext(
            feature_context=feature_context,
            model_specific_formatting=model_formatting,
            prompt_template=optimized_prompt,
            expected_response_format=response_format,
            token_count_estimate=token_estimate,
            context_priority_order=priority_order,
            fallback_context=fallback_context
        )
        
        self.logger.debug(
            f"Generated prompt context for {model_name}: "
            f"{token_estimate} estimated tokens, complexity: {complexity.value}"
        )
        
        return prompt_context
    
    def optimize_context_for_model(self, context: FeatureContext, model_name: str) -> FeatureContext:
        """
        Optimize context based on model capabilities and constraints.
        
        Args:
            context: Feature context to optimize
            model_name: Target model identifier
            
        Returns:
            Optimized FeatureContext
            
        Example:
            >>> optimized = builder.optimize_context_for_model(context, "llama3.1:70b")
        """
        # Create a copy to avoid modifying original
        optimized_context = FeatureContext(
            feature_id=context.feature_id,
            coherence_score=context.coherence_score,
            pattern_category=context.pattern_category,
            pattern_keywords=context.pattern_keywords.copy(),
            top_activations=context.top_activations.copy(),
            activation_context=context.activation_context.copy(),
            complexity_indicators=context.complexity_indicators.copy(),
            representative_samples=context.representative_samples.copy(),
            extracted_patterns=context.extracted_patterns.copy(),
            semantic_themes=context.semantic_themes.copy(),
            statistical_summary=context.statistical_summary.copy(),
            quality_indicators=context.quality_indicators.copy()
        )
        
        # Model-specific optimizations
        if "70b" in model_name.lower():
            # Larger model can handle more context
            optimized_context.representative_samples = context.representative_samples[:self.MAX_ACTIVATION_SAMPLES]
        elif "8b" in model_name.lower():
            # Smaller model needs more focused context
            optimized_context.representative_samples = context.representative_samples[:5]
            optimized_context.extracted_patterns = context.extracted_patterns[:6]
        
        # Optimize text lengths
        optimized_context.representative_samples = [
            self._truncate_text(sample, self.MAX_TEXT_LENGTH) 
            for sample in optimized_context.representative_samples
        ]
        
        return optimized_context
    
    def _select_representative_samples(self, feature_data: FeatureData) -> List[str]:
        """Select diverse, representative activation samples."""
        if not feature_data.top_activations:
            return []
        
        # Extract texts and their metadata
        samples = []
        for activation in feature_data.top_activations:
            text = str(activation.get('text', '')).strip()
            if text and len(text) > 10:  # Filter very short texts
                samples.append({
                    'text': text,
                    'strength': activation.get('activation_strength', 0.0),
                    'length': len(text)
                })
        
        if not samples:
            return []
        
        # Sort by activation strength
        samples.sort(key=lambda x: x['strength'], reverse=True)
        
        # Select diverse samples
        selected = []
        selected_texts = set()
        
        for sample in samples:
            text = sample['text']
            
            # Skip if too similar to already selected
            if self._is_too_similar(text, selected_texts):
                continue
            
            selected.append(text)
            selected_texts.add(text.lower())
            
            if len(selected) >= self.MAX_ACTIVATION_SAMPLES:
                break
        
        return selected
    
    def _identify_semantic_themes(self, feature_data: FeatureData) -> List[str]:
        """Identify semantic themes from activation patterns."""
        themes = []
        
        # Analyze keywords for themes
        keywords = [kw.lower() for kw in feature_data.pattern_keywords]
        keyword_text = ' '.join(keywords)
        
        # Check for common semantic categories
        semantic_categories = {
            'technical': ['api', 'function', 'code', 'system', 'data', 'algorithm'],
            'temporal': ['time', 'date', 'schedule', 'when', 'duration', 'period'],
            'social': ['people', 'relationship', 'communication', 'social', 'community'],
            'cognitive': ['think', 'understand', 'learn', 'memory', 'knowledge'],
            'emotional': ['feel', 'emotion', 'happy', 'sad', 'angry', 'excited'],
            'spatial': ['location', 'place', 'direction', 'distance', 'geography'],
            'quantitative': ['number', 'amount', 'quantity', 'measure', 'calculate']
        }
        
        for theme, theme_keywords in semantic_categories.items():
            matches = sum(1 for kw in theme_keywords if kw in keyword_text)
            if matches >= 2:
                themes.append(theme)
        
        # Analyze activation texts for additional themes
        activation_texts = [str(act.get('text', '')) for act in feature_data.top_activations]
        combined_text = ' '.join(activation_texts).lower()
        
        # Domain-specific theme detection
        domain_indicators = {
            'medical': ['patient', 'diagnosis', 'treatment', 'medical', 'health'],
            'legal': ['law', 'legal', 'court', 'contract', 'agreement'],
            'financial': ['money', 'payment', 'investment', 'cost', 'price'],
            'educational': ['learn', 'teach', 'student', 'course', 'study'],
            'technical': ['software', 'hardware', 'programming', 'computer']
        }
        
        for domain, indicators in domain_indicators.items():
            matches = sum(1 for indicator in indicators if indicator in combined_text)
            if matches >= 2 and domain not in themes:
                themes.append(f"domain:{domain}")
        
        return themes[:5]  # Limit to top themes
    
    def _create_statistical_summary(self, feature_data: FeatureData) -> Dict[str, float]:
        """Create statistical summary of feature characteristics."""
        stats = feature_data.activation_statistics.copy()
        
        # Add derived statistics
        if feature_data.top_activations:
            strengths = [act.get('activation_strength', 0) for act in feature_data.top_activations]
            text_lengths = [len(str(act.get('text', ''))) for act in feature_data.top_activations]
            
            if strengths:
                stats['activation_strength_mean'] = statistics.mean(strengths)
                stats['activation_strength_std'] = statistics.stdev(strengths) if len(strengths) > 1 else 0.0
                stats['activation_strength_max'] = max(strengths)
            
            if text_lengths:
                stats['text_length_mean'] = statistics.mean(text_lengths)
                stats['text_length_std'] = statistics.stdev(text_lengths) if len(text_lengths) > 1 else 0.0
        
        # Add keyword statistics
        stats['keyword_count'] = len(feature_data.pattern_keywords)
        stats['activation_count'] = len(feature_data.top_activations)
        
        return stats
    
    def _analyze_quality_indicators(self, feature_data: FeatureData) -> Dict[str, Any]:
        """Analyze feature quality indicators."""
        indicators = {
            'coherence_level': 'unknown',
            'keyword_quality': 'unknown',
            'activation_consistency': 'unknown',
            'pattern_clarity': 'unknown'
        }
        
        # Coherence level
        if feature_data.coherence_score >= 0.8:
            indicators['coherence_level'] = 'excellent'
        elif feature_data.coherence_score >= 0.6:
            indicators['coherence_level'] = 'good'
        elif feature_data.coherence_score >= 0.4:
            indicators['coherence_level'] = 'fair'
        else:
            indicators['coherence_level'] = 'poor'
        
        # Keyword quality
        keyword_count = len(feature_data.pattern_keywords)
        if keyword_count >= 5:
            indicators['keyword_quality'] = 'rich'
        elif keyword_count >= 3:
            indicators['keyword_quality'] = 'adequate'
        elif keyword_count >= 1:
            indicators['keyword_quality'] = 'minimal'
        else:
            indicators['keyword_quality'] = 'poor'
        
        # Activation consistency
        if feature_data.top_activations:
            strengths = [act.get('activation_strength', 0) for act in feature_data.top_activations]
            if strengths and len(strengths) > 1:
                std_dev = statistics.stdev(strengths)
                if std_dev < 0.1:
                    indicators['activation_consistency'] = 'high'
                elif std_dev < 0.2:
                    indicators['activation_consistency'] = 'medium'
                else:
                    indicators['activation_consistency'] = 'low'
        
        return indicators
    
    def _analyze_complexity_indicators(self, feature_data: FeatureData, 
                                     priority_info: Optional[PriorityFeature]) -> Dict[str, Any]:
        """Analyze complexity indicators for the feature."""
        indicators = {}
        
        # Basic complexity factors
        indicators['keyword_diversity'] = len(set(feature_data.pattern_keywords))
        indicators['activation_diversity'] = len(feature_data.top_activations)
        indicators['category'] = feature_data.pattern_category
        
        # Text complexity analysis
        if feature_data.top_activations:
            texts = [str(act.get('text', '')) for act in feature_data.top_activations]
            avg_length = statistics.mean([len(text) for text in texts])
            indicators['average_text_length'] = avg_length
            
            # Vocabulary richness
            all_words = []
            for text in texts:
                words = re.findall(r'\w+', text.lower())
                all_words.extend(words)
            
            unique_words = len(set(all_words))
            total_words = len(all_words)
            indicators['vocabulary_richness'] = unique_words / max(1, total_words)
        
        # Priority-based indicators
        if priority_info:
            indicators['complexity_level'] = priority_info.complexity.value
            indicators['business_relevance'] = priority_info.business_relevance.value
            indicators['priority_score'] = priority_info.priority_score
        
        return indicators
    
    def _build_activation_context(self, feature_data: FeatureData) -> Dict[str, Any]:
        """Build comprehensive activation context."""
        context = {
            'total_activations': len(feature_data.top_activations),
            'has_metadata': bool(feature_data.activation_statistics),
            'categories_present': [feature_data.pattern_category] if feature_data.pattern_category else []
        }
        
        # Analyze activation patterns
        if feature_data.top_activations:
            contexts = [act.get('context', '') for act in feature_data.top_activations if act.get('context')]
            if contexts:
                context['context_types'] = list(set(contexts))
            
            # Activation strength distribution
            strengths = [act.get('activation_strength', 0) for act in feature_data.top_activations]
            if strengths:
                context['strength_range'] = {
                    'min': min(strengths),
                    'max': max(strengths),
                    'mean': statistics.mean(strengths)
                }
        
        return context
    
    def _map_to_pattern_category(self, category_str: str) -> PatternCategory:
        """Map string category to PatternCategory enum."""
        category_mapping = {
            'technical': PatternCategory.TECHNICAL,
            'conversational': PatternCategory.CONVERSATIONAL,
            'behavioral': PatternCategory.BEHAVIORAL,
            'temporal': PatternCategory.TEMPORAL,
            'domain_specific': PatternCategory.DOMAIN_SPECIFIC,
            'reasoning': PatternCategory.REASONING,
            'emotional': PatternCategory.EMOTIONAL
        }
        
        return category_mapping.get(category_str.lower(), PatternCategory.GENERAL)
    
    def _initialize_pattern_extractors(self) -> Dict[str, callable]:
        """Initialize pattern extraction functions."""
        return {
            'repeated_phrases': self._extract_repeated_phrases,
            'structural_patterns': self._extract_structural_patterns,
            'linguistic_markers': self._extract_linguistic_markers,
            'domain_terminology': self._extract_domain_terminology
        }
    
    def _initialize_semantic_analyzers(self) -> Dict[str, Any]:
        """Initialize semantic analysis components."""
        return {
            'concept_categories': [
                'abstract_concepts', 'concrete_objects', 'actions_processes',
                'relationships', 'properties_attributes', 'temporal_concepts'
            ],
            'complexity_indicators': [
                'multi_step_reasoning', 'domain_expertise', 'contextual_nuance',
                'abstract_thinking', 'causal_relationships'
            ]
        }
    
    def _extract_repeated_phrases(self, texts: List[str], combined_text: str) -> List[str]:
        """Extract repeated phrases across activation texts."""
        patterns = []
        
        # Find common word sequences
        words = combined_text.split()
        word_counts = Counter(words)
        
        # Find frequent words (appearing in multiple texts)
        frequent_words = [word for word, count in word_counts.items() if count >= 2 and len(word) > 3]
        
        if frequent_words:
            patterns.append(f"Frequent terms: {', '.join(frequent_words[:5])}")
        
        # Look for repeated bigrams
        bigrams = [' '.join(words[i:i+2]) for i in range(len(words)-1)]
        bigram_counts = Counter(bigrams)
        frequent_bigrams = [bg for bg, count in bigram_counts.items() if count >= 2]
        
        if frequent_bigrams:
            patterns.append(f"Repeated phrases: {', '.join(frequent_bigrams[:3])}")
        
        return patterns
    
    def _extract_structural_patterns(self, texts: List[str], combined_text: str) -> List[str]:
        """Extract structural and formatting patterns."""
        patterns = []
        
        # Check for common structural elements
        structure_patterns = {
            'questions': r'\?',
            'lists': r'^\s*[-*•]\s',
            'code_blocks': r'```|`.*`',
            'urls': r'https?://\S+',
            'numbers': r'\b\d+\.?\d*\b',
            'parentheses': r'\([^)]+\)',
            'quotes': r'"[^"]*"'
        }
        
        for pattern_name, regex in structure_patterns.items():
            matches = len(re.findall(regex, combined_text, re.MULTILINE))
            if matches >= 2:
                patterns.append(f"{pattern_name.replace('_', ' ').title()}: {matches} occurrences")
        
        return patterns
    
    def _extract_linguistic_markers(self, texts: List[str], combined_text: str) -> List[str]:
        """Extract linguistic markers and discourse patterns."""
        patterns = []
        
        # Linguistic markers
        markers = {
            'modal_verbs': r'\b(can|could|may|might|must|should|will|would)\b',
            'transition_words': r'\b(however|therefore|moreover|furthermore|although)\b',
            'temporal_markers': r'\b(before|after|during|while|when|then)\b',
            'causal_markers': r'\b(because|since|due to|as a result)\b'
        }
        
        for marker_type, regex in markers.items():
            matches = len(re.findall(regex, combined_text, re.IGNORECASE))
            if matches >= 1:
                patterns.append(f"{marker_type.replace('_', ' ').title()}: {matches} instances")
        
        return patterns
    
    def _extract_domain_terminology(self, texts: List[str], combined_text: str) -> List[str]:
        """Extract domain-specific terminology."""
        patterns = []
        
        # Domain-specific patterns
        domain_patterns = {
            'technical': r'\b[A-Z]{2,}|API|HTTP|JSON|XML|SQL\b',
            'academic': r'\b(research|study|analysis|hypothesis|methodology)\b',
            'business': r'\b(revenue|profit|market|customer|strategy)\b',
            'medical': r'\b(patient|diagnosis|treatment|clinical|therapy)\b'
        }
        
        for domain, regex in domain_patterns.items():
            matches = len(re.findall(regex, combined_text, re.IGNORECASE))
            if matches >= 1:
                patterns.append(f"{domain.title()} terminology: {matches} terms")
        
        return patterns
    
    def _is_too_similar(self, text: str, existing_texts: set) -> bool:
        """Check if text is too similar to existing selections."""
        text_lower = text.lower()
        text_words = set(text_lower.split())
        
        for existing in existing_texts:
            existing_words = set(existing.split())
            
            # Calculate Jaccard similarity
            intersection = len(text_words & existing_words)
            union = len(text_words | existing_words)
            
            if union > 0:
                similarity = intersection / union
                if similarity > self.MIN_SAMPLE_DIVERSITY:
                    return True
        
        return False
    
    def _infer_complexity_from_context(self, context: FeatureContext) -> FeatureComplexity:
        """Infer complexity level from context indicators."""
        # Use complexity indicators if available
        complexity_level = context.complexity_indicators.get('complexity_level')
        if complexity_level:
            try:
                return FeatureComplexity(complexity_level)
            except ValueError:
                pass
        
        # Fallback to heuristic-based inference
        keyword_count = len(context.pattern_keywords)
        pattern_count = len(context.extracted_patterns)
        theme_count = len(context.semantic_themes)
        
        complexity_score = keyword_count + pattern_count + theme_count
        
        if complexity_score >= 15:
            return FeatureComplexity.COMPLEX
        elif complexity_score >= 8:
            return FeatureComplexity.MEDIUM
        else:
            return FeatureComplexity.SIMPLE
    
    def _prepare_template_variables(self, context: FeatureContext, 
                                  complexity: FeatureComplexity) -> Dict[str, str]:
        """Prepare variables for prompt template formatting."""
        vars_dict = {
            'feature_id': str(context.feature_id),
            'coherence': context.coherence_score,
            'category': context.pattern_category.value,
            'keywords': ', '.join(context.pattern_keywords[:8]),
            'examples': self._format_examples(context.representative_samples),
            'frequency': context.statistical_summary.get('frequency', 0.0)
        }
        
        # Add complexity-specific variables
        if complexity == FeatureComplexity.MEDIUM:
            vars_dict.update({
                'complexity_indicators': self._format_complexity_indicators(context),
                'pattern_analysis': self._format_pattern_analysis(context),
                'statistical_summary': self._format_statistical_summary(context)
            })
        elif complexity == FeatureComplexity.COMPLEX:
            vars_dict.update({
                'keywords_and_themes': self._format_keywords_and_themes(context),
                'advanced_patterns': self._format_advanced_patterns(context),
                'contextual_factors': self._format_contextual_factors(context),
                'statistical_summary': self._format_detailed_statistical_summary(context)
            })
        
        return vars_dict
    
    def _format_examples(self, samples: List[str]) -> str:
        """Format activation examples for prompt inclusion."""
        if not samples:
            return "No activation examples available."
        
        formatted = []
        for i, sample in enumerate(samples[:6], 1):
            truncated = self._truncate_text(sample, self.MAX_TEXT_LENGTH)
            formatted.append(f"{i}. \"{truncated}\"")
        
        return '\n'.join(formatted)
    
    def _format_complexity_indicators(self, context: FeatureContext) -> str:
        """Format complexity indicators for display."""
        indicators = context.complexity_indicators
        items = []
        
        for key, value in indicators.items():
            if isinstance(value, (int, float)):
                items.append(f"{key.replace('_', ' ').title()}: {value:.3f}")
            else:
                items.append(f"{key.replace('_', ' ').title()}: {value}")
        
        return '; '.join(items[:5])
    
    def _format_pattern_analysis(self, context: FeatureContext) -> str:
        """Format pattern analysis for medium complexity prompts."""
        if not context.extracted_patterns:
            return "No specific patterns identified."
        
        return '\n'.join([f"• {pattern}" for pattern in context.extracted_patterns[:5]])
    
    def _format_statistical_summary(self, context: FeatureContext) -> str:
        """Format statistical summary for prompts."""
        stats = context.statistical_summary
        summary_items = []
        
        key_stats = ['frequency', 'activation_strength_mean', 'text_length_mean', 'keyword_count']
        for stat in key_stats:
            if stat in stats:
                value = stats[stat]
                if isinstance(value, float):
                    summary_items.append(f"{stat.replace('_', ' ').title()}: {value:.4f}")
                else:
                    summary_items.append(f"{stat.replace('_', ' ').title()}: {value}")
        
        return '; '.join(summary_items)
    
    def _format_keywords_and_themes(self, context: FeatureContext) -> str:
        """Format keywords and themes for complex prompts."""
        sections = []
        
        if context.pattern_keywords:
            sections.append(f"Keywords: {', '.join(context.pattern_keywords[:10])}")
        
        if context.semantic_themes:
            sections.append(f"Themes: {', '.join(context.semantic_themes)}")
        
        return '\n'.join(sections)
    
    def _format_advanced_patterns(self, context: FeatureContext) -> str:
        """Format advanced patterns for complex analysis."""
        if not context.extracted_patterns:
            return "No advanced patterns detected."
        
        return '\n'.join([f"• {pattern}" for pattern in context.extracted_patterns])
    
    def _format_contextual_factors(self, context: FeatureContext) -> str:
        """Format contextual factors for complex prompts."""
        factors = []
        
        # Quality indicators
        quality = context.quality_indicators
        if quality:
            quality_items = [f"{k.replace('_', ' ').title()}: {v}" for k, v in quality.items()]
            factors.append(f"Quality: {'; '.join(quality_items)}")
        
        # Activation context
        if context.activation_context:
            ctx_info = context.activation_context
            if 'context_types' in ctx_info:
                factors.append(f"Context Types: {', '.join(ctx_info['context_types'][:3])}")
        
        return '\n'.join(factors) if factors else "Limited contextual information available."
    
    def _format_detailed_statistical_summary(self, context: FeatureContext) -> str:
        """Format detailed statistical summary for complex prompts."""
        stats = context.statistical_summary
        
        sections = []
        
        # Activation statistics
        activation_stats = []
        for key in ['activation_strength_mean', 'activation_strength_std', 'activation_strength_max']:
            if key in stats:
                activation_stats.append(f"{key.replace('_', ' ').title()}: {stats[key]:.4f}")
        
        if activation_stats:
            sections.append(f"Activation Stats: {'; '.join(activation_stats)}")
        
        # Text statistics
        text_stats = []
        for key in ['text_length_mean', 'text_length_std', 'vocabulary_richness']:
            if key in stats:
                text_stats.append(f"{key.replace('_', ' ').title()}: {stats[key]:.3f}")
        
        if text_stats:
            sections.append(f"Text Stats: {'; '.join(text_stats)}")
        
        # General statistics
        general_stats = []
        for key in ['frequency', 'keyword_count', 'activation_count']:
            if key in stats:
                general_stats.append(f"{key.replace('_', ' ').title()}: {stats[key]}")
        
        if general_stats:
            sections.append(f"General: {'; '.join(general_stats)}")
        
        return '\n'.join(sections) if sections else "Limited statistical data available."
    
    def _optimize_for_model(self, prompt: str, model_name: str) -> Tuple[str, Dict[str, str]]:
        """Optimize prompt for specific model characteristics."""
        model_formatting = {
            'model_name': model_name,
            'token_limit': self.MODEL_TOKEN_LIMITS.get(model_name, 8192),
            'optimization_applied': []
        }
        
        optimized_prompt = prompt
        
        # Model-specific optimizations
        if "llama" in model_name.lower():
            # Llama models prefer more structured prompts
            if not prompt.startswith("### Task:"):
                optimized_prompt = f"### Task: Feature Analysis\n\n{prompt}\n\n### Response:"
                model_formatting['optimization_applied'].append('llama_structure')
        
        elif "gpt" in model_name.lower():
            # GPT models work well with role-based prompts
            if not prompt.startswith("You are"):
                optimized_prompt = f"You are an expert AI researcher analyzing neural network features.\n\n{prompt}"
                model_formatting['optimization_applied'].append('gpt_role')
        
        elif "claude" in model_name.lower():
            # Claude prefers clear, direct instructions
            if not "Please provide" in prompt:
                optimized_prompt = f"{prompt}\n\nPlease provide a detailed, structured analysis."
                model_formatting['optimization_applied'].append('claude_directive')
        
        # Token count optimization
        token_estimate = self._estimate_token_count(optimized_prompt)
        token_limit = model_formatting['token_limit']
        
        if token_estimate > token_limit * 0.8:  # Leave room for response
            # Compress the prompt
            optimized_prompt = self._compress_prompt(optimized_prompt, int(token_limit * 0.6))
            model_formatting['optimization_applied'].append('token_compression')
        
        return optimized_prompt, model_formatting
    
    def _estimate_token_count(self, text: str) -> int:
        """Estimate token count for text."""
        # Simple estimation: ~0.75 tokens per character for English text
        return int(len(text) * self.TOKEN_ESTIMATE_RATIO)
    
    def _determine_context_priority(self, context: FeatureContext, 
                                  complexity: FeatureComplexity) -> List[str]:
        """Determine context priority order for prompt optimization."""
        priority_order = ['coherence_score', 'pattern_keywords', 'representative_samples']
        
        if complexity == FeatureComplexity.MEDIUM:
            priority_order.extend(['extracted_patterns', 'statistical_summary'])
        elif complexity == FeatureComplexity.COMPLEX:
            priority_order.extend(['semantic_themes', 'complexity_indicators', 
                                 'quality_indicators', 'activation_context'])
        
        return priority_order
    
    def _generate_response_format(self, complexity: FeatureComplexity) -> str:
        """Generate expected response format specification."""
        base_format = """
Please structure your response as follows:

**Feature Summary**: One sentence describing what this feature detects.

**Detailed Explanation**: 2-3 paragraphs explaining the pattern and its significance.

**Technical Insights**: How this feature contributes to the model's capabilities.

**Implications**: Potential impacts on model behavior and applications.
"""
        
        if complexity == FeatureComplexity.COMPLEX:
            base_format += """
**Safety Considerations**: Any potential risks or bias concerns.

**Research Value**: Significance for AI interpretability research.
"""
        
        return base_format.strip()
    
    def _create_fallback_prompt(self, context: FeatureContext) -> str:
        """Create a simple fallback prompt when template formatting fails."""
        return f"""
Analyze AI Feature {context.feature_id}:

Coherence Score: {context.coherence_score:.3f}
Category: {context.pattern_category.value}
Keywords: {', '.join(context.pattern_keywords[:5])}

Top Activation Examples:
{self._format_examples(context.representative_samples[:3])}

Explain what pattern this feature detects and why it's significant for the AI model.
"""
    
    def _create_compressed_context(self, context: FeatureContext) -> str:
        """Create compressed context for token-limited scenarios."""
        return f"""
Feature {context.feature_id} (coherence: {context.coherence_score:.2f})
Category: {context.pattern_category.value}
Key patterns: {', '.join(context.pattern_keywords[:3])}
Top example: "{self._truncate_text(context.representative_samples[0] if context.representative_samples else '', 100)}"
"""
    
    def _compress_prompt(self, prompt: str, target_chars: int) -> str:
        """Compress prompt to fit within character/token limits."""
        if len(prompt) <= target_chars:
            return prompt
        
        # Split into sections and prioritize
        lines = prompt.split('\n')
        essential_lines = []
        optional_lines = []
        
        for line in lines:
            if any(keyword in line.lower() for keyword in 
                  ['coherence', 'feature', 'examples', 'explain', 'analyze']):
                essential_lines.append(line)
            else:
                optional_lines.append(line)
        
        # Rebuild with essential content first
        compressed = '\n'.join(essential_lines)
        
        # Add optional content if space allows
        for line in optional_lines:
            test_content = compressed + '\n' + line
            if len(test_content) <= target_chars:
                compressed = test_content
            else:
                break
        
        return compressed
    
    def _truncate_text(self, text: str, max_length: int) -> str:
        """Truncate text to maximum length with ellipsis."""
        if len(text) <= max_length:
            return text
        
        # Try to truncate at word boundary
        truncated = text[:max_length-3]
        last_space = truncated.rfind(' ')
        
        if last_space > max_length * 0.8:  # If we can preserve most of the text
            return truncated[:last_space] + "..."
        else:
            return truncated + "..."'