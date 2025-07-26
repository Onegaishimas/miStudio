"""
Metrics Collection for miStudioExplain Service
"""

import time
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from contextlib import contextmanager

logger = logging.getLogger(__name__)


@dataclass
class ProcessingMetrics:
    """Metrics for explanation processing"""
    explanations_generated: int = 0
    total_processing_time: float = 0.0
    average_explanation_time: float = 0.0
    quality_scores: List[float] = field(default_factory=list)
    model_usage_count: Dict[str, int] = field(default_factory=dict)
    gpu_utilization_history: List[Dict[str, float]] = field(default_factory=list)
    error_count: int = 0
    cache_hit_rate: float = 0.0


@dataclass
class ModelMetrics:
    """Metrics for individual models"""
    model_name: str
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    average_response_time: float = 0.0
    total_tokens_generated: int = 0
    quality_score_average: float = 0.0
    gpu_memory_peak: int = 0
    last_used: Optional[datetime] = None


class MetricsCollector:
    """Collects and manages service metrics"""
    
    def __init__(self):
        self.processing_metrics = ProcessingMetrics()
        self.model_metrics: Dict[str, ModelMetrics] = {}
        self.start_time = datetime.now()
        
    @contextmanager
    def measure_processing_time(self, operation: str):
        """Context manager to measure processing time"""
        start_time = time.time()
        try:
            yield
        finally:
            duration = time.time() - start_time
            self._record_processing_time(operation, duration)
    
    def record_explanation_generated(self, model_name: str, quality_score: float, 
                                   processing_time: float, tokens_generated: int):
        """Record successful explanation generation"""
        self.processing_metrics.explanations_generated += 1
        self.processing_metrics.total_processing_time += processing_time
        self.processing_metrics.quality_scores.append(quality_score)
        
        # Update model metrics
        if model_name not in self.model_metrics:
            self.model_metrics[model_name] = ModelMetrics(model_name)
        
        model_metrics = self.model_metrics[model_name]
        model_metrics.total_requests += 1
        model_metrics.successful_requests += 1
        model_metrics.total_tokens_generated += tokens_generated
        model_metrics.last_used = datetime.now()
        
        # Update averages
        self._update_averages()
    
    def _record_processing_time(self, operation: str, duration: float):
        """Record processing time for specific operation"""
        logger.debug(f"Operation {operation} took {duration:.2f} seconds")
    
    def _update_averages(self):
        """Update average metrics"""
        if self.processing_metrics.explanations_generated > 0:
            self.processing_metrics.average_explanation_time = (
                self.processing_metrics.total_processing_time / 
                self.processing_metrics.explanations_generated
            )

