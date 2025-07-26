# core/processing_service.py
"""
Job orchestration and processing management for miStudioFind service.

This module orchestrates the complete feature analysis pipeline,
manages background jobs, and coordinates between all components.
"""

import asyncio
import logging
import time
import uuid
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

from models.analysis_models import ProcessingJob, ActivationData
from models.api_models import FindRequest, JobStatus
from core.input_manager import InputManager, InputValidationError
from core.feature_analyzer import FeatureAnalyzer
from core.pattern_discovery import PatternDiscovery
from core.result_manager import ResultManager
from config.find_config import config

logger = logging.getLogger(__name__)


class ProcessingError(Exception):
    """Exception raised during feature analysis processing."""

    pass


class ProcessingService:
    """Orchestrates feature analysis jobs and manages the processing pipeline."""

    def __init__(self, data_path: str = None):
        """
        Initialize ProcessingService with all required components.

        Args:
            data_path: Base path for data storage
        """
        self.data_path = data_path or config.data_path
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

        # Initialize components
        self.input_manager = InputManager(self.data_path)
        self.feature_analyzer = FeatureAnalyzer()
        self.pattern_discovery = PatternDiscovery()
        self.result_manager = ResultManager(self.data_path)

        # Job tracking
        self.active_jobs: Dict[str, ProcessingJob] = {}
        self.job_history: Dict[str, ProcessingJob] = {}

        # Threading for background processing
        self.executor = ThreadPoolExecutor(max_workers=config.max_concurrent_jobs)

        self.logger.info("ProcessingService initialized successfully")

    def generate_job_id(self) -> str:
        """Generate unique job identifier."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = str(uuid.uuid4())[:8]
        return f"find_{timestamp}_{unique_id}"

    async def start_analysis_job(self, request: FindRequest) -> str:
        """
        Start a new feature analysis job.

        Args:
            request: Analysis request parameters

        Returns:
            Unique job identifier

        Raises:
            ProcessingError: If job cannot be started
        """
        job_id = self.generate_job_id()

        self.logger.info(
            f"Starting analysis job {job_id} for source job {request.source_job_id}"
        )

        # Validate input availability first
        try:
            self.input_manager.discover_input_files(request.source_job_id)
        except InputValidationError as e:
            raise ProcessingError(
                f"Invalid input for source job {request.source_job_id}: {e}"
            )

        # Create job tracking entry
        job = ProcessingJob(
            job_id=job_id,
            source_job_id=request.source_job_id,
            config=request.model_dump(),
            status=JobStatus.QUEUED.value,
            start_time=time.time(),
            progress={
                "features_processed": 0,
                "total_features": 0,
                "current_feature": 0,
                "last_updated": time.time(),
                "estimated_time_remaining": None,
            },
        )

        self.active_jobs[job_id] = job

        # Submit job for background processing
        self.executor.submit(self._execute_analysis_job, job_id)

        self.logger.info(f"Analysis job {job_id} queued successfully")
        return job_id

    def _execute_analysis_job(self, job_id: str) -> None:
        """
        Execute analysis job in background thread.

        Args:
            job_id: Job identifier to execute
        """
        job = self.active_jobs.get(job_id)
        if not job:
            self.logger.error(f"Job {job_id} not found in active jobs")
            return

        try:
            self.logger.info(f"Executing analysis job {job_id}")
            job.status = JobStatus.RUNNING.value

            # Step 1: Load and validate inputs
            self.logger.info(f"Job {job_id}: Loading inputs")
            metadata, sae_model, activation_data = self.input_manager.load_all_inputs(
                job.source_job_id
            )

            processing_metadata = self.input_manager.get_processing_metadata(metadata)
            job.progress["total_features"] = activation_data.feature_count

            # Step 2: Analyze all features
            self.logger.info(
                f"Job {job_id}: Analyzing {activation_data.feature_count} features"
            )
            feature_ids = list(range(activation_data.feature_count))

            def progress_callback(processed: int, total: int, current_feature: int):
                job.update_progress(processed, total, current_feature)

            # Configure analyzer with job-specific settings
            analyzer = FeatureAnalyzer(
                top_k=job.config.get("top_k", config.top_k_selections),
                memory_optimization=config.memory_optimization,
            )

            analysis_results = analyzer.analyze_batch_features(
                feature_ids, activation_data, progress_callback
            )

            if not analysis_results:
                raise ProcessingError("No features were successfully analyzed")

            # Step 3: Advanced pattern discovery and quality assessment
            self.logger.info(f"Job {job_id}: Running pattern discovery")
            enhanced_results = self._enhance_with_pattern_discovery(analysis_results)

            # Step 4: Structure and organize results
            self.logger.info(f"Job {job_id}: Structuring results")
            structured_results = self.result_manager.structure_feature_results(
                enhanced_results, processing_metadata
            )

            # Step 5: Generate summary statistics
            summary_stats = self.result_manager.generate_summary_statistics(
                enhanced_results
            )

            # Step 6: Create human-readable preview
            feature_preview = self.result_manager.create_feature_preview(
                enhanced_results
            )

            # Step 7: Validate output quality
            quality_validation = self.result_manager.validate_output_quality(
                enhanced_results
            )

            if not quality_validation["valid"]:
                self.logger.warning(f"Job {job_id}: Quality validation issues detected")

            # Step 8: Save all artifacts
            self.logger.info(f"Job {job_id}: Saving artifacts")
            saved_files = self.result_manager.save_analysis_artifacts(
                job_id,
                structured_results,
                summary_stats,
                feature_preview,
                processing_metadata,
            )

            # Step 9: Prepare for explain service
            explain_readiness = self.result_manager.prepare_for_explain_service(
                job_id, enhanced_results, saved_files
            )

            # Complete job successfully
            job.status = JobStatus.COMPLETED.value
            job.results = enhanced_results
            job.progress["features_processed"] = len(enhanced_results)
            job.progress["estimated_time_remaining"] = 0

            # Store completion metadata
            completion_metadata = {
                "structured_results": structured_results,
                "summary_statistics": summary_stats,
                "feature_preview": [p.model_dump() for p in feature_preview],
                "quality_validation": quality_validation,
                "saved_files": saved_files,
                "explain_readiness": explain_readiness,
                "total_processing_time": time.time() - job.start_time,
            }

            # Move to job history
            job.results = completion_metadata
            self.job_history[job_id] = job
            if job_id in self.active_jobs:
                del self.active_jobs[job_id]

            self.logger.info(
                f"Job {job_id} completed successfully in {completion_metadata['total_processing_time']:.1f}s"
            )

        except Exception as e:
            self.logger.error(f"Job {job_id} failed: {str(e)}", exc_info=True)
            job.status = JobStatus.FAILED.value
            job.error_message = str(e)

            # Move failed job to history
            self.job_history[job_id] = job
            if job_id in self.active_jobs:
                del self.active_jobs[job_id]

    def _enhance_with_pattern_discovery(self, results: List) -> List:
        """
        Enhance analysis results with advanced pattern discovery.

        Args:
            results: List of basic feature analysis results

        Returns:
            List of enhanced results with pattern discovery
        """
        enhanced_results = []

        for result in results:
            try:
                # Run advanced pattern discovery
                quality_assessment = self.pattern_discovery.validate_feature_quality(
                    result
                )

                # Update result with enhanced information
                result.coherence_score = quality_assessment["enhanced_coherence_score"]
                result.quality_level = quality_assessment["quality_classification"]

                # Add pattern discovery metadata
                if hasattr(result, "pattern_metadata"):
                    result.pattern_metadata = quality_assessment["pattern_signature"]
                else:
                    # Store in activation_statistics for now
                    result.activation_statistics["pattern_metadata"] = (
                        quality_assessment["pattern_signature"]
                    )
                    result.activation_statistics["outlier_count"] = quality_assessment[
                        "outlier_count"
                    ]
                    result.activation_statistics["diversity_score"] = (
                        quality_assessment["diversity_score"]
                    )

                enhanced_results.append(result)

            except Exception as e:
                self.logger.warning(
                    f"Pattern discovery failed for feature {result.feature_id}: {e}"
                )
                # Include original result even if enhancement fails
                enhanced_results.append(result)

        return enhanced_results

    def get_job_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """
        Get current status of a processing job.

        Args:
            job_id: Job identifier

        Returns:
            Job status information or None if job not found
        """
        # Check active jobs first
        if job_id in self.active_jobs:
            job = self.active_jobs[job_id]
            return self._format_job_status(job)

        # Check job history
        if job_id in self.job_history:
            job = self.job_history[job_id]
            return self._format_job_status(job)

        return None

    def _format_job_status(self, job: ProcessingJob) -> Dict[str, Any]:
        """Format job for status response."""
        status_info = {
            "job_id": job.job_id,
            "source_job_id": job.source_job_id,
            "status": job.status,
            "start_time": datetime.fromtimestamp(job.start_time).isoformat(),
            "progress": job.progress.copy(),
            "message": self._generate_status_message(job),
        }

        if job.error_message:
            status_info["error_message"] = job.error_message

        if job.status == JobStatus.COMPLETED.value and job.results:
            # Add summary information for completed jobs
            if isinstance(job.results, dict):
                status_info["completion_summary"] = {
                    "features_analyzed": job.results.get("summary_statistics", {}).get(
                        "total_features", 0
                    ),
                    "high_quality_features": job.results.get("summary_statistics", {})
                    .get("quality_distribution", {})
                    .get("high", 0),
                    "interpretability_rate": job.results.get(
                        "summary_statistics", {}
                    ).get("interpretability_rate", 0.0),
                    "processing_time": job.results.get("total_processing_time", 0),
                    "ready_for_explain": job.results.get("explain_readiness", {}).get(
                        "readiness_percentage", 0
                    )
                    >= 50.0,
                }

        return status_info

    def _generate_status_message(self, job: ProcessingJob) -> str:
        """Generate human-readable status message."""
        if job.status == JobStatus.QUEUED.value:
            return f"Job queued for analysis of source job {job.source_job_id}"

        elif job.status == JobStatus.RUNNING.value:
            progress = job.progress
            processed = progress.get("features_processed", 0)
            total = progress.get("total_features", 0)
            current = progress.get("current_feature", 0)

            if total > 0:
                percentage = (processed / total) * 100
                eta = progress.get("estimated_time_remaining")
                eta_str = f", ETA: {eta}s" if eta else ""
                return f"Analyzing feature {current}/{total} ({percentage:.1f}% complete{eta_str})"
            else:
                return "Initializing analysis..."

        elif job.status == JobStatus.COMPLETED.value:
            if isinstance(job.results, dict):
                features_count = job.results.get("summary_statistics", {}).get(
                    "total_features", 0
                )
                interpretable_count = job.results.get("explain_readiness", {}).get(
                    "features_ready_for_explanation", 0
                )
                return f"Analysis completed: {features_count} features analyzed, {interpretable_count} ready for explanation"
            else:
                return "Analysis completed successfully"

        elif job.status == JobStatus.FAILED.value:
            return f"Analysis failed: {job.error_message or 'Unknown error'}"

        elif job.status == JobStatus.CANCELLED.value:
            return "Analysis was cancelled"

        else:
            return f"Unknown status: {job.status}"

    def get_job_results(self, job_id: str) -> Optional[Dict[str, Any]]:
        """
        Get results for a completed job.

        Args:
            job_id: Job identifier

        Returns:
            Job results or None if job not found or not completed
        """
        job = self.job_history.get(job_id)
        if not job or job.status != JobStatus.COMPLETED.value:
            return None

        if not isinstance(job.results, dict):
            return None

        # Return comprehensive results
        return {
            "job_id": job_id,
            "source_job_id": job.source_job_id,
            "status": job.status,
            "features_analyzed": job.results.get("summary_statistics", {}).get(
                "total_features", 0
            ),
            "processing_time_seconds": job.results.get("total_processing_time", 0),
            "quality_distribution": job.results.get("summary_statistics", {}).get(
                "quality_distribution", {}
            ),
            "output_files": job.results.get("saved_files", {}),
            "ready_for_explain_service": job.results.get("explain_readiness", {}).get(
                "readiness_percentage", 0
            )
            >= 50.0,
            "explain_readiness": job.results.get("explain_readiness", {}),
            "quality_validation": job.results.get("quality_validation", {}),
            "summary_statistics": job.results.get("summary_statistics", {}),
            "completion_time": datetime.fromtimestamp(
                job.start_time + job.results.get("total_processing_time", 0)
            ).isoformat(),
        }

    def list_all_jobs(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        List all jobs (active and completed).

        Returns:
            Dictionary containing active and completed job lists
        """
        active_jobs_info = []
        for job_id, job in self.active_jobs.items():
            active_jobs_info.append(
                {
                    "job_id": job_id,
                    "source_job_id": job.source_job_id,
                    "status": job.status,
                    "start_time": datetime.fromtimestamp(job.start_time).isoformat(),
                    "progress_percentage": (
                        job.progress.get("features_processed", 0)
                        / max(1, job.progress.get("total_features", 1))
                    )
                    * 100,
                }
            )

        completed_jobs_info = []
        for job_id, job in self.job_history.items():
            job_info = {
                "job_id": job_id,
                "source_job_id": job.source_job_id,
                "status": job.status,
                "start_time": datetime.fromtimestamp(job.start_time).isoformat(),
            }

            if job.status == JobStatus.COMPLETED.value and isinstance(
                job.results, dict
            ):
                job_info.update(
                    {
                        "features_analyzed": job.results.get(
                            "summary_statistics", {}
                        ).get("total_features", 0),
                        "processing_time": job.results.get("total_processing_time", 0),
                        "interpretability_rate": job.results.get(
                            "summary_statistics", {}
                        ).get("interpretability_rate", 0.0),
                    }
                )
            elif job.status == JobStatus.FAILED.value:
                job_info["error_message"] = job.error_message

            completed_jobs_info.append(job_info)

        return {
            "active_jobs": active_jobs_info,
            "completed_jobs": completed_jobs_info,
            "total_active": len(active_jobs_info),
            "total_completed": len(completed_jobs_info),
        }

    def cancel_job(self, job_id: str) -> bool:
        """
        Cancel an active job.

        Args:
            job_id: Job identifier to cancel

        Returns:
            True if job was cancelled, False if job not found or not cancellable
        """
        if job_id not in self.active_jobs:
            return False

        job = self.active_jobs[job_id]
        if job.status not in [JobStatus.QUEUED.value, JobStatus.RUNNING.value]:
            return False

        job.status = JobStatus.CANCELLED.value
        job.error_message = "Job cancelled by user request"

        # Move to history
        self.job_history[job_id] = job
        del self.active_jobs[job_id]

        self.logger.info(f"Job {job_id} cancelled successfully")
        return True

    def get_service_health(self) -> Dict[str, Any]:
        """
        Get service health and performance information.

        Returns:
            Dictionary containing service health metrics
        """
        active_count = len(self.active_jobs)
        completed_count = len(self.job_history)

        # Calculate average processing time from recent completed jobs
        recent_jobs = list(self.job_history.values())[-10:]  # Last 10 jobs
        avg_processing_time = 0.0

        if recent_jobs:
            processing_times = []
            for job in recent_jobs:
                if (
                    job.status == JobStatus.COMPLETED.value
                    and isinstance(job.results, dict)
                    and "total_processing_time" in job.results
                ):
                    processing_times.append(job.results["total_processing_time"])

            if processing_times:
                avg_processing_time = sum(processing_times) / len(processing_times)

        return {
            "service": "miStudioFind",
            "version": config.service_version,
            "status": "healthy",
            "active_jobs": active_count,
            "completed_jobs": completed_count,
            "max_concurrent_jobs": config.max_concurrent_jobs,
            "capacity_available": config.max_concurrent_jobs - active_count,
            "average_processing_time_seconds": avg_processing_time,
            "data_path": self.data_path,
            "configuration": {
                "top_k_selections": config.top_k_selections,
                "coherence_threshold": config.coherence_threshold,
                "memory_optimization": config.memory_optimization,
                "processing_timeout_minutes": config.processing_timeout_minutes,
            },
            "timestamp": datetime.now().isoformat(),
        }

    def cleanup_old_jobs(self, max_history_size: int = 100) -> int:
        """
        Clean up old job history to prevent memory buildup.

        Args:
            max_history_size: Maximum number of jobs to keep in history

        Returns:
            Number of jobs removed
        """
        if len(self.job_history) <= max_history_size:
            return 0

        # Sort by start time and keep most recent
        sorted_jobs = sorted(
            self.job_history.items(), key=lambda x: x[1].start_time, reverse=True
        )

        jobs_to_keep = dict(sorted_jobs[:max_history_size])
        removed_count = len(self.job_history) - len(jobs_to_keep)

        self.job_history = jobs_to_keep

        self.logger.info(f"Cleaned up {removed_count} old jobs from history")
        return removed_count
