# core/simple_processing_service.py
"""
Simple processing service to orchestrate feature analysis with file persistence.
"""

import asyncio
import logging
import time
import uuid
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

from core.input_manager import InputManager
from core.feature_analyzer import FeatureAnalyzer

logger = logging.getLogger(__name__)


class SimpleProcessingService:
    """Simple processing service for feature analysis jobs with file persistence."""
    
    def __init__(self, data_path: str, enhanced_persistence=None):
        """Initialize processing service."""
        self.data_path = data_path
        self.input_manager = InputManager(data_path)
        self.enhanced_persistence = enhanced_persistence
        self.active_jobs: Dict[str, Dict[str, Any]] = {}
        self.completed_jobs: Dict[str, Dict[str, Any]] = {}
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    def generate_job_id(self) -> str:
        """Generate unique job identifier."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = str(uuid.uuid4())[:8]
        return f"find_{timestamp}_{unique_id}"
    
    async def start_analysis_job(self, source_job_id: str, top_k: int = 20) -> str:
        """
        Start a new feature analysis job.
        
        Args:
            source_job_id: Source training job ID
            top_k: Number of top activations per feature
            
        Returns:
            Job ID for the new analysis job
        """
        job_id = self.generate_job_id()
        
        # Create job entry
        job_info = {
            "job_id": job_id,
            "source_job_id": source_job_id,
            "status": "queued",
            "start_time": time.time(),
            "top_k": top_k,
            "progress": {
                "features_processed": 0,
                "total_features": 0,
                "estimated_time_remaining": None
            },
            "results": None,
            "error": None
        }
        
        self.active_jobs[job_id] = job_info
        
        # Start processing in background
        asyncio.create_task(self._process_job(job_id))
        
        self.logger.info(f"Started analysis job {job_id} for source {source_job_id}")
        return job_id
    
    async def _process_job(self, job_id: str):
        """Process a feature analysis job."""
        job_info = self.active_jobs[job_id]
        
        try:
            job_info["status"] = "running"
            source_job_id = job_info["source_job_id"]
            top_k = job_info["top_k"]
            
            self.logger.info(f"Processing job {job_id}...")
            
            # Load inputs
            metadata, sae_model, activation_data = self.input_manager.load_all_inputs(source_job_id)
            
            # Update progress with total features
            total_features = activation_data["feature_count"]
            job_info["progress"]["total_features"] = total_features
            
            # Analyze features
            analyzer = FeatureAnalyzer(top_k=top_k)
            
            def progress_callback(processed, total, current):
                job_info["progress"]["features_processed"] = processed
                if processed > 0:
                    elapsed = time.time() - job_info["start_time"]
                    time_per_feature = elapsed / processed
                    remaining = (total - processed) * time_per_feature
                    job_info["progress"]["estimated_time_remaining"] = int(remaining)
            
            results = analyzer.analyze_all_features(activation_data, progress_callback)
            
            # Complete job
            job_info["status"] = "completed"
            job_info["results"] = results
            job_info["completion_time"] = time.time()
            job_info["processing_time"] = job_info["completion_time"] - job_info["start_time"]
            
            # Save results to files if enhanced persistence is available
            if self.enhanced_persistence:
                try:
                    # Create results structure for file saving
                    file_results = {
                        "job_id": job_id,
                        "source_job_id": source_job_id,
                        "status": "completed",
                        "results": results,
                        "processing_time": job_info["processing_time"],
                        "timestamp": datetime.now().isoformat(),
                        "metadata": {
                            "total_features": len(results),
                            "source_job_id": source_job_id,
                            "top_k": top_k,
                            "analysis_type": "feature_analysis",
                            "service": "miStudioFind",
                            "version": "1.0.0"
                        }
                    }
                    
                    # Save using enhanced persistence (multiple formats)
                    saved_files = self.enhanced_persistence.save_comprehensive_results(job_id, file_results)
                    self.logger.info(f"✅ Saved results in multiple formats: {list(saved_files.keys())}")
                    
                    # Also save in the specific location expected by miStudioExplain
                    explain_results_dir = Path(self.data_path) / "results" / "find"
                    explain_results_dir.mkdir(parents=True, exist_ok=True)
                    
                    explain_results_file = explain_results_dir / f"{job_id}_results.json"
                    with open(explain_results_file, 'w') as f:
                        json.dump(file_results, f, indent=2)
                    
                    self.logger.info(f"✅ Saved results for miStudioExplain: {explain_results_file}")
                    
                    # Store file paths in job info for reference
                    job_info["saved_files"] = {
                        "enhanced_persistence": saved_files,
                        "explain_service_file": str(explain_results_file)
                    }
                    
                except Exception as e:
                    self.logger.warning(f"Failed to save result files: {e}")
                    # Don't fail the job if file saving fails
            
            # Move to completed jobs
            self.completed_jobs[job_id] = job_info
            del self.active_jobs[job_id]
            
            self.logger.info(f"Job {job_id} completed successfully. Processed {len(results)} features.")
            
        except Exception as e:
            self.logger.error(f"Job {job_id} failed: {e}")
            job_info["status"] = "failed"
            job_info["error"] = str(e)
            
            # Move to completed jobs even if failed
            self.completed_jobs[job_id] = job_info
            if job_id in self.active_jobs:
                del self.active_jobs[job_id]
    
    def get_job_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a job."""
        # Check active jobs
        if job_id in self.active_jobs:
            return self.active_jobs[job_id]
        
        # Check completed jobs
        if job_id in self.completed_jobs:
            return self.completed_jobs[job_id]
        
        return None
    
    def get_job_results(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get results of a completed job."""
        job_info = self.completed_jobs.get(job_id)
        if job_info and job_info["status"] == "completed":
            return job_info
        return None
    
    def list_jobs(self) -> List[Dict[str, Any]]:
        """List all jobs (active and completed)."""
        all_jobs = []
        
        # Add active jobs
        for job_id, job_info in self.active_jobs.items():
            all_jobs.append({
                "job_id": job_id,
                "status": job_info["status"],
                "source_job_id": job_info["source_job_id"],
                "created_at": datetime.fromtimestamp(job_info["start_time"]).isoformat(),
                "completed_at": None
            })
        
        # Add completed jobs
        for job_id, job_info in self.completed_jobs.items():
            completed_time = None
            if job_info.get("completion_time"):
                completed_time = datetime.fromtimestamp(job_info["completion_time"]).isoformat()
            
            all_jobs.append({
                "job_id": job_id,
                "status": job_info["status"],
                "source_job_id": job_info["source_job_id"],
                "created_at": datetime.fromtimestamp(job_info["start_time"]).isoformat(),
                "completed_at": completed_time
            })
        
        return all_jobs
    
    async def cancel_job(self, job_id: str) -> bool:
        """Cancel an active job."""
        if job_id in self.active_jobs:
            job_info = self.active_jobs[job_id]
            job_info["status"] = "cancelled"
            job_info["error"] = "Job cancelled by user request"
            
            # Move to completed jobs
            self.completed_jobs[job_id] = job_info
            del self.active_jobs[job_id]
            
            self.logger.info(f"Job {job_id} cancelled successfully")
            return True
        
        return False
    
    def get_saved_files(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get information about saved files for a job."""
        job_info = self.completed_jobs.get(job_id)
        if job_info and job_info.get("saved_files"):
            return job_info["saved_files"]
        return None