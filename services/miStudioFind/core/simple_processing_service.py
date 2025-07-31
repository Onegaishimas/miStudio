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
from typing import Dict, Any, Optional, List  # Fixed: Added List import

from core.input_manager import InputManager
from core.feature_analyzer import FeatureAnalyzer

logger = logging.getLogger(__name__)


class SimpleProcessingService:
    """Simple processing service for feature analysis jobs with file persistence."""
    
    def __init__(self, data_path: str, enhanced_persistence=None):
        """Initialize processing service."""
        self.data_path = data_path
        # Ensure results go to /data/results/find
        self.results_path = Path(data_path) / "results" / "find"
        self.results_path.mkdir(parents=True, exist_ok=True)
        
        self.input_manager = InputManager(data_path)
        self.enhanced_persistence = enhanced_persistence
        self.active_jobs: Dict[str, Dict[str, Any]] = {}
        self.completed_jobs: Dict[str, Dict[str, Any]] = {}
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        self.logger.info(f"SimpleProcessingService initialized with results path: {self.results_path}")
    
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
            "error": None,
            "results_path": str(self.results_path / job_id)  # Specific results path for this job
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
            
            # Save results to files - ensure they go to /data/results/find
            await self._save_results(job_id, job_info, results)
            
            # Move to completed jobs
            self.completed_jobs[job_id] = self.active_jobs.pop(job_id)
            
            self.logger.info(f"✅ Job {job_id} completed successfully")
            
        except Exception as e:
            self.logger.error(f"❌ Job {job_id} failed: {str(e)}")
            job_info["status"] = "failed"
            job_info["error"] = str(e)
            job_info["completion_time"] = time.time()
            
            # Move to completed jobs even if failed
            self.completed_jobs[job_id] = self.active_jobs.pop(job_id)
    
    async def _save_results(self, job_id: str, job_info: Dict[str, Any], results: Dict[str, Any]):
        """Save results to /data/results/find directory."""
        try:
            # Create job-specific directory in /data/results/find
            job_results_dir = self.results_path / job_id
            job_results_dir.mkdir(parents=True, exist_ok=True)
            
            # Create comprehensive results structure
            file_results = {
                "job_id": job_id,
                "source_job_id": job_info["source_job_id"],
                "status": "completed",
                "results": results,
                "processing_time": job_info["processing_time"],
                "timestamp": datetime.now().isoformat(),
                "metadata": {
                    "total_features": len(results),
                    "source_job_id": job_info["source_job_id"],
                    "top_k": job_info["top_k"],
                    "analysis_type": "feature_analysis",
                    "service": "miStudioFind",
                    "version": "1.0.0"
                }
            }
            
            # Save as JSON
            json_path = job_results_dir / "analysis_results.json"
            with open(json_path, 'w') as f:
                json.dump(file_results, f, indent=2, default=str)
            
            # Save job info
            job_info_path = job_results_dir / "job_info.json"
            with open(job_info_path, 'w') as f:
                json.dump(job_info, f, indent=2, default=str)
            
            # Update job info with saved file paths
            job_info["saved_files"] = {
                "results_json": str(json_path),
                "job_info": str(job_info_path),
                "results_directory": str(job_results_dir)
            }
            
            # Use enhanced persistence if available
            if self.enhanced_persistence:
                try:
                    saved_files = self.enhanced_persistence.save_comprehensive_results(job_id, file_results)
                    job_info["enhanced_persistence_files"] = saved_files
                    self.logger.info(f"✅ Enhanced persistence saved files: {saved_files}")
                except Exception as e:
                    self.logger.warning(f"Enhanced persistence failed but basic save succeeded: {e}")
            
            self.logger.info(f"✅ Results saved to {job_results_dir}")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to save results for job {job_id}: {e}")
            raise
    
    def get_job_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get job status."""
        if job_id in self.active_jobs:
            return self.active_jobs[job_id]
        elif job_id in self.completed_jobs:
            return self.completed_jobs[job_id]
        else:
            return None
    
    def get_job_results(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get job results."""
        job_info = self.get_job_status(job_id)
        if job_info and job_info.get("status") == "completed":
            return job_info.get("results")
        return None
    
    def list_jobs(self) -> List[Dict[str, Any]]:  # Fixed: Now List is properly imported
        """List all jobs."""
        all_jobs = []
        
        # Add active jobs
        for job_id, job_info in self.active_jobs.items():
            all_jobs.append({
                "job_id": job_id,
                "status": job_info["status"],
                "source_job_id": job_info["source_job_id"],
                "start_time": job_info["start_time"],
                "progress": job_info["progress"]
            })
        
        # Add completed jobs
        for job_id, job_info in self.completed_jobs.items():
            all_jobs.append({
                "job_id": job_id,
                "status": job_info["status"],
                "source_job_id": job_info["source_job_id"],
                "start_time": job_info["start_time"],
                "completion_time": job_info.get("completion_time"),
                "processing_time": job_info.get("processing_time"),
                "results_available": job_info.get("results") is not None
            })
        
        # Sort by start time (newest first)
        all_jobs.sort(key=lambda x: x["start_time"], reverse=True)
        return all_jobs
    
    def cancel_job(self, job_id: str) -> bool:
        """Cancel a running job."""
        if job_id in self.active_jobs:
            job_info = self.active_jobs[job_id]
            if job_info["status"] in ["queued", "running"]:
                job_info["status"] = "cancelled"
                job_info["completion_time"] = time.time()
                self.completed_jobs[job_id] = self.active_jobs.pop(job_id)
                self.logger.info(f"Job {job_id} cancelled")
                return True
        return False
