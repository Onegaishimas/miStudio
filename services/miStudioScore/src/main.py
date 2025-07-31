# src/main.py - Updated miStudioScore with Shared Storage Integration
"""
Main FastAPI application for miStudioScore service - Integrated with shared storage.
"""

import os
import sys
import json
import logging
from typing import Dict, Any, Optional, List
from pathlib import Path

# Add the parent directory to Python path to access core modules
SERVICE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, SERVICE_ROOT)

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse, StreamingResponse
from contextlib import asynccontextmanager
from datetime import datetime
from pydantic import BaseModel, Field
import asyncio
import zipfile
import io

# Configure logging first
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Suppress noisy loggers
logging.getLogger("transformers").setLevel(logging.WARNING)
logging.getLogger("torch").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("uvicorn.access").setLevel(logging.WARNING)


# =============================================================================
# Enhanced Configuration with Shared Storage Integration
# =============================================================================

class ServiceConfig:
    """Enhanced configuration for miStudioScore with shared storage integration"""
    
    def __init__(self):
        # Primary data path - same pattern for all services
        self.data_path = os.getenv("DATA_PATH", "/data")
        self.data_path_obj = Path(self.data_path)
        self.data_path_obj.mkdir(parents=True, exist_ok=True)
        
        # Service metadata
        self.service_name = "miStudioScore"
        self.service_version = "1.0.0"
        
        # API configuration
        self.api_host = os.getenv("API_HOST", "0.0.0.0")
        self.api_port = int(os.getenv("API_PORT", "8004"))
        self.log_level = os.getenv("LOG_LEVEL", "INFO")
        
        # Processing configuration
        self.max_concurrent_jobs = int(os.getenv("MAX_CONCURRENT_JOBS", "2"))
        self.scoring_timeout = int(os.getenv("SCORING_TIMEOUT", "300"))
        
        # Enhanced directory structure
        self._ensure_directories()
        
        logger.info(f"🔧 ServiceConfig initialized:")
        logger.info(f"   Data path: {self.data_path}")
        logger.info(f"   Find results: {self.find_results_dir}")
        logger.info(f"   Explain results: {self.explain_results_dir}")
        logger.info(f"   Score results: {self.score_results_dir}")
        logger.info(f"   Service: {self.service_name} v{self.service_version}")
    
    @property
    def find_results_dir(self) -> Path:
        """Directory where miStudioFind results are stored"""
        return self.data_path_obj / "results" / "find"
    
    @property
    def explain_results_dir(self) -> Path:
        """Directory where miStudioExplain results are stored"""
        return self.data_path_obj / "results" / "explain"
    
    @property
    def score_results_dir(self) -> Path:
        """Directory where miStudioScore results are stored"""
        return self.data_path_obj / "results" / "score"
    
    @property
    def cache_dir(self) -> Path:
        """Cache directory"""
        return self.data_path_obj / "cache" / "score"
    
    @property
    def logs_dir(self) -> Path:
        """Logs directory"""
        return self.data_path_obj / "logs" / "score"
    
    def _ensure_directories(self):
        """Ensure all required directories exist"""
        directories = [
            self.find_results_dir,
            self.explain_results_dir,
            self.score_results_dir,
            self.cache_dir,
            self.logs_dir
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Ensured directory: {directory}")

# Global configuration instance
config = ServiceConfig()


# =============================================================================
# Enhanced API Models
# =============================================================================

class ScoreRequest(BaseModel):
    """Enhanced request model for scoring"""
    source_type: str = Field(..., description="Source type: 'find' or 'explain'")
    source_job_id: str = Field(..., description="Source job ID from Find or Explain service")
    scoring_config: Dict[str, Any] = Field(..., description="Scoring configuration")
    output_formats: List[str] = Field(default=["json", "csv"], description="Output formats to generate")

class ScoreParameters(BaseModel):
    """Parameters included in response"""
    source_type: str
    source_job_id: str
    scoring_jobs: List[Dict[str, Any]]
    output_formats: List[str]

class NextSteps(BaseModel):
    """Next steps information"""
    check_status: str
    get_results: str

class ScoreJobResponse(BaseModel):
    """Enhanced response model for starting scoring job"""
    job_id: str
    status: str
    message: str
    source_type: str
    source_job_id: str
    parameters: ScoreParameters
    timestamp: str
    next_steps: NextSteps

class JobProgress(BaseModel):
    """Enhanced job progress information"""
    scoring_jobs_processed: int
    total_scoring_jobs: int
    current_job: Optional[str] = None
    estimated_time_remaining: Optional[int] = None

class JobStatusResponse(BaseModel):
    """Enhanced response model for job status"""
    job_id: str
    status: str
    source_type: str
    source_job_id: str
    start_time: Optional[str] = None
    completion_time: Optional[str] = None
    processing_time: Optional[float] = None
    progress: Optional[JobProgress] = None
    error: Optional[str] = None
    results_path: Optional[str] = None

class JobResultResponse(BaseModel):
    """Enhanced response model for job results"""
    job_id: str
    status: str
    source_type: str
    source_job_id: str
    scores_added: Optional[List[str]] = None
    results_summary: Optional[Dict[str, Any]] = None
    download_links: Optional[Dict[str, str]] = None
    processing_time: Optional[float] = None

class HealthResponse(BaseModel):
    """Enhanced health check response model"""
    status: str
    service: str
    version: str
    data_path: str
    find_results_path: str
    explain_results_path: str
    score_results_path: str
    timestamp: str
    components: Dict[str, bool]
    available_scorers: List[str]

class JobSummary(BaseModel):
    """Job summary model"""
    job_id: str
    status: str
    source_type: str
    source_job_id: str
    created_at: str
    completed_at: Optional[str] = None

class JobListResponse(BaseModel):
    """Response model for job listing"""
    jobs: List[JobSummary]
    total: int


# =============================================================================
# Enhanced Service Implementation
# =============================================================================

class IntegratedScoreService:
    """Integrated service that works with Find/Explain outputs and stores in proper locations"""
    
    def __init__(self, config: ServiceConfig):
        self.config = config
        self.active_jobs: Dict[str, Dict[str, Any]] = {}
        self.completed_jobs: Dict[str, Dict[str, Any]] = {}
        
        # Initialize scorers
        self.available_scorers = []
        self._initialize_scorers()
        
        logger.info(f"IntegratedScoreService initialized")
        logger.info(f"  Find results path: {self.config.find_results_dir}")
        logger.info(f"  Explain results path: {self.config.explain_results_dir}")
        logger.info(f"  Score results path: {self.config.score_results_dir}")
        logger.info(f"  Available scorers: {self.available_scorers}")
    
    def _initialize_scorers(self):
        """Initialize available scoring modules"""
        try:
            from core.scoring.relevance_scorer import RelevanceScorer
            self.available_scorers.append("relevance_scorer")
        except ImportError:
            logger.warning("RelevanceScorer not available")
        
        try:
            from core.scoring.ablation_scorer import AblationScorer
            self.available_scorers.append("ablation_scorer")
        except ImportError:
            logger.warning("AblationScorer not available")
        
        # Add simple pattern-based scorer as fallback
        self.available_scorers.append("pattern_scorer")
    
    def discover_source_jobs(self, source_type: str) -> List[Dict[str, Any]]:
        """Discover available source jobs for scoring"""
        available_jobs = []
        
        if source_type == "find":
            source_dir = self.config.find_results_dir
            if source_dir.exists():
                for job_dir in source_dir.iterdir():
                    if job_dir.is_dir():
                        results_file = job_dir / "analysis_results.json"
                        if results_file.exists():
                            try:
                                with open(results_file, 'r') as f:
                                    data = json.load(f)
                                
                                available_jobs.append({
                                    "source_job_id": job_dir.name,
                                    "source_type": "find",
                                    "feature_count": len(data.get('results', [])),
                                    "status": "available",
                                    "results_path": str(results_file)
                                })
                            except Exception as e:
                                logger.warning(f"Could not read Find job {job_dir.name}: {e}")
            
            # Also check enhanced persistence directories
            results_base = self.config.data_path_obj / "results"
            if results_base.exists():
                for job_dir in results_base.glob("find_*"):
                    if job_dir.is_dir() and job_dir.name not in [j["source_job_id"] for j in available_jobs]:
                        results_file = job_dir / f"{job_dir.name}_complete_results.json"
                        if results_file.exists():
                            try:
                                with open(results_file, 'r') as f:
                                    data = json.load(f)
                                
                                available_jobs.append({
                                    "source_job_id": job_dir.name,
                                    "source_type": "find",
                                    "feature_count": len(data.get('results', [])),
                                    "status": "available",
                                    "results_path": str(results_file)
                                })
                            except Exception as e:
                                logger.warning(f"Could not read enhanced Find job {job_dir.name}: {e}")
        
        elif source_type == "explain":
            source_dir = self.config.explain_results_dir
            if source_dir.exists():
                for job_dir in source_dir.iterdir():
                    if job_dir.is_dir():
                        results_file = job_dir / "explanation_results.json"
                        if results_file.exists():
                            try:
                                with open(results_file, 'r') as f:
                                    data = json.load(f)
                                
                                available_jobs.append({
                                    "source_job_id": job_dir.name,
                                    "source_type": "explain",
                                    "explanation_count": len(data.get('explanations', [])),
                                    "status": "available", 
                                    "results_path": str(results_file)
                                })
                            except Exception as e:
                                logger.warning(f"Could not read Explain job {job_dir.name}: {e}")
        
        logger.info(f"Discovered {len(available_jobs)} available {source_type} jobs")
        return available_jobs
    
    def load_source_data(self, source_type: str, source_job_id: str) -> Dict[str, Any]:
        """Load source data from Find or Explain results"""
        if source_type == "find":
            # Try main results directory first
            main_path = self.config.find_results_dir / source_job_id / "analysis_results.json"
            
            if main_path.exists():
                with open(main_path, 'r') as f:
                    data = json.load(f)
                logger.info(f"Loaded Find results from main directory: {main_path}")
                return data
            
            # Try enhanced persistence directory
            enhanced_path = self.config.data_path_obj / "results" / source_job_id / f"{source_job_id}_complete_results.json"
            
            if enhanced_path.exists():
                with open(enhanced_path, 'r') as f:
                    data = json.load(f)
                logger.info(f"Loaded Find results from enhanced directory: {enhanced_path}")
                return data
        
        elif source_type == "explain":
            explain_path = self.config.explain_results_dir / source_job_id / "explanation_results.json"
            
            if explain_path.exists():
                with open(explain_path, 'r') as f:
                    data = json.load(f)
                logger.info(f"Loaded Explain results: {explain_path}")
                return data
        
        # List available jobs for error message
        available = self.discover_source_jobs(source_type)
        available_ids = [job["source_job_id"] for job in available]
        
        raise ValueError(f"{source_type.title()} job {source_job_id} not found. Available jobs: {available_ids}")
    
    async def start_scoring_job(self, request: ScoreRequest) -> str:
        """Start scoring job"""
        job_id = self.generate_job_id()
        
        try:
            # Load and validate source data
            source_data = self.load_source_data(request.source_type, request.source_job_id)
            
            # Extract features based on source type
            if request.source_type == "find":
                features = source_data.get('results', [])
                if not features:
                    raise ValueError(f"No features found in Find job {request.source_job_id}")
            elif request.source_type == "explain":
                explanations = source_data.get('explanations', [])
                if not explanations:
                    raise ValueError(f"No explanations found in Explain job {request.source_job_id}")
                # Convert explanations to feature format for scoring
                features = [
                    {
                        "feature_id": exp.get("feature_id"),
                        "coherence_score": exp.get("coherence_score", 0),
                        "pattern_keywords": exp.get("pattern_keywords", []),
                        "explanation": exp.get("explanation", ""),
                        **exp
                    }
                    for exp in explanations
                ]
            
            # Validate scoring configuration
            scoring_jobs = request.scoring_config.get("scoring_jobs", [])
            if not scoring_jobs:
                raise ValueError("No scoring jobs defined in configuration")
            
            # Create job entry
            job_info = {
                "job_id": job_id,
                "source_type": request.source_type,
                "source_job_id": request.source_job_id,
                "status": "queued",
                "start_time": datetime.now().isoformat(),
                "parameters": request.dict(),
                "source_features": features,
                "scoring_config": request.scoring_config,
                "progress": {
                    "scoring_jobs_processed": 0,
                    "total_scoring_jobs": len(scoring_jobs),
                    "current_job": None,
                    "estimated_time_remaining": None
                },
                "results": None,
                "error": None,
                "results_path": str(self.config.score_results_dir / job_id)
            }
            
            self.active_jobs[job_id] = job_info
            
            # Start processing in background
            asyncio.create_task(self._process_scoring_job(job_id))
            
            logger.info(f"Started scoring job {job_id} for {request.source_type} job {request.source_job_id}")
            logger.info(f"Processing {len(features)} features with {len(scoring_jobs)} scoring jobs")
            
            return job_id
            
        except Exception as e:
            logger.error(f"Failed to start scoring job: {e}")
            raise
    
    async def _process_scoring_job(self, job_id: str):
        """Process scoring job"""
        job_info = self.active_jobs[job_id]
        
        try:
            job_info["status"] = "running"
            logger.info(f"Processing scoring job {job_id}...")
            
            features = job_info["source_features"].copy()  # Work on a copy
            scoring_config = job_info["scoring_config"]
            scoring_jobs = scoring_config.get("scoring_jobs", [])
            
            added_scores = []
            
            # Process each scoring job
            for i, scoring_job in enumerate(scoring_jobs):
                scorer_name = scoring_job.get("scorer")
                job_name = scoring_job.get("name")
                job_params = scoring_job.get("params", {})
                
                job_info["progress"]["current_job"] = job_name
                
                # Apply scoring
                features = await self._apply_scorer(features, scorer_name, job_name, job_params)
                added_scores.append(job_name)
                
                # Update progress
                job_info["progress"]["scoring_jobs_processed"] = i + 1
                
                # Simulate processing time
                await asyncio.sleep(0.5)
            
            # Complete job
            job_info["status"] = "completed"
            job_info["completion_time"] = datetime.now().isoformat()
            job_info["results"] = {
                "features": features,
                "scores_added": added_scores,
                "total_features": len(features)
            }
            
            # Save results to files
            await self._save_scoring_results(job_id, job_info, features, added_scores)
            
            # Move to completed jobs
            self.completed_jobs[job_id] = self.active_jobs.pop(job_id)
            
            logger.info(f"✅ Scoring job {job_id} completed with {len(added_scores)} scores added")
            
        except Exception as e:
            logger.error(f"❌ Scoring job {job_id} failed: {str(e)}")
            job_info["status"] = "failed"
            job_info["error"] = str(e)
            job_info["completion_time"] = datetime.now().isoformat()
            self.completed_jobs[job_id] = self.active_jobs.pop(job_id)
    
    async def _apply_scorer(self, features: List[Dict[str, Any]], scorer_name: str, 
                           job_name: str, params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Apply a specific scorer to features"""
        try:
            if scorer_name == "relevance_scorer":
                # Simple relevance scoring based on keywords
                positive_keywords = params.get("positive_keywords", [])
                negative_keywords = params.get("negative_keywords", [])
                
                for feature in features:
                    score = self._calculate_relevance_score(feature, positive_keywords, negative_keywords)
                    feature[job_name] = score
            
            elif scorer_name == "pattern_scorer":
                # Pattern-based scoring
                for feature in features:
                    score = self._calculate_pattern_score(feature, params)
                    feature[job_name] = score
            
            elif scorer_name == "ablation_scorer":
                # Placeholder for ablation scoring (would require model integration)
                for feature in features:
                    # Simulate ablation score based on coherence
                    coherence = feature.get("coherence_score", 0)
                    score = min(1.0, coherence * 1.2)  # Simple simulation
                    feature[job_name] = score
            
            else:
                logger.warning(f"Unknown scorer: {scorer_name}")
                # Add default score
                for feature in features:
                    feature[job_name] = 0.5
            
            logger.info(f"Applied scorer '{scorer_name}' as job '{job_name}'")
            return features
            
        except Exception as e:
            logger.error(f"Error applying scorer {scorer_name}: {e}")
            # Add error score
            for feature in features:
                feature[job_name] = 0.0
            return features
    
    def _calculate_relevance_score(self, feature: Dict[str, Any], 
                                 positive_keywords: List[str], 
                                 negative_keywords: List[str]) -> float:
        """Calculate relevance score based on keywords"""
        text_content = ""
        
        # Gather text content from feature
        if "pattern_keywords" in feature:
            text_content += " ".join(str(k) for k in feature["pattern_keywords"])
        
        if "explanation" in feature:
            text_content += " " + str(feature["explanation"])
        
        if "top_activations" in feature:
            for activation in feature["top_activations"][:3]:
                if isinstance(activation, dict) and "text" in activation:
                    text_content += " " + str(activation["text"])
        
        text_content = text_content.lower()
        
        # Calculate score
        positive_score = sum(1 for keyword in positive_keywords if keyword.lower() in text_content)
        negative_score = sum(1 for keyword in negative_keywords if keyword.lower() in text_content)
        
        # Normalize score between 0 and 1
        max_positive = len(positive_keywords) if positive_keywords else 1
        max_negative = len(negative_keywords) if negative_keywords else 1
        
        final_score = (positive_score / max_positive) - (negative_score / max_negative)
        return max(0.0, min(1.0, final_score))
    
    def _calculate_pattern_score(self, feature: Dict[str, Any], params: Dict[str, Any]) -> float:
        """Calculate pattern-based score"""
        coherence = feature.get("coherence_score", 0)
        keyword_count = len(feature.get("pattern_keywords", []))
        
        # Simple pattern scoring
        pattern_score = (coherence * 0.7) + (min(keyword_count / 10, 1.0) * 0.3)
        return min(1.0, pattern_score)
    
    async def _save_scoring_results(self, job_id: str, job_info: Dict[str, Any], 
                                  features: List[Dict[str, Any]], added_scores: List[str]):
        """Save scoring results to shared storage"""
        try:
            # Create job-specific directory in /data/results/score
            job_results_dir = self.config.score_results_dir / job_id
            job_results_dir.mkdir(parents=True, exist_ok=True)
            
            # Calculate processing time
            start_time = datetime.fromisoformat(job_info["start_time"])
            end_time = datetime.fromisoformat(job_info["completion_time"])
            processing_time = (end_time - start_time).total_seconds()
            
            # Create comprehensive results structure
            results_data = {
                "job_id": job_id,
                "source_type": job_info["source_type"],
                "source_job_id": job_info["source_job_id"],
                "status": "completed",
                "start_time": job_info["start_time"],
                "completion_time": job_info["completion_time"],
                "processing_time": processing_time,
                "parameters": job_info["parameters"],
                "features": features,
                "scores_added": added_scores,
                "summary": {
                    "total_features": len(features),
                    "scores_added": added_scores,
                    "source_type": job_info["source_type"],
                    "source_job_id": job_info["source_job_id"],
                    "service": "miStudioScore",
                    "version": self.config.service_version
                },
                "timestamp": datetime.now().isoformat()
            }
            
            # Save main results JSON
            results_path = job_results_dir / "scoring_results.json"
            with open(results_path, 'w') as f:
                json.dump(results_data, f, indent=2, default=str)
            
            # Save job info
            job_info_path = job_results_dir / "job_info.json"
            with open(job_info_path, 'w') as f:
                json.dump(job_info, f, indent=2, default=str)
            
            # Save features in CSV format
            csv_path = job_results_dir / "scored_features.csv"
            await self._save_csv_format(features, added_scores, csv_path)
            
            # Save summary report
            summary_path = job_results_dir / "summary_report.txt"
            with open(summary_path, 'w') as f:
                f.write(f"miStudioScore Summary Report\n")
                f.write(f"============================\n\n")
                f.write(f"Job ID: {job_id}\n")
                f.write(f"Source Type: {job_info['source_type']}\n")
                f.write(f"Source Job ID: {job_info['source_job_id']}\n")
                f.write(f"Generated: {job_info['completion_time']}\n")
                f.write(f"Processing Time: {processing_time:.2f} seconds\n\n")
                f.write(f"Results Summary:\n")
                f.write(f"Total Features: {len(features)}\n")
                f.write(f"Scores Added: {len(added_scores)}\n")
                f.write(f"Score Types: {', '.join(added_scores)}\n\n")
                f.write(f"Score Statistics:\n")
                for score_name in added_scores:
                    scores = [f.get(score_name, 0) for f in features if score_name in f]
                    if scores:
                        f.write(f"{score_name}:\n")
                        f.write(f"  Average: {sum(scores)/len(scores):.3f}\n")
                        f.write(f"  Max: {max(scores):.3f}\n")
                        f.write(f"  Min: {min(scores):.3f}\n")
                f.write(f"\nTop 5 Features by Average Score:\n")
                f.write(f"--------------------------------\n")
                # Calculate average scores and show top features
                for i, feature in enumerate(features[:5]):
                    avg_score = sum(feature.get(score, 0) for score in added_scores) / len(added_scores)
                    f.write(f"{i+1}. Feature {feature.get('feature_id', 'unknown')} (avg: {avg_score:.3f})\n")
            
            # Update job info with file paths
            job_info["saved_files"] = {
                "results_json": str(results_path),
                "job_info": str(job_info_path),
                "csv": str(csv_path),
                "summary": str(summary_path),
                "results_directory": str(job_results_dir)
            }
            
            logger.info(f"✅ Scoring results saved to {job_results_dir}")
            
        except Exception as e:
            logger.error(f"❌ Failed to save scoring results for job {job_id}: {e}")
            raise
    
    async def _save_csv_format(self, features: List[Dict[str, Any]], 
                             added_scores: List[str], csv_path: Path):
        """Save features in CSV format"""
        import csv
        
        # Determine all columns
        base_columns = ["feature_id", "coherence_score"]
        score_columns = added_scores
        optional_columns = ["quality_level", "pattern_keywords", "explanation"]
        
        # Check what columns are available
        available_columns = base_columns.copy()
        available_columns.extend(score_columns)
        
        for col in optional_columns:
            if any(col in feature for feature in features):
                available_columns.append(col)
        
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=available_columns)
            writer.writeheader()
            
            for feature in features:
                row = {}
                for col in available_columns:
                    if col == "pattern_keywords" and col in feature:
                        # Join keywords with semicolons
                        row[col] = "; ".join(str(k) for k in feature[col])
                    else:
                        row[col] = feature.get(col, "")
                writer.writerow(row)
    
    def generate_job_id(self) -> str:
        """Generate unique job ID"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        import uuid
        unique_id = str(uuid.uuid4())[:8]
        return f"score_{timestamp}_{unique_id}"
    
    def get_job_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get job status"""
        if job_id in self.active_jobs:
            return self.active_jobs[job_id]
        elif job_id in self.completed_jobs:
            return self.completed_jobs[job_id]
        else:
            return None
    
    def get_job_results(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get job results"""
        job_info = self.get_job_status(job_id)
        if job_info and job_info.get("status") == "completed":
            return job_info.get("results")
        return None
    
    def list_jobs(self) -> List[Dict[str, Any]]:
        """List all jobs"""
        all_jobs = []
        
        # Add active jobs
        for job_id, job_info in self.active_jobs.items():
            all_jobs.append({
                "job_id": job_id,
                "status": job_info["status"],
                "source_type": job_info["source_type"],
                "source_job_id": job_info["source_job_id"],
                "start_time": job_info["start_time"],
                "progress": job_info["progress"]
            })
        
        # Add completed jobs
        for job_id, job_info in self.completed_jobs.items():
            all_jobs.append({
                "job_id": job_id,
                "status": job_info["status"],
                "source_type": job_info["source_type"],
                "source_job_id": job_info["source_job_id"],
                "start_time": job_info["start_time"],
                "completion_time": job_info.get("completion_time"),
                "results_available": job_info.get("results") is not None
            })
        
        # Sort by start time (newest first)
        all_jobs.sort(key=lambda x: x["start_time"], reverse=True)
        return all_jobs


# =============================================================================
# Service Initialization
# =============================================================================

# Initialize core service
score_service = IntegratedScoreService(config)


# =============================================================================
# FastAPI Application Setup
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan - startup and shutdown."""
    logger.info(f"🚀 {config.service_name} API starting up...")
    logger.info(f"📁 Data path: {config.data_path}")
    logger.info(f"🔍 Find results: {config.find_results_dir}")
    logger.info(f"💬 Explain results: {config.explain_results_dir}")
    logger.info(f"🎯 Score results: {config.score_results_dir}")
    
    yield
    
    logger.info(f"🛑 {config.service_name} API shutting down...")

app = FastAPI(
    title="miStudioScore API",
    description="Enhanced scoring service integrated with miStudioFind and miStudioExplain",
    version=config.service_version,
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# Enhanced API Endpoints
# =============================================================================

@app.get("/", response_model=Dict[str, Any])
async def root():
    """Service information endpoint"""
    return {
        "service": config.service_name,
        "status": "running",
        "version": config.service_version,
        "description": "Enhanced scoring service for SAE features with shared storage integration",
        "available_scorers": score_service.available_scorers,
        "integration_ready": True,
        "documentation": "/docs"
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Enhanced health check endpoint"""
    
    return HealthResponse(
        status="healthy",
        service=config.service_name,
        version=config.service_version,
        data_path=config.data_path,
        find_results_path=str(config.find_results_dir),
        explain_results_path=str(config.explain_results_dir),
        score_results_path=str(config.score_results_dir),
        timestamp=datetime.now().isoformat(),
        components={
            "score_service": score_service is not None,
            "find_results_accessible": config.find_results_dir.exists(),
            "explain_results_accessible": config.explain_results_dir.exists(),
            "score_results_writable": config.score_results_dir.exists()
        },
        available_scorers=score_service.available_scorers
    )

@app.post("/api/v1/score/analyze", response_model=ScoreJobResponse)
async def start_scoring_job(request: ScoreRequest):
    """Start scoring job"""
    
    try:
        # Start scoring job using integrated service
        job_id = await score_service.start_scoring_job(request)
        
        logger.info(f"🎯 Started scoring job: {job_id}")
        logger.info(f"📊 Processing {request.source_type} job: {request.source_job_id}")
        logger.info(f"⚙️  Scoring jobs: {len(request.scoring_config.get('scoring_jobs', []))}")
        logger.info(f"💾 Results will be saved to: {config.score_results_dir / job_id}")
        
        return ScoreJobResponse(
            job_id=job_id,
            status="queued",
            message=f"Scoring job '{job_id}' started for {request.source_type} job {request.source_job_id}",
            source_type=request.source_type,
            source_job_id=request.source_job_id,
            parameters=ScoreParameters(
                source_type=request.source_type,
                source_job_id=request.source_job_id,
                scoring_jobs=request.scoring_config.get("scoring_jobs", []),
                output_formats=request.output_formats
            ),
            timestamp=datetime.now().isoformat(),
            next_steps=NextSteps(
                check_status=f"/api/v1/score/{job_id}/status",
                get_results=f"/api/v1/score/{job_id}/results"
            )
        )
        
    except Exception as e:
        logger.error(f"Error starting scoring job: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/score/{job_id}/status", response_model=JobStatusResponse)
async def get_job_status(job_id: str):
    """Get status of scoring job"""
    
    try:
        status_data = score_service.get_job_status(job_id)
        
        if status_data is None:
            raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
        
        # Convert to response model
        progress_data = status_data.get("progress", {})
        
        # Calculate processing time if completed
        processing_time = None
        if status_data.get("start_time") and status_data.get("completion_time"):
            start = datetime.fromisoformat(status_data["start_time"])
            end = datetime.fromisoformat(status_data["completion_time"])
            processing_time = (end - start).total_seconds()
        
        return JobStatusResponse(
            job_id=job_id,
            status=status_data.get("status", "unknown"),
            source_type=status_data.get("source_type"),
            source_job_id=status_data.get("source_job_id"),
            start_time=status_data.get("start_time"),
            completion_time=status_data.get("completion_time"),
            processing_time=processing_time,
            progress=JobProgress(
                scoring_jobs_processed=progress_data.get("scoring_jobs_processed", 0),
                total_scoring_jobs=progress_data.get("total_scoring_jobs", 0),
                current_job=progress_data.get("current_job"),
                estimated_time_remaining=progress_data.get("estimated_time_remaining")
            ) if progress_data else None,
            error=status_data.get("error"),
            results_path=status_data.get("results_path")
        )
        
    except Exception as e:
        logger.error(f"Error getting job status: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/score/{job_id}/results", response_model=JobResultResponse)
async def get_job_results(job_id: str):
    """Get results of completed scoring job"""
    
    try:
        status_data = score_service.get_job_status(job_id)
        
        if status_data is None:
            raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
        
        if status_data.get("status") != "completed":
            raise HTTPException(
                status_code=400, 
                detail=f"Job {job_id} is not completed (status: {status_data.get('status')})"
            )
        
        results = score_service.get_job_results(job_id)
        
        if results is None:
            raise HTTPException(status_code=404, detail=f"Results not found for job {job_id}")
        
        # Calculate processing time
        processing_time = None
        if status_data.get("start_time") and status_data.get("completion_time"):
            start = datetime.fromisoformat(status_data["start_time"])
            end = datetime.fromisoformat(status_data["completion_time"])
            processing_time = (end - start).total_seconds()
        
        # Prepare download links for saved files
        download_links = {}
        saved_files = status_data.get("saved_files", {})
        if saved_files:
            for file_type, file_path in saved_files.items():
                download_links[file_type] = f"/api/v1/score/{job_id}/download/{file_type}"
        
        return JobResultResponse(
            job_id=job_id,
            status=status_data["status"],
            source_type=status_data["source_type"],
            source_job_id=status_data["source_job_id"],
            scores_added=results.get("scores_added", []),
            results_summary={
                "total_features": results.get("total_features", 0),
                "scores_added": results.get("scores_added", []),
                "source_type": status_data["source_type"],
                "source_job_id": status_data["source_job_id"],
                "analysis_type": "feature_scoring",
                "results_available": True
            },
            download_links=download_links if download_links else None,
            processing_time=processing_time
        )
        
    except Exception as e:
        logger.error(f"Error getting job results: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/score/jobs", response_model=JobListResponse)
async def list_jobs():
    """List all scoring jobs"""
    
    try:
        jobs_data = score_service.list_jobs()
        
        jobs = []
        for job_data in jobs_data:
            jobs.append(JobSummary(
                job_id=job_data["job_id"],
                status=job_data["status"],
                source_type=job_data["source_type"],
                source_job_id=job_data["source_job_id"],
                created_at=job_data["start_time"],
                completed_at=job_data.get("completion_time")
            ))
        
        return JobListResponse(
            jobs=jobs,
            total=len(jobs)
        )
        
    except Exception as e:
        logger.error(f"Error listing jobs: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/score/sources/available")
async def list_available_source_jobs(source_type: str = "find"):
    """List available source jobs for scoring"""
    
    try:
        if source_type not in ["find", "explain"]:
            raise HTTPException(status_code=400, detail="source_type must be 'find' or 'explain'")
        
        available_jobs = score_service.discover_source_jobs(source_type)
        
        return {
            "available_jobs": available_jobs,
            "total": len(available_jobs),
            "source_type": source_type,
            "results_path": str(config.find_results_dir if source_type == "find" else config.explain_results_dir)
        }
        
    except Exception as e:
        logger.error(f"Error listing available source jobs: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/score/sources/{source_type}/{source_job_id}/preview")
async def preview_source_job(source_type: str, source_job_id: str, limit: int = 5):
    """Preview a source job before scoring"""
    
    try:
        if source_type not in ["find", "explain"]:
            raise HTTPException(status_code=400, detail="source_type must be 'find' or 'explain'")
        
        # Load source data
        source_data = score_service.load_source_data(source_type, source_job_id)
        
        if source_type == "find":
            features = source_data.get('results', [])
            preview_items = features[:limit]
            
            return {
                "source_job_id": source_job_id,
                "source_type": source_type,
                "total_features": len(features),
                "preview_features": [
                    {
                        "feature_id": f.get('feature_id', 'unknown'),
                        "coherence_score": f.get('coherence_score', 0),
                        "quality_level": f.get('quality_level', 'unknown'),
                        "pattern_keywords": f.get('pattern_keywords', [])
                    }
                    for f in preview_items
                ],
                "recommended_scorers": ["relevance_scorer", "pattern_scorer"]
            }
        
        elif source_type == "explain":
            explanations = source_data.get('explanations', [])
            preview_items = explanations[:limit]
            
            return {
                "source_job_id": source_job_id,
                "source_type": source_type,
                "total_explanations": len(explanations),
                "preview_explanations": [
                    {
                        "feature_id": e.get('feature_id', 'unknown'),
                        "explanation": e.get('explanation', '')[:100] + "..." if len(e.get('explanation', '')) > 100 else e.get('explanation', ''),
                        "confidence": e.get('confidence', 'unknown'),
                        "coherence_score": e.get('coherence_score', 0)
                    }
                    for e in preview_items
                ],
                "recommended_scorers": ["relevance_scorer", "pattern_scorer"]
            }
        
    except Exception as e:
        logger.error(f"Error previewing source job: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/score/{job_id}/download/{file_type}")
async def download_result_file(job_id: str, file_type: str):
    """Download a specific result file"""
    
    try:
        status_data = score_service.get_job_status(job_id)
        
        if status_data is None:
            raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
        
        saved_files = status_data.get("saved_files", {})
        
        if file_type not in saved_files:
            raise HTTPException(status_code=404, detail=f"File type '{file_type}' not found for job {job_id}")
        
        file_path = Path(saved_files[file_type])
        
        if not file_path.exists():
            raise HTTPException(status_code=404, detail=f"File not found: {file_path}")
        
        return FileResponse(
            path=str(file_path),
            filename=f"{job_id}_{file_type}{file_path.suffix}",
            media_type="application/octet-stream"
        )
        
    except Exception as e:
        logger.error(f"Error downloading file: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/score/{job_id}/download/all")
async def download_all_results(job_id: str):
    """Download all result files as a ZIP archive"""
    
    try:
        status_data = score_service.get_job_status(job_id)
        
        if status_data is None:
            raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
        
        if status_data.get("status") != "completed":
            raise HTTPException(status_code=400, detail=f"Job {job_id} is not completed")
        
        # Create ZIP archive in memory
        zip_buffer = io.BytesIO()
        
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            # Add saved files to ZIP
            saved_files = status_data.get("saved_files", {})
            for file_type, file_path in saved_files.items():
                file_path_obj = Path(file_path)
                if file_path_obj.exists():
                    zip_file.write(file_path_obj, f"{job_id}_{file_type}{file_path_obj.suffix}")
        
        zip_buffer.seek(0)
        
        return StreamingResponse(
            io.BytesIO(zip_buffer.read()),
            media_type="application/zip",
            headers={"Content-Disposition": f"attachment; filename={job_id}_scores.zip"}
        )
        
    except Exception as e:
        logger.error(f"Error creating ZIP download: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Legacy endpoint for backward compatibility
@app.post("/score")
async def legacy_score_endpoint(
    features_path: str,
    config_path: str,
    output_dir: str
):
    """Legacy scoring endpoint for backward compatibility"""
    
    return JSONResponse(
        status_code=400,
        content={
            "error": "Legacy endpoint deprecated",
            "message": "This endpoint has been replaced with the new integrated API",
            "new_endpoint": "/api/v1/score/analyze",
            "migration_guide": "Use the new API with source_type and source_job_id parameters"
        }
    )


# =============================================================================
# Application Entry Point
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    
    logger.info(f"🚀 Starting {config.service_name} v{config.service_version}")
    logger.info(f"📁 Data path: {config.data_path}")
    logger.info(f"🔍 Find results: {config.find_results_dir}")
    logger.info(f"💬 Explain results: {config.explain_results_dir}")
    logger.info(f"🎯 Score results: {config.score_results_dir}")
    logger.info(f"🌐 API server: {config.api_host}:{config.api_port}")
    
    uvicorn.run(
        app,
        host=config.api_host,
        port=config.api_port,
        log_level=config.log_level.lower(),
        access_log=True
    )