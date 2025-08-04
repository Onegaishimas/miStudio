# src/main.py - Updated miStudioExplain with Find Integration and Proper Storage
"""
Main FastAPI application for miStudioExplain service - Integrated with Find outputs.
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
# Enhanced Configuration with Find Integration
# =============================================================================

class ServiceConfig:
    """Enhanced configuration for miStudioExplain with Find integration"""
    
    def __init__(self):
        # Primary data path - same pattern for all services
        self.data_path = os.getenv("DATA_PATH", "/data")
        self.data_path_obj = Path(self.data_path)
        self.data_path_obj.mkdir(parents=True, exist_ok=True)
        
        # Service metadata
        self.service_name = "miStudioExplain"
        self.service_version = "1.0.0"
        
        # API configuration
        self.api_host = os.getenv("API_HOST", "0.0.0.0")
        self.api_port = int(os.getenv("API_PORT", "8003"))
        self.log_level = os.getenv("LOG_LEVEL", "INFO")
        
        # Ollama configuration
        self.ollama_endpoint = os.getenv("OLLAMA_ENDPOINT", "http://ollama.mcslab.io")
        self.ollama_namespace = os.getenv("OLLAMA_NAMESPACE", "mistudio")
        self.ollama_models = os.getenv("OLLAMA_MODELS", "llama3.1:70b,llama3.1:8b").split(",")
        self.ollama_timeout = int(os.getenv("OLLAMA_TIMEOUT", "300"))
        self.ollama_max_retries = int(os.getenv("OLLAMA_MAX_RETRIES", "3"))
        
        # Processing configuration
        self.default_quality_threshold = float(os.getenv("QUALITY_THRESHOLD", "0.5"))
        self.high_quality_threshold = float(os.getenv("HIGH_QUALITY_THRESHOLD", "0.8"))
        self.excellent_threshold = float(os.getenv("EXCELLENT_THRESHOLD", "0.9"))
        self.max_batch_size = int(os.getenv("MAX_BATCH_SIZE", "10"))
        self.max_concurrent_explanations = int(os.getenv("MAX_CONCURRENT_EXPLANATIONS", "3"))
        self.explanation_timeout = int(os.getenv("EXPLANATION_TIMEOUT", "120"))
        
        # Enhanced directory structure
        self._ensure_directories()
        
        logger.info(f"🔧 ServiceConfig initialized:")
        logger.info(f"   Data path: {self.data_path}")
        logger.info(f"   Find results: {self.find_results_dir}")
        logger.info(f"   Explain results: {self.explain_results_dir}")
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
    def cache_dir(self) -> Path:
        """Cache directory"""
        return self.data_path_obj / "cache" / "explain"
    
    @property
    def logs_dir(self) -> Path:
        """Logs directory"""
        return self.data_path_obj / "logs" / "explain"
    
    def _ensure_directories(self):
        """Ensure all required directories exist"""
        directories = [
            self.find_results_dir,
            self.explain_results_dir,
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

class ExplainRequest(BaseModel):
    """Enhanced request model for explanation generation"""
    find_job_id: str = Field(..., description="Source miStudioFind job ID")
    analysis_type: str = Field(default="behavioral_patterns", description="Type of analysis to perform")
    complexity: str = Field(default="medium", description="Explanation complexity level")
    model: Optional[str] = Field(default=None, description="Specific LLM model to use")
    quality_threshold: float = Field(default=0.5, description="Minimum feature quality threshold")
    max_features: int = Field(default=20, description="Maximum number of features to explain")
    include_examples: bool = Field(default=True, description="Include text examples in explanations")

class ExplainParameters(BaseModel):
    """Parameters included in response"""
    analysis_type: str
    complexity: str
    model: Optional[str]
    quality_threshold: float
    max_features: int
    include_examples: bool

class NextSteps(BaseModel):
    """Next steps information"""
    check_status: str
    get_results: str

class ExplainJobResponse(BaseModel):
    """Enhanced response model for starting explanation job"""
    job_id: str
    status: str
    message: str
    find_job_id: str
    parameters: ExplainParameters
    timestamp: str
    next_steps: NextSteps

class JobProgress(BaseModel):
    """Enhanced job progress information"""
    features_processed: int
    total_features: int
    explanations_generated: int
    estimated_time_remaining: Optional[int] = None

class JobStatusResponse(BaseModel):
    """Enhanced response model for job status"""
    job_id: str
    status: str
    find_job_id: str
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
    find_job_id: str
    explanation_count: Optional[int] = None
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
    timestamp: str
    components: Dict[str, bool]
    ollama_status: Dict[str, Any]

class JobSummary(BaseModel):
    """Job summary model"""
    job_id: str
    status: str
    find_job_id: str
    created_at: str
    completed_at: Optional[str] = None

class JobListResponse(BaseModel):
    """Response model for job listing"""
    jobs: List[JobSummary]
    total: int


# =============================================================================
# Enhanced Service Implementation
# =============================================================================

class IntegratedExplainService:
    """Integrated service that works with Find outputs and stores in proper locations"""
    
    def __init__(self, config: ServiceConfig):
        self.config = config
        self.active_jobs: Dict[str, Dict[str, Any]] = {}
        self.completed_jobs: Dict[str, Dict[str, Any]] = {}
        
        logger.info(f"IntegratedExplainService initialized")
        logger.info(f"  Find results path: {self.config.find_results_dir}")
        logger.info(f"  Explain results path: {self.config.explain_results_dir}")
    
    def discover_find_jobs(self) -> List[Dict[str, Any]]:
        """Discover available Find jobs for explanation"""
        available_jobs = []
        
        # Check main Find results directory
        if self.config.find_results_dir.exists():
            for job_dir in self.config.find_results_dir.iterdir():
                if job_dir.is_dir():
                    results_file = job_dir / "analysis_results.json"
                    if results_file.exists():
                        try:
                            with open(results_file, 'r') as f:
                                data = json.load(f)
                            
                            available_jobs.append({
                                "find_job_id": job_dir.name,
                                "source_job_id": data.get('source_job_id', 'unknown'),
                                "feature_count": len(data.get('results', [])),
                                "status": "available",
                                "results_path": str(results_file),
                                "location": "main"
                            })
                        except Exception as e:
                            logger.warning(f"Could not read Find job {job_dir.name}: {e}")
        
        # Check enhanced persistence directories
        results_base = self.config.data_path_obj / "results"
        if results_base.exists():
            for job_dir in results_base.glob("find_*"):
                if job_dir.is_dir() and job_dir.name not in [j["find_job_id"] for j in available_jobs]:
                    results_file = job_dir / f"{job_dir.name}_complete_results.json"
                    if results_file.exists():
                        try:
                            with open(results_file, 'r') as f:
                                data = json.load(f)
                            
                            available_jobs.append({
                                "find_job_id": job_dir.name,
                                "source_job_id": data.get('source_job_id', 'unknown'),
                                "feature_count": len(data.get('results', [])),
                                "status": "available",
                                "results_path": str(results_file),
                                "location": "enhanced"
                            })
                        except Exception as e:
                            logger.warning(f"Could not read enhanced Find job {job_dir.name}: {e}")
        
        logger.info(f"Discovered {len(available_jobs)} available Find jobs")
        return available_jobs
    
    def load_find_results(self, find_job_id: str) -> Dict[str, Any]:
        """Load Find results with comprehensive fallback logic - enhanced like miStudioScore"""
        
        logger.info(f"🔍 Loading Find data for job: {find_job_id}")
        
        # Define multiple possible file paths based on observed structure (like miStudioScore)
        possible_paths = [
            self.config.find_results_dir / find_job_id / "analysis_results.json",
            self.config.find_results_dir / find_job_id / f"{find_job_id}_analysis_results.json",
            self.config.find_results_dir / find_job_id / f"{find_job_id}_complete_results.json",
            self.config.data_path_obj / "results" / find_job_id / f"{find_job_id}_complete_results.json",
            self.config.find_results_dir / find_job_id / f"{find_job_id}_analysis.json",
            self.config.data_path_obj / "results" / find_job_id / f"{find_job_id}_analysis.json",
            self.config.find_results_dir / find_job_id / "results.json"
        ]
        
        # Try each path until we find valid data
        for source_file in possible_paths:
            if source_file.exists():
                try:
                    logger.info(f"📂 Attempting to load: {source_file}")
                    with open(source_file, 'r') as f:
                        data = json.load(f)
                    
                    # Validate data contains expected content
                    if "results" in data and data["results"]:
                        logger.info(f"✅ Loaded {len(data['results'])} features from: {source_file}")
                        return data
                    elif "features" in data and data["features"]:
                        # Handle case where results are stored as features
                        logger.info(f"✅ Loaded {len(data['features'])} features (as results) from: {source_file}")
                        return {"results": data["features"], **{k: v for k, v in data.items() if k != "features"}}
                    
                    logger.warning(f"📂 File {source_file} exists but doesn't contain expected data structure")
                    
                except Exception as e:
                    logger.warning(f"Error reading {source_file}: {e}")
                    continue
            else:
                logger.debug(f"📂 File not found: {source_file}")
        
        # If we get here, no valid data was found - provide comprehensive debugging info
        logger.error(f"❌ No valid source data found for Find job {find_job_id}")
        logger.error(f"🔍 Searched paths:")
        for path in possible_paths:
            logger.error(f"   - {path} (exists: {path.exists()})")
        
        # List available jobs for debugging
        available_jobs = self.discover_find_jobs()
        available_ids = [job["find_job_id"] for job in available_jobs]
        logger.error(f"📋 Available Find jobs: {available_ids}")
        
        # Show directory contents for debugging
        find_dir = self.config.find_results_dir
        if find_dir.exists():
            logger.error(f"📁 Contents of {find_dir}:")
            try:
                for item in find_dir.iterdir():
                    if item.is_dir():
                        logger.error(f"   📁 {item.name}/")
                        try:
                            for subitem in item.iterdir():
                                logger.error(f"      📄 {subitem.name}")
                        except:
                            pass
            except Exception as e:
                logger.error(f"   Error listing contents: {e}")
        
        raise ValueError(f"Find job {find_job_id} not found. Available jobs: {available_ids}")
    
    def filter_features(self, features: List[Dict[str, Any]], quality_threshold: float, max_features: int) -> List[Dict[str, Any]]:
        """Filter and select features based on quality and limit"""
        # Filter by quality threshold
        high_quality_features = [
            f for f in features 
            if f.get('coherence_score', 0) >= quality_threshold
        ]
        
        # Sort by coherence score (descending)
        high_quality_features.sort(key=lambda x: x.get('coherence_score', 0), reverse=True)
        
        # Limit number of features
        selected_features = high_quality_features[:max_features]
        
        logger.info(f"Filtered {len(features)} features to {len(selected_features)} (threshold: {quality_threshold})")
        return selected_features
    
    async def start_explanation_job(self, request: ExplainRequest) -> str:
        """Start explanation generation job"""
        job_id = self.generate_job_id()
        
        try:
            # Load and validate Find results
            find_data = self.load_find_results(request.find_job_id)
            
            # Extract and filter features
            features = find_data.get('results', [])
            if not features:
                raise ValueError(f"No features found in Find job {request.find_job_id}")
            
            selected_features = self.filter_features(features, request.quality_threshold, request.max_features)
            
            if not selected_features:
                raise ValueError(f"No features meet quality threshold {request.quality_threshold}")
            
            # Create job entry
            job_info = {
                "job_id": job_id,
                "find_job_id": request.find_job_id,
                "status": "queued",
                "start_time": datetime.now().isoformat(),
                "parameters": request.dict(),
                "selected_features": selected_features,
                "progress": {
                    "features_processed": 0,
                    "total_features": len(selected_features),
                    "explanations_generated": 0,
                    "estimated_time_remaining": None
                },
                "results": None,
                "error": None,
                "results_path": str(self.config.explain_results_dir / job_id)
            }
            
            self.active_jobs[job_id] = job_info
            
            # Start processing in background
            asyncio.create_task(self._process_explanation_job(job_id))
            
            logger.info(f"Started explanation job {job_id} for Find job {request.find_job_id}")
            logger.info(f"Selected {len(selected_features)} features above threshold {request.quality_threshold}")
            
            return job_id
            
        except Exception as e:
            logger.error(f"Failed to start explanation job: {e}")
            raise
    
    async def _process_explanation_job(self, job_id: str):
        """Process explanation job"""
        job_info = self.active_jobs[job_id]
        
        try:
            job_info["status"] = "running"
            logger.info(f"Processing explanation job {job_id}...")
            
            selected_features = job_info["selected_features"]
            explanations = []
            
            # Process each feature
            for i, feature in enumerate(selected_features):
                explanation = self._generate_explanation(feature, job_info["parameters"])
                explanations.append(explanation)
                
                # Update progress
                job_info["progress"]["features_processed"] = i + 1
                job_info["progress"]["explanations_generated"] = len(explanations)
                
                # Simulate processing time
                await asyncio.sleep(0.2)
            
            # Complete job
            job_info["status"] = "completed"
            job_info["completion_time"] = datetime.now().isoformat()
            job_info["results"] = explanations
            
            # Save results to files
            await self._save_explanation_results(job_id, job_info, explanations)
            
            # Move to completed jobs
            self.completed_jobs[job_id] = self.active_jobs.pop(job_id)
            
            logger.info(f"✅ Explanation job {job_id} completed with {len(explanations)} explanations")
            
        except Exception as e:
            logger.error(f"❌ Explanation job {job_id} failed: {str(e)}")
            job_info["status"] = "failed"
            job_info["error"] = str(e)
            job_info["completion_time"] = datetime.now().isoformat()
            self.completed_jobs[job_id] = self.active_jobs.pop(job_id)
    
    def _generate_explanation(self, feature: Dict[str, Any], parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Generate explanation for a feature"""
        feature_id = feature.get('feature_id', 'unknown')
        coherence = feature.get('coherence_score', 0)
        keywords = feature.get('pattern_keywords', [])
        top_activations = feature.get('top_activations', [])
        quality_level = feature.get('quality_level', 'unknown')
        
        # Extract sample texts
        sample_texts = []
        for activation in top_activations[:3]:
            if isinstance(activation, dict) and 'text' in activation:
                sample_texts.append(activation['text'])
        
        # Generate explanation based on available data
        if keywords and len(keywords) > 0:
            keyword_str = ', '.join(str(k) for k in keywords[:5])
            if sample_texts:
                explanation_text = f"This feature detects text patterns containing: {keyword_str}. "
                explanation_text += f"It activates strongly on content like: '{sample_texts[0][:100]}...'"
            else:
                explanation_text = f"This feature recognizes patterns related to: {keyword_str}."
        else:
            if sample_texts:
                explanation_text = f"This feature activates on specific text patterns. Example: '{sample_texts[0][:100]}...'"
            else:
                explanation_text = "This feature detects a specific but undefined text pattern."
        
        # Determine confidence based on coherence and available data
        if coherence >= 0.9 and keywords and sample_texts:
            confidence = "high"
        elif coherence >= 0.7 and (keywords or sample_texts):
            confidence = "medium"
        else:
            confidence = "low"
        
        return {
            "feature_id": feature_id,
            "explanation": explanation_text,
            "confidence": confidence,
            "coherence_score": coherence,
            "quality_level": quality_level,
            "pattern_keywords": keywords,
            "sample_texts": sample_texts[:3],
            "explanation_type": "pattern_based",
            "analysis_parameters": parameters,
            "generated_by": "integrated_explain_service",
            "timestamp": datetime.now().isoformat()
        }
    
    async def _save_explanation_results(self, job_id: str, job_info: Dict[str, Any], explanations: List[Dict[str, Any]]):
        """Save explanation results to shared storage"""
        try:
            # Create job-specific directory in /data/results/explain
            job_results_dir = self.config.explain_results_dir / job_id
            job_results_dir.mkdir(parents=True, exist_ok=True)
            
            # Calculate processing time
            start_time = datetime.fromisoformat(job_info["start_time"])
            end_time = datetime.fromisoformat(job_info["completion_time"])
            processing_time = (end_time - start_time).total_seconds()
            
            # Create comprehensive results structure
            results_data = {
                "job_id": job_id,
                "find_job_id": job_info["find_job_id"],
                "status": "completed",
                "start_time": job_info["start_time"],
                "completion_time": job_info["completion_time"],
                "processing_time": processing_time,
                "parameters": job_info["parameters"],
                "explanations": explanations,
                "summary": {
                    "total_explanations": len(explanations),
                    "find_job_id": job_info["find_job_id"],
                    "high_confidence": sum(1 for e in explanations if e["confidence"] == "high"),
                    "medium_confidence": sum(1 for e in explanations if e["confidence"] == "medium"),
                    "low_confidence": sum(1 for e in explanations if e["confidence"] == "low"),
                    "average_coherence": sum(e["coherence_score"] for e in explanations) / len(explanations),
                    "service": "miStudioExplain",
                    "version": self.config.service_version
                },
                "timestamp": datetime.now().isoformat()
            }
            
            # Save main results JSON
            results_path = job_results_dir / "explanation_results.json"
            with open(results_path, 'w') as f:
                json.dump(results_data, f, indent=2, default=str)
            
            # Save job info
            job_info_path = job_results_dir / "job_info.json"
            with open(job_info_path, 'w') as f:
                json.dump(job_info, f, indent=2, default=str)
            
            # Save explanations in CSV format
            csv_path = job_results_dir / "explanations.csv"
            with open(csv_path, 'w') as f:
                f.write("feature_id,explanation,confidence,coherence_score,quality_level,keywords,sample_text\n")
                for exp in explanations:
                    keywords_str = '; '.join(str(k) for k in exp.get('pattern_keywords', []))
                    sample_text = exp.get('sample_texts', [''])[0][:100] if exp.get('sample_texts') else ''
                    # Escape quotes in CSV
                    explanation_clean = exp['explanation'].replace('"', '""')
                    sample_clean = sample_text.replace('"', '""')
                    
                    f.write(f"{exp['feature_id']},\"{explanation_clean}\",{exp['confidence']},{exp['coherence_score']},{exp['quality_level']},\"{keywords_str}\",\"{sample_clean}\"\n")
            
            # Save summary report
            summary_path = job_results_dir / "summary_report.txt"
            with open(summary_path, 'w') as f:
                f.write(f"miStudioExplain Summary Report\n")
                f.write(f"==============================\n\n")
                f.write(f"Job ID: {job_id}\n")
                f.write(f"Find Job ID: {job_info['find_job_id']}\n")
                f.write(f"Generated: {job_info['completion_time']}\n")
                f.write(f"Processing Time: {processing_time:.2f} seconds\n\n")
                f.write(f"Results Summary:\n")
                f.write(f"Total Explanations: {len(explanations)}\n")
                f.write(f"High Confidence: {results_data['summary']['high_confidence']}\n")
                f.write(f"Medium Confidence: {results_data['summary']['medium_confidence']}\n")
                f.write(f"Low Confidence: {results_data['summary']['low_confidence']}\n")
                f.write(f"Average Coherence: {results_data['summary']['average_coherence']:.3f}\n\n")
                f.write(f"Parameters Used:\n")
                f.write(f"Quality Threshold: {job_info['parameters']['quality_threshold']}\n")
                f.write(f"Max Features: {job_info['parameters']['max_features']}\n")
                f.write(f"Analysis Type: {job_info['parameters']['analysis_type']}\n")
                f.write(f"Complexity: {job_info['parameters']['complexity']}\n\n")
                f.write(f"Top 5 Explanations:\n")
                f.write(f"-------------------\n")
                for i, exp in enumerate(explanations[:5]):
                    f.write(f"{i+1}. Feature {exp['feature_id']} ({exp['confidence']} confidence):\n")
                    f.write(f"   {exp['explanation'][:150]}...\n\n")
            
            # Update job info with file paths
            job_info["saved_files"] = {
                "results_json": str(results_path),
                "job_info": str(job_info_path),
                "csv": str(csv_path),
                "summary": str(summary_path),
                "results_directory": str(job_results_dir)
            }
            
            logger.info(f"✅ Explanation results saved to {job_results_dir}")
            
        except Exception as e:
            logger.error(f"❌ Failed to save explanation results for job {job_id}: {e}")
            raise
    
    def generate_job_id(self) -> str:
        """Generate unique job ID"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        import uuid
        unique_id = str(uuid.uuid4())[:8]
        return f"explain_{timestamp}_{unique_id}"
    
    def get_job_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get job status"""
        if job_id in self.active_jobs:
            return self.active_jobs[job_id]
        elif job_id in self.completed_jobs:
            return self.completed_jobs[job_id]
        else:
            return None
    
    def get_job_results(self, job_id: str) -> Optional[List[Dict[str, Any]]]:
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
                "find_job_id": job_info["find_job_id"],
                "start_time": job_info["start_time"],
                "progress": job_info["progress"]
            })
        
        # Add completed jobs
        for job_id, job_info in self.completed_jobs.items():
            all_jobs.append({
                "job_id": job_id,
                "status": job_info["status"],
                "find_job_id": job_info["find_job_id"],
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
explain_service = IntegratedExplainService(config)

# Initialize other components with fallback handling
ollama_manager = None
input_manager = None
feature_prioritizer = None
context_builder = None
explanation_generator = None
quality_validator = None
result_manager = None

# Initialize Ollama manager inline (if needed)
ollama_manager = None
input_manager = None

# Basic Ollama configuration without separate module
ollama_config = {
    "endpoint": config.ollama_endpoint,
    "models": config.ollama_models,
    "timeout": config.ollama_timeout
}

logger.info("✅ Service initialized without external managers")

# Additional component initializations...
# (keeping the existing pattern but not required for basic functionality)


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
    logger.info(f"🤖 Ollama endpoint: {config.ollama_endpoint}")
    
    # Initialize Ollama if available
    if ollama_manager:
        try:
            await ollama_manager.initialize()
            logger.info("✅ Ollama manager initialized")
        except Exception as e:
            logger.warning(f"Ollama initialization failed: {e}")
    
    yield
    
    # Cleanup
    if ollama_manager:
        try:
            await ollama_manager.cleanup()
            logger.info("✅ Ollama manager cleaned up")
        except Exception as e:
            logger.warning(f"Ollama cleanup failed: {e}")
    
    logger.info(f"🛑 {config.service_name} API shutting down...")

app = FastAPI(
    title="miStudioExplain API",
    description="Enhanced explanation generation service integrated with miStudioFind",
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

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Enhanced health check endpoint"""
    
    # Check Ollama status
    ollama_status = {"available": False, "models": [], "endpoint": config.ollama_endpoint}
    if ollama_manager:
        try:
            ollama_health = await ollama_manager.health_check()
            ollama_status = {
                "available": ollama_health,
                "models": config.ollama_models,
                "endpoint": config.ollama_endpoint,
                "timeout": config.ollama_timeout
            }
        except Exception as e:
            ollama_status["error"] = str(e)
    
    return HealthResponse(
        status="healthy",
        service=config.service_name,
        version=config.service_version,
        data_path=config.data_path,
        find_results_path=str(config.find_results_dir),
        explain_results_path=str(config.explain_results_dir),
        timestamp=datetime.now().isoformat(),
        components={
            "explain_service": explain_service is not None,
            "ollama_manager": ollama_manager is not None,
            "find_results_accessible": config.find_results_dir.exists(),
            "explain_results_writable": config.explain_results_dir.exists()
        },
        ollama_status=ollama_status
    )

@app.post("/api/v1/explain/analyze", response_model=ExplainJobResponse)
async def start_explanation_job(request: ExplainRequest):
    """Start explanation generation job"""
    
    try:
        # Start explanation job using integrated service
        job_id = await explain_service.start_explanation_job(request)
        
        logger.info(f"💬 Started explanation job: {job_id}")
        logger.info(f"🔍 Processing Find job: {request.find_job_id}")
        logger.info(f"⚙️  Parameters: quality_threshold={request.quality_threshold}, max_features={request.max_features}")
        logger.info(f"💾 Results will be saved to: {config.explain_results_dir / job_id}")
        
        return ExplainJobResponse(
            job_id=job_id,
            status="queued",
            message=f"Explanation job '{job_id}' started for Find job {request.find_job_id}",
            find_job_id=request.find_job_id,
            parameters=ExplainParameters(
                analysis_type=request.analysis_type,
                complexity=request.complexity,
                model=request.model,
                quality_threshold=request.quality_threshold,
                max_features=request.max_features,
                include_examples=request.include_examples
            ),
            timestamp=datetime.now().isoformat(),
            next_steps=NextSteps(
                check_status=f"/api/v1/explain/{job_id}/status",
                get_results=f"/api/v1/explain/{job_id}/results"
            )
        )
        
    except Exception as e:
        logger.error(f"Error starting explanation job: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/explain/{job_id}/status", response_model=JobStatusResponse)
async def get_job_status(job_id: str):
    """Get status of explanation job"""
    
    try:
        status_data = explain_service.get_job_status(job_id)
        
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
            find_job_id=status_data.get("find_job_id"),
            start_time=status_data.get("start_time"),
            completion_time=status_data.get("completion_time"),
            processing_time=processing_time,
            progress=JobProgress(
                features_processed=progress_data.get("features_processed", 0),
                total_features=progress_data.get("total_features", 0),
                explanations_generated=progress_data.get("explanations_generated", 0),
                estimated_time_remaining=progress_data.get("estimated_time_remaining")
            ) if progress_data else None,
            error=status_data.get("error"),
            results_path=status_data.get("results_path")
        )
        
    except Exception as e:
        logger.error(f"Error getting job status: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/explain/{job_id}/results", response_model=JobResultResponse)
async def get_job_results(job_id: str):
    """Get results of completed explanation job"""
    
    try:
        status_data = explain_service.get_job_status(job_id)
        
        if status_data is None:
            raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
        
        if status_data.get("status") != "completed":
            raise HTTPException(
                status_code=400, 
                detail=f"Job {job_id} is not completed (status: {status_data.get('status')})"
            )
        
        results = explain_service.get_job_results(job_id)
        
        if results is None:
            raise HTTPException(status_code=404, detail=f"Results not found for job {job_id}")
        
        # Calculate processing time
        processing_time = None
        if status_data.get("start_time") and status_data.get("completion_time"):
            start = datetime.fromisoformat(status_data["start_time"])
            end = datetime.fromisoformat(status_data["completion_time"])
            processing_time = (end - start).total_seconds()
        
        # Prepare download links with standardized endpoints like miStudioScore
        download_links = {
            "json": f"/api/v1/explain/{job_id}/download/json",
            "csv": f"/api/v1/explain/{job_id}/download/csv"
        }
        
        # Add additional file download links for saved files
        saved_files = status_data.get("saved_files", {})
        if saved_files:
            for file_type, file_path in saved_files.items():
                if file_type not in download_links:  # Don't override standard endpoints
                    download_links[file_type] = f"/api/v1/explain/{job_id}/download/{file_type}"
        
        # Calculate summary statistics
        high_conf = sum(1 for r in results if r.get("confidence") == "high")
        medium_conf = sum(1 for r in results if r.get("confidence") == "medium")
        low_conf = sum(1 for r in results if r.get("confidence") == "low")
        avg_coherence = sum(r.get("coherence_score", 0) for r in results) / len(results) if results else 0
        
        return JobResultResponse(
            job_id=job_id,
            status=status_data["status"],
            find_job_id=status_data["find_job_id"],
            explanation_count=len(results) if results else 0,
            results_summary={
                "total_explanations": len(results) if results else 0,
                "high_confidence": high_conf,
                "medium_confidence": medium_conf,
                "low_confidence": low_conf,
                "average_coherence": avg_coherence,
                "find_job_id": status_data["find_job_id"],
                "analysis_type": "feature_explanation",
                "results_available": True
            },
            download_links=download_links if download_links else None,
            processing_time=processing_time
        )
        
    except Exception as e:
        logger.error(f"Error getting job results: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/explain/jobs", response_model=JobListResponse)
async def list_jobs():
    """List all explanation jobs"""
    
    try:
        jobs_data = explain_service.list_jobs()
        
        jobs = []
        for job_data in jobs_data:
            jobs.append(JobSummary(
                job_id=job_data["job_id"],
                status=job_data["status"],
                find_job_id=job_data["find_job_id"],
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

@app.get("/api/v1/explain/find/available")
async def list_available_find_jobs():
    """List available Find jobs that can be explained"""
    
    try:
        available_jobs = explain_service.discover_find_jobs()
        
        return {
            "available_find_jobs": available_jobs,
            "total": len(available_jobs),
            "find_results_path": str(config.find_results_dir)
        }
        
    except Exception as e:
        logger.error(f"Error listing available Find jobs: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/explain/find/{find_job_id}/preview")
async def preview_find_job(find_job_id: str, limit: int = 5):
    """Preview a Find job's features before starting explanation"""
    
    try:
        # Load Find results
        find_data = explain_service.load_find_results(find_job_id)
        
        features = find_data.get('results', [])
        if not features:
            raise HTTPException(status_code=404, detail=f"No features found in Find job {find_job_id}")
        
        # Sort by coherence score and take top features
        sorted_features = sorted(features, key=lambda x: x.get('coherence_score', 0), reverse=True)
        preview_features = sorted_features[:limit]
        
        # Create quality distribution
        high_quality = sum(1 for f in features if f.get('coherence_score', 0) >= 0.8)
        medium_quality = sum(1 for f in features if 0.5 <= f.get('coherence_score', 0) < 0.8)
        low_quality = sum(1 for f in features if f.get('coherence_score', 0) < 0.5)
        
        return {
            "find_job_id": find_job_id,
            "source_job_id": find_data.get('source_job_id', 'unknown'),
            "total_features": len(features),
            "quality_distribution": {
                "high_quality": high_quality,
                "medium_quality": medium_quality,
                "low_quality": low_quality
            },
            "preview_features": [
                {
                    "feature_id": f.get('feature_id', 'unknown'),
                    "coherence_score": f.get('coherence_score', 0),
                    "quality_level": f.get('quality_level', 'unknown'),
                    "pattern_keywords": f.get('pattern_keywords', []),
                    "sample_text": f.get('top_activations', [{}])[0].get('text', '') if f.get('top_activations') else ''
                }
                for f in preview_features
            ],
            "recommended_threshold": 0.5 if high_quality < 10 else 0.8
        }
        
    except Exception as e:
        logger.error(f"Error previewing Find job: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/explain/{job_id}/download/json")
async def download_json_results(job_id: str):
    """Download JSON results file - standardized like miStudioScore"""
    
    try:
        results_path = config.explain_results_dir / job_id / "explanation_results.json"
        
        if not results_path.exists():
            raise HTTPException(status_code=404, detail=f"Results file not found for job {job_id}")
        
        return FileResponse(
            path=str(results_path),
            filename=f"explanation_results_{job_id}.json",
            media_type="application/json"
        )
        
    except Exception as e:
        logger.error(f"Error downloading JSON results: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/explain/{job_id}/download/csv")
async def download_csv_results(job_id: str):
    """Download CSV results file - standardized like miStudioScore"""
    
    try:
        results_path = config.explain_results_dir / job_id / "explanations.csv"
        
        if not results_path.exists():
            raise HTTPException(status_code=404, detail=f"CSV results file not found for job {job_id}")
        
        return FileResponse(
            path=str(results_path),
            filename=f"explanations_{job_id}.csv",
            media_type="text/csv"
        )
        
    except Exception as e:
        logger.error(f"Error downloading CSV results: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/explain/{job_id}/download/{file_type}")
async def download_result_file(job_id: str, file_type: str):
    """Download a specific result file"""
    
    try:
        status_data = explain_service.get_job_status(job_id)
        
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

@app.get("/api/v1/explain/{job_id}/download/all")
async def download_all_results(job_id: str):
    """Download all result files as a ZIP archive"""
    
    try:
        status_data = explain_service.get_job_status(job_id)
        
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
            headers={"Content-Disposition": f"attachment; filename={job_id}_explanations.zip"}
        )
        
    except Exception as e:
        logger.error(f"Error creating ZIP download: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Legacy endpoints for compatibility
@app.get("/api/v1/config")
async def get_config():
    """Get current service configuration"""
    return {
        "service_name": config.service_name,
        "service_version": config.service_version,
        "data_path": config.data_path,
        "find_results_path": str(config.find_results_dir),
        "explain_results_path": str(config.explain_results_dir),
        "api_host": config.api_host,
        "api_port": config.api_port,
        "ollama_endpoint": config.ollama_endpoint,
        "ollama_models": config.ollama_models,
        "quality_threshold": config.default_quality_threshold,
        "max_concurrent_explanations": config.max_concurrent_explanations,
        "explanation_timeout": config.explanation_timeout
    }

@app.post("/api/v1/explain", response_model=ExplainJobResponse)
async def start_explanation_job_legacy(request: ExplainRequest):
    """Legacy endpoint - redirect to new endpoint"""
    return await start_explanation_job(request)

@app.get("/api/v1/find-jobs")
async def list_available_find_jobs_legacy():
    """Legacy endpoint - redirect to new endpoint"""
    result = await list_available_find_jobs()
    return {
        "find_jobs": result["available_find_jobs"],
        "total": result["total"],
        "find_results_directory": result["find_results_path"]
    }


# =============================================================================
# Application Entry Point
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    
    logger.info(f"🚀 Starting {config.service_name} v{config.service_version}")
    logger.info(f"📁 Data path: {config.data_path}")
    logger.info(f"🔍 Find results: {config.find_results_dir}")
    logger.info(f"💬 Explain results: {config.explain_results_dir}")
    logger.info(f"🌐 API server: {config.api_host}:{config.api_port}")
    logger.info(f"🤖 Ollama endpoint: {config.ollama_endpoint}")
    
    uvicorn.run(
        app,
        host=config.api_host,
        port=config.api_port,
        log_level=config.log_level.lower(),
        access_log=True
    )