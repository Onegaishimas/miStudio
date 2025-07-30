# src/main.py - Unified Configuration Version for miStudioExplain
"""
Main FastAPI application for miStudioExplain service - Standardized Implementation.
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
from fastapi.responses import JSONResponse, FileResponse
from contextlib import asynccontextmanager
from datetime import datetime
from pydantic import BaseModel, Field
import asyncio

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
# Unified Configuration - Same Pattern as Other Services
# =============================================================================

class ServiceConfig:
    """Unified configuration for miStudioExplain - Environment-first approach"""
    
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
        self.default_quality_threshold = float(os.getenv("QUALITY_THRESHOLD", "0.4"))
        self.high_quality_threshold = float(os.getenv("HIGH_QUALITY_THRESHOLD", "0.6"))
        self.excellent_threshold = float(os.getenv("EXCELLENT_THRESHOLD", "0.8"))
        self.max_batch_size = int(os.getenv("MAX_BATCH_SIZE", "10"))
        self.max_concurrent_explanations = int(os.getenv("MAX_CONCURRENT_EXPLANATIONS", "4"))
        self.explanation_timeout = int(os.getenv("EXPLANATION_TIMEOUT", "120"))
        
        # Storage configuration
        self.cache_path = self.data_path_obj / "cache"
        self.input_path = self.data_path_obj / "input"
        self.output_path = self.data_path_obj / "output" / "explain"
        self.logs_path = self.data_path_obj / "logs" / "explain"
        self.max_cache_size_gb = int(os.getenv("MAX_CACHE_SIZE_GB", "10"))
        
        # Create subdirectories
        self._ensure_directories()
    
    def _ensure_directories(self):
        """Ensure all required directories exist"""
        directories = [
            self.data_path_obj / "results" / "find",
            self.data_path_obj / "results" / "explain", 
            self.cache_path,
            self.input_path,
            self.output_path,
            self.logs_path
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    @property
    def find_results_dir(self) -> Path:
        """Directory where miStudioFind results are stored"""
        return self.data_path_obj / "results" / "find"
    
    @property
    def explain_results_dir(self) -> Path:
        """Directory where miStudioExplain results are stored"""
        return self.data_path_obj / "results" / "explain"


# Global configuration instance
config = ServiceConfig()


# =============================================================================
# API Models - Standardized with Other Services
# =============================================================================

class FindResultInput(BaseModel):
    """Input from miStudioFind results"""
    find_job_id: str = Field(description="The miStudioFind job ID to explain")
    feature_filter: Optional[Dict[str, Any]] = Field(default=None, description="Optional filters for features")
    quality_threshold: Optional[float] = Field(default=None, description="Minimum quality threshold for features")


class ExplainRequest(BaseModel):
    """Request model for explanation generation"""
    find_job_id: str = Field(description="Source miStudioFind job ID")
    analysis_type: str = Field(default="behavioral_patterns", description="Type of analysis to perform")
    complexity: str = Field(default="medium", description="Explanation complexity level")
    model: Optional[str] = Field(default=None, description="Specific LLM model to use")
    quality_threshold: float = Field(default=0.5, description="Minimum feature quality threshold")
    max_features: int = Field(default=20, description="Maximum number of features to explain")


class ExplainJobResponse(BaseModel):
    """Response model for starting explanation job"""
    job_id: str
    status: str
    message: str
    find_job_id: str
    parameters: Dict[str, Any]
    timestamp: str
    next_steps: Dict[str, str]


class JobStatusResponse(BaseModel):
    """Response model for job status"""
    job_id: str
    status: str
    progress: float
    message: str
    start_time: Optional[str] = None
    estimated_time_remaining: Optional[int] = None
    features_processed: Optional[int] = None
    total_features: Optional[int] = None


class JobResultResponse(BaseModel):
    """Response model for job results"""
    job_id: str
    status: str
    results_path: Optional[str] = None
    explanation_count: Optional[int] = None
    processing_time: Optional[float] = None
    quality_score: Optional[float] = None


class HealthResponse(BaseModel):
    """Health check response model"""
    status: str
    service: str
    version: str
    data_path: str
    timestamp: str
    components: Dict[str, bool]
    ollama_status: Dict[str, Any]


# =============================================================================
# Service Initialization with Fallback Handling
# =============================================================================

# Import core modules with fallback handling
ollama_manager = None
input_manager = None
feature_prioritizer = None
context_builder = None
explanation_generator = None
quality_validator = None
result_manager = None

# Simple job tracking for basic functionality
active_jobs: Dict[str, Dict[str, Any]] = {}
completed_jobs: Dict[str, Dict[str, Any]] = {}

def generate_job_id() -> str:
    """Generate unique job ID"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    import uuid
    unique_id = str(uuid.uuid4())[:8]
    return f"explain_{timestamp}_{unique_id}"

try:
    from infrastructure.ollama_manager import OllamaManager
    ollama_manager = OllamaManager(ollama_endpoint=config.ollama_endpoint)
    logger.info("✅ Successfully initialized Ollama manager")
except ImportError as e:
    logger.warning(f"Ollama manager not available: {e}")

try:
    from core.input_manager import InputManager
    input_manager = InputManager()
    logger.info("✅ Successfully initialized input manager")
except ImportError as e:
    logger.warning(f"Input manager not available: {e}")

try:
    from core.feature_prioritizer import FeaturePrioritizer
    feature_prioritizer = FeaturePrioritizer()
    logger.info("✅ Successfully initialized feature prioritizer")
except ImportError as e:
    logger.warning(f"Feature prioritizer not available: {e}")

try:
    from core.context_builder import ContextBuilder
    context_builder = ContextBuilder()
    logger.info("✅ Successfully initialized context builder")
except ImportError as e:
    logger.warning(f"Context builder not available: {e}")

try:
    from .core.explanation_generator import ExplanationGenerator
    if ollama_manager:
        explanation_generator = ExplanationGenerator(ollama_manager=ollama_manager)
        logger.info("✅ Successfully initialized explanation generator")
except ImportError as e:
    logger.warning(f"Explanation generator not available: {e}")

try:
    from core.quality_validator import QualityValidator
    quality_validator = QualityValidator()
    logger.info("✅ Successfully initialized quality validator")
except ImportError as e:
    logger.warning(f"Quality validator not available: {e}")

try:
    from core.result_manager import ResultManager
    result_manager = ResultManager(output_directory=str(config.output_path))
    logger.info("✅ Successfully initialized result manager")
except ImportError as e:
    logger.warning(f"Result manager not available: {e}")


# =============================================================================
# FastAPI Application Setup
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan - startup and shutdown."""
    logger.info(f"🚀 {config.service_name} API starting up...")
    logger.info(f"Data path: {config.data_path}")
    logger.info(f"Ollama endpoint: {config.ollama_endpoint}")
    logger.info(f"Service version: {config.service_version}")
    
    # Initialize Ollama if available
    if ollama_manager:
        try:
            await ollama_manager.initialize()
            logger.info("✅ Ollama manager initialized")
        except Exception as e:
            logger.warning(f"Ollama initialization failed: {e}")
    
    logger.info(f"✅ Data path accessible")
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
    description="Natural language explanation generation service for AI features",
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
# API Endpoints
# =============================================================================

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    
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
        timestamp=datetime.now().isoformat(),
        components={
            "ollama_manager": ollama_manager is not None,
            "input_manager": input_manager is not None,
            "feature_prioritizer": feature_prioritizer is not None,
            "context_builder": context_builder is not None,
            "explanation_generator": explanation_generator is not None,
            "quality_validator": quality_validator is not None,
            "result_manager": result_manager is not None
        },
        ollama_status=ollama_status
    )


@app.get("/api/v1/config")
async def get_config():
    """Get current service configuration"""
    return {
        "service_name": config.service_name,
        "service_version": config.service_version,
        "data_path": config.data_path,
        "api_host": config.api_host,
        "api_port": config.api_port,
        "ollama_endpoint": config.ollama_endpoint,
        "ollama_models": config.ollama_models,
        "quality_threshold": config.default_quality_threshold,
        "max_concurrent_explanations": config.max_concurrent_explanations,
        "explanation_timeout": config.explanation_timeout
    }


@app.post("/api/v1/explain", response_model=ExplainJobResponse)
async def start_explanation_job(request: ExplainRequest, background_tasks: BackgroundTasks):
    """Start explanation generation job"""
    
    try:
        # Validate that Find results exist
        find_results_path = config.find_results_dir / f"{request.find_job_id}_results.json"
        if not find_results_path.exists():
            # Try alternative paths
            alternative_paths = [
                config.data_path_obj / "results" / "find" / f"{request.find_job_id}.json",
                config.data_path_obj / f"{request.find_job_id}_results.json"
            ]
            
            found_path = None
            for alt_path in alternative_paths:
                if alt_path.exists():
                    found_path = alt_path
                    break
            
            if not found_path:
                raise HTTPException(
                    status_code=404,
                    detail=f"Find results not found for job {request.find_job_id}. Expected at: {find_results_path}"
                )
            find_results_path = found_path
        
        # Generate job ID
        job_id = generate_job_id()
        
        # Create job entry
        job_info = {
            "job_id": job_id,
            "find_job_id": request.find_job_id,
            "status": "queued",
            "start_time": datetime.now().timestamp(),
            "progress": 0.0,
            "features_processed": 0,
            "total_features": 0,
            "parameters": request.model_dump(),
            "results": None,
            "error": None
        }
        
        active_jobs[job_id] = job_info
        
        # Log job creation
        logger.info(f"🔍 Started explanation job: {job_id}")
        logger.info(f"📊 Processing Find job: {request.find_job_id}")
        logger.info(f"⚙️  Parameters: model={request.model}, complexity={request.complexity}, quality_threshold={request.quality_threshold}")
        
        # Start background processing
        background_tasks.add_task(_process_explanation_job, job_id, request, find_results_path)
        
        return ExplainJobResponse(
            job_id=job_id,
            status="queued",
            message=f"Explanation job '{job_id}' started for Find job {request.find_job_id}",
            find_job_id=request.find_job_id,
            parameters={
                "analysis_type": request.analysis_type,
                "complexity": request.complexity,
                "model": request.model,
                "quality_threshold": request.quality_threshold,
                "max_features": request.max_features
            },
            timestamp=datetime.now().isoformat(),
            next_steps={
                "check_status": f"/api/v1/explain/{job_id}/status",
                "get_results": f"/api/v1/explain/{job_id}/results"
            }
        )
        
    except Exception as e:
        logger.error(f"Error starting explanation job: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


async def _process_explanation_job(job_id: str, request: ExplainRequest, find_results_path: Path):
    """Background task to process explanation job"""
    job_info = active_jobs[job_id]
    
    try:
        job_info["status"] = "running"
        logger.info(f"Processing explanation job {job_id}...")
        
        # Load Find results
        with open(find_results_path, 'r') as f:
            find_results = json.load(f)
        
        # Simulate processing if core modules aren't available
        if not all([input_manager, explanation_generator, ollama_manager]):
            logger.warning("Core modules not available, simulating explanation generation...")
            
            # Simulate processing time
            await asyncio.sleep(5)
            
            # Create mock explanation
            job_info["total_features"] = request.max_features
            job_info["features_processed"] = request.max_features
            job_info["progress"] = 1.0
            job_info["status"] = "completed"
            job_info["completion_time"] = datetime.now().timestamp()
            job_info["processing_time"] = job_info["completion_time"] - job_info["start_time"]
            
            mock_results = {
                "explanations": [
                    {
                        "feature_id": f"feature_{i}",
                        "explanation": f"This feature appears to detect patterns related to {request.analysis_type}. (Mock explanation - core modules not available)",
                        "confidence": 0.8,
                        "quality_score": 0.7
                    }
                    for i in range(min(5, request.max_features))
                ],
                "summary": f"Generated {min(5, request.max_features)} mock explanations for {request.analysis_type} analysis",
                "model_used": request.model or "mock-model",
                "total_processing_time": job_info["processing_time"]
            }
            
            job_info["results"] = mock_results
            
        else:
            # Use actual core modules for processing
            logger.info("Using core modules for explanation generation...")
            
            # Process with actual modules
            # (This would be the full implementation using the core modules)
            raise NotImplementedError("Full core module integration not yet implemented")
        
        # Move to completed jobs
        completed_jobs[job_id] = job_info
        del active_jobs[job_id]
        
        logger.info(f"Job {job_id} completed successfully. Generated {len(job_info['results']['explanations'])} explanations.")
        
    except Exception as e:
        logger.error(f"Job {job_id} failed: {e}")
        job_info["status"] = "failed"
        job_info["error"] = str(e)
        
        # Move to completed jobs even if failed
        completed_jobs[job_id] = job_info
        if job_id in active_jobs:
            del active_jobs[job_id]


@app.get("/api/v1/explain/{job_id}/status", response_model=JobStatusResponse)
async def get_job_status(job_id: str):
    """Get status of explanation job"""
    
    # Check active jobs
    if job_id in active_jobs:
        job_info = active_jobs[job_id]
    elif job_id in completed_jobs:
        job_info = completed_jobs[job_id]
    else:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    
    estimated_time = None
    if job_info["status"] == "running" and job_info["features_processed"] > 0:
        elapsed = datetime.now().timestamp() - job_info["start_time"]
        time_per_feature = elapsed / job_info["features_processed"]
        remaining_features = job_info["total_features"] - job_info["features_processed"]
        estimated_time = int(remaining_features * time_per_feature)
    
    return JobStatusResponse(
        job_id=job_id,
        status=job_info["status"],
        progress=job_info["progress"],
        message=f"Processing {job_info['features_processed']}/{job_info['total_features']} features",
        start_time=datetime.fromtimestamp(job_info["start_time"]).isoformat(),
        estimated_time_remaining=estimated_time,
        features_processed=job_info["features_processed"],
        total_features=job_info["total_features"]
    )


@app.get("/api/v1/explain/{job_id}/results", response_model=JobResultResponse)
async def get_job_results(job_id: str):
    """Get results of completed explanation job"""
    
    if job_id in completed_jobs:
        job_info = completed_jobs[job_id]
    else:
        # Check if job exists but isn't completed
        if job_id in active_jobs:
            job_info = active_jobs[job_id]
            raise HTTPException(
                status_code=409,
                detail=f"Job {job_id} is not completed yet. Current status: {job_info['status']}"
            )
        else:
            raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    
    if job_info["status"] != "completed":
        raise HTTPException(
            status_code=409,
            detail=f"Job {job_id} did not complete successfully. Status: {job_info['status']}"
        )
    
    results = job_info.get("results", {})
    explanations = results.get("explanations", [])
    
    # Calculate quality score
    quality_scores = [exp.get("quality_score", 0.0) for exp in explanations]
    avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0.0
    
    return JobResultResponse(
        job_id=job_id,
        status=job_info["status"],
        results_path=f"/api/v1/explain/{job_id}/download/json",
        explanation_count=len(explanations),
        processing_time=job_info.get("processing_time", 0),
        quality_score=avg_quality
    )


@app.get("/api/v1/jobs")
async def list_jobs():
    """List all explanation jobs"""
    
    all_jobs = []
    
    # Add active jobs
    for job_id, job_info in active_jobs.items():
        all_jobs.append({
            "job_id": job_id,
            "status": job_info["status"],
            "find_job_id": job_info["find_job_id"],
            "created_at": datetime.fromtimestamp(job_info["start_time"]).isoformat(),
            "completed_at": None
        })
    
    # Add completed jobs
    for job_id, job_info in completed_jobs.items():
        completed_time = None
        if job_info.get("completion_time"):
            completed_time = datetime.fromtimestamp(job_info["completion_time"]).isoformat()
        
        all_jobs.append({
            "job_id": job_id,
            "status": job_info["status"],
            "find_job_id": job_info["find_job_id"],
            "created_at": datetime.fromtimestamp(job_info["start_time"]).isoformat(),
            "completed_at": completed_time
        })
    
    return {"jobs": all_jobs, "total": len(all_jobs)}


@app.get("/api/v1/find-jobs")
async def list_available_find_jobs():
    """List available miStudioFind jobs that can be explained"""
    
    try:
        find_jobs = []
        find_results_dir = config.find_results_dir
        
        if find_results_dir.exists():
            for result_file in find_results_dir.glob("*_results.json"):
                try:
                    with open(result_file, 'r') as f:
                        find_data = json.load(f)
                    
                    job_id = result_file.stem.replace("_results", "")
                    find_jobs.append({
                        "find_job_id": job_id,
                        "result_file": str(result_file),
                        "feature_count": len(find_data.get("results", [])),
                        "ready_for_explanation": True
                    })
                except Exception as e:
                    logger.warning(f"Error reading find result file {result_file}: {e}")
        
        return {
            "find_jobs": find_jobs,
            "total": len(find_jobs),
            "find_results_directory": str(find_results_dir)
        }
    except Exception as e:
        logger.error(f"Error listing find jobs: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# Application Entry Point
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    
    logger.info(f"🚀 Starting {config.service_name} on {config.api_host}:{config.api_port}")
    logger.info(f"📂 Data path: {config.data_path}")
    logger.info(f"🔗 Points to: {config.data_path_obj.resolve()}")
    logger.info(f"🤖 Ollama endpoint: {config.ollama_endpoint}")
    
    uvicorn.run(
        app, 
        host=config.api_host, 
        port=config.api_port, 
        log_level=config.log_level.lower(),
        access_log=True
    )