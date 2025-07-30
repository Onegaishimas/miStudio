# src/main.py - Unified Configuration Version with Fixed Response Models
"""
Main FastAPI application for miStudioFind service - Standardized Implementation.
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
from fastapi.responses import StreamingResponse, FileResponse, JSONResponse
from contextlib import asynccontextmanager
from datetime import datetime
from pydantic import BaseModel, Field
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
# Unified Configuration - Same Pattern as miStudioTrain
# =============================================================================

class ServiceConfig:
    """Unified configuration for miStudioFind - Environment-first approach"""
    
    def __init__(self):
        # Primary data path - same pattern for all services
        self.data_path = os.getenv("DATA_PATH", "/data")
        self.data_path_obj = Path(self.data_path)
        self.data_path_obj.mkdir(parents=True, exist_ok=True)
        
        # Service metadata
        self.service_name = "miStudioFind"
        self.service_version = "1.0.0"
        
        # API configuration
        self.api_host = os.getenv("API_HOST", "0.0.0.0")
        self.api_port = int(os.getenv("API_PORT", "8002"))
        self.log_level = os.getenv("LOG_LEVEL", "INFO")
        
        # Feature analysis configuration
        self.top_k_selections = int(os.getenv("TOP_K_SELECTIONS", "20"))
        self.coherence_threshold = float(os.getenv("COHERENCE_THRESHOLD", "0.5"))
        self.batch_size = int(os.getenv("BATCH_SIZE", "100"))
        self.memory_optimization = os.getenv("MEMORY_OPTIMIZATION", "true").lower() == "true"
        
        # Performance configuration
        self.max_concurrent_jobs = int(os.getenv("MAX_CONCURRENT_JOBS", "4"))
        self.processing_timeout_minutes = int(os.getenv("PROCESSING_TIMEOUT_MINUTES", "60"))
        self.memory_limit_gb = float(os.getenv("MEMORY_LIMIT_GB", "8.0"))
        
        # Output configuration
        self.save_intermediate_results = os.getenv("SAVE_INTERMEDIATE_RESULTS", "true").lower() == "true"
        self.compress_outputs = os.getenv("COMPRESS_OUTPUTS", "false").lower() == "true"
        self.feature_preview_count = int(os.getenv("FEATURE_PREVIEW_COUNT", "50"))
        
        # Create subdirectories
        self._ensure_directories()
    
    def _ensure_directories(self):
        """Ensure all required directories exist"""
        directories = [
            self.data_path_obj / "models",
            self.data_path_obj / "activations",
            self.data_path_obj / "results" / "find",
            self.data_path_obj / "cache",
            self.data_path_obj / "logs" / "find"
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    @property
    def models_dir(self) -> Path:
        return self.data_path_obj / "models"
    
    @property
    def activations_dir(self) -> Path:
        return self.data_path_obj / "activations"
    
    @property
    def results_dir(self) -> Path:
        return self.data_path_obj / "results" / "find"
    
    @property
    def cache_dir(self) -> Path:
        return self.data_path_obj / "cache"
    
    @property
    def logs_dir(self) -> Path:
        return self.data_path_obj / "logs" / "find"


# Global configuration instance
config = ServiceConfig()


# =============================================================================
# API Models - Fixed Response Models
# =============================================================================

class FindRequest(BaseModel):
    """Request model for feature analysis"""
    source_job_id: str = Field(description="Source training job ID from miStudioTrain")
    top_k: int = Field(default=20, description="Number of top activations per feature")
    coherence_threshold: float = Field(default=0.5, description="Minimum coherence threshold for features")


class AnalysisParameters(BaseModel):
    """Analysis parameters model"""
    top_k: int
    coherence_threshold: float


class NextSteps(BaseModel):
    """Next steps model"""
    check_status: str
    get_results: str


class FindJobResponse(BaseModel):
    """Response model for starting feature analysis"""
    job_id: str
    status: str
    message: str
    source_job_id: str
    parameters: AnalysisParameters
    timestamp: str
    next_steps: NextSteps


class ComponentStatus(BaseModel):
    """Component status model"""
    processing_service: bool
    enhanced_persistence: bool
    advanced_filter: bool


class HealthResponse(BaseModel):
    """Health check response model"""
    status: str
    service: str
    version: str
    data_path: str
    timestamp: str
    components: ComponentStatus


class ConfigResponse(BaseModel):
    """Configuration response model"""
    service_name: str
    service_version: str
    data_path: str
    api_host: str
    api_port: int
    top_k_selections: int
    coherence_threshold: float
    max_concurrent_jobs: int
    processing_timeout_minutes: int
    memory_optimization: bool


class JobStatusResponse(BaseModel):
    """Response model for job status"""
    job_id: str
    status: str
    progress: float
    message: str
    start_time: Optional[str] = None
    estimated_time_remaining: Optional[int] = None


class JobResultResponse(BaseModel):
    """Response model for job results"""
    job_id: str
    status: str
    results_path: Optional[str] = None
    feature_count: Optional[int] = None
    processing_time: Optional[float] = None


class JobSummary(BaseModel):
    """Job summary model"""
    job_id: str
    status: str
    source_job_id: str
    created_at: str
    completed_at: Optional[str] = None


class JobListResponse(BaseModel):
    """Response model for job listing"""
    jobs: List[JobSummary]
    total: int


class CancelJobResponse(BaseModel):
    """Response model for job cancellation"""
    message: str


# =============================================================================
# Service Initialization with Fallback Handling
# =============================================================================

# Import core modules with fallback handling
processing_service = None
enhanced_persistence = None
advanced_filter = None

try:
    from core.simple_processing_service import SimpleProcessingService
    processing_service = SimpleProcessingService(config.data_path)
    logger.info("✅ Successfully initialized processing service")
except ImportError as e:
    logger.error(f"Failed to import SimpleProcessingService: {e}")
    processing_service = None

try:
    from core.result_persistence import EnhancedResultPersistence
    enhanced_persistence = EnhancedResultPersistence(config.data_path)
    logger.info("✅ Successfully initialized enhanced persistence")
except ImportError as e:
    logger.warning(f"Enhanced persistence not available: {e}")
    enhanced_persistence = None

try:
    from core.advanced_filtering import AdvancedFeatureFilter, PatternCategory, QualityTier
    advanced_filter = AdvancedFeatureFilter()
    logger.info("✅ Successfully initialized advanced filter")
except ImportError as e:
    logger.warning(f"Advanced filter not available: {e}")
    advanced_filter = None
    # Create dummy classes for fallback
    class PatternCategory:
        pass
    class QualityTier:
        pass


# =============================================================================
# FastAPI Application Setup
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan - startup and shutdown."""
    logger.info(f"🚀 {config.service_name} API starting up...")
    logger.info(f"Data path: {config.data_path}")
    logger.info(f"Service version: {config.service_version}")
    logger.info(f"✅ Data path accessible")
    yield
    logger.info(f"🛑 {config.service_name} API shutting down...")


app = FastAPI(
    title="miStudioFind API",
    description="Feature analysis and discovery service for sparse autoencoders",
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
# API Endpoints with Proper Response Models
# =============================================================================

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        service=config.service_name,
        version=config.service_version,
        data_path=config.data_path,
        timestamp=datetime.now().isoformat(),
        components=ComponentStatus(
            processing_service=processing_service is not None,
            enhanced_persistence=enhanced_persistence is not None,
            advanced_filter=advanced_filter is not None
        )
    )


@app.get("/api/v1/config", response_model=ConfigResponse)
async def get_config():
    """Get current service configuration"""
    return ConfigResponse(
        service_name=config.service_name,
        service_version=config.service_version,
        data_path=config.data_path,
        api_host=config.api_host,
        api_port=config.api_port,
        top_k_selections=config.top_k_selections,
        coherence_threshold=config.coherence_threshold,
        max_concurrent_jobs=config.max_concurrent_jobs,
        processing_timeout_minutes=config.processing_timeout_minutes,
        memory_optimization=config.memory_optimization
    )


@app.post("/api/v1/find", response_model=FindJobResponse)
async def start_feature_analysis(request: FindRequest, background_tasks: BackgroundTasks):
    """Start feature analysis job"""
    
    if processing_service is None:
        raise HTTPException(
            status_code=503, 
            detail="Processing service not available"
        )
    
    try:
        # Validate source job exists
        source_models_dir = config.models_dir / request.source_job_id
        source_activations_dir = config.activations_dir / request.source_job_id
        
        if not source_models_dir.exists() or not source_activations_dir.exists():
            raise HTTPException(
                status_code=404,
                detail=f"Source job {request.source_job_id} not found. Expected directories: {source_models_dir}, {source_activations_dir}"
            )
        
        # Check for required files
        sae_model_path = source_models_dir / "sae_model.pt"
        feature_activations_path = source_activations_dir / "feature_activations.pt"
        metadata_path = source_activations_dir / "metadata.json"
        
        missing_files = []
        for path in [sae_model_path, feature_activations_path, metadata_path]:
            if not path.exists():
                missing_files.append(str(path))
        
        if missing_files:
            raise HTTPException(
                status_code=400,
                detail=f"Missing required files: {missing_files}"
            )
        
        # Start analysis job
        job_id = await processing_service.start_analysis_job(
            source_job_id=request.source_job_id,
            top_k=request.top_k
        )
        
        return FindJobResponse(
            job_id=job_id,
            status="queued",
            message=f"Feature analysis started for {request.source_job_id}",
            source_job_id=request.source_job_id,
            parameters=AnalysisParameters(
                top_k=request.top_k,
                coherence_threshold=request.coherence_threshold
            ),
            timestamp=datetime.now().isoformat(),
            next_steps=NextSteps(
                check_status=f"/api/v1/find/{job_id}/status",
                get_results=f"/api/v1/find/{job_id}/results"
            )
        )
        
    except Exception as e:
        logger.error(f"Error starting feature analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/find/{job_id}/status", response_model=JobStatusResponse)
async def get_job_status(job_id: str):
    """Get status of feature analysis job"""
    
    if processing_service is None:
        raise HTTPException(status_code=503, detail="Processing service not available")
    
    try:
        # Call non-async method from SimpleProcessingService
        status_data = processing_service.get_job_status(job_id)
        
        if status_data is None:
            raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
        
        # Convert SimpleProcessingService format to our response model
        return JobStatusResponse(
            job_id=job_id,
            status=status_data.get("status", "unknown"),
            progress=status_data.get("progress", {}).get("features_processed", 0) / max(1, status_data.get("progress", {}).get("total_features", 1)),
            message=f"Processing {status_data.get('progress', {}).get('features_processed', 0)}/{status_data.get('progress', {}).get('total_features', 0)} features",
            start_time=datetime.fromtimestamp(status_data.get("start_time", 0)).isoformat() if status_data.get("start_time") else None,
            estimated_time_remaining=status_data.get("progress", {}).get("estimated_time_remaining")
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting job status: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/find/{job_id}/results", response_model=JobResultResponse)
async def get_job_results(job_id: str):
    """Get results of completed feature analysis job"""
    
    if processing_service is None:
        raise HTTPException(status_code=503, detail="Processing service not available")
    
    try:
        # Call non-async method from SimpleProcessingService
        results_data = processing_service.get_job_results(job_id)
        
        if results_data is None:
            # Check if job exists but isn't completed
            status_data = processing_service.get_job_status(job_id)
            if status_data:
                raise HTTPException(
                    status_code=409,
                    detail=f"Job {job_id} is not completed yet. Current status: {status_data.get('status', 'unknown')}"
                )
            else:
                raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
        
        # Convert SimpleProcessingService format to our response model
        return JobResultResponse(
            job_id=job_id,
            status=results_data.get("status", "unknown"),
            results_path=f"/api/v1/find/{job_id}/download/json",  # Provide download path
            feature_count=len(results_data.get("results", [])),
            processing_time=results_data.get("processing_time", 0)
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting job results: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/jobs", response_model=JobListResponse)
async def list_jobs():
    """List all analysis jobs"""
    
    if processing_service is None:
        raise HTTPException(status_code=503, detail="Processing service not available")
    
    try:
        # Get jobs from both active and completed dictionaries
        job_summaries = []
        
        # Add active jobs
        for job_id, job_data in processing_service.active_jobs.items():
            job_summaries.append(JobSummary(
                job_id=job_id,
                status=job_data.get("status", "unknown"),
                source_job_id=job_data.get("source_job_id", ""),
                created_at=datetime.fromtimestamp(job_data.get("start_time", 0)).isoformat(),
                completed_at=None
            ))
        
        # Add completed jobs
        for job_id, job_data in processing_service.completed_jobs.items():
            completed_time = None
            if job_data.get("completion_time"):
                completed_time = datetime.fromtimestamp(job_data["completion_time"]).isoformat()
            
            job_summaries.append(JobSummary(
                job_id=job_id,
                status=job_data.get("status", "unknown"),
                source_job_id=job_data.get("source_job_id", ""),
                created_at=datetime.fromtimestamp(job_data.get("start_time", 0)).isoformat(),
                completed_at=completed_time
            ))
        
        return JobListResponse(jobs=job_summaries, total=len(job_summaries))
    except Exception as e:
        logger.error(f"Error listing jobs: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/v1/find/{job_id}", response_model=CancelJobResponse)
async def cancel_job(job_id: str):
    """Cancel a running analysis job"""
    
    if processing_service is None:
        raise HTTPException(status_code=503, detail="Processing service not available")
    
    try:
        success = await processing_service.cancel_job(job_id)
        if success:
            return CancelJobResponse(message=f"Job {job_id} cancelled successfully")
        else:
            raise HTTPException(status_code=400, detail=f"Job {job_id} cannot be cancelled")
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    except Exception as e:
        logger.error(f"Error cancelling job: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# File Download Endpoints (if enhanced_persistence is available)
# =============================================================================

@app.get("/api/v1/find/{job_id}/download/{format}")
async def download_results(job_id: str, format: str):
    """Download analysis results in specified format"""
    
    if enhanced_persistence is None:
        raise HTTPException(status_code=503, detail="Enhanced persistence not available")
    
    try:
        file_path = await enhanced_persistence.get_result_file(job_id, format)
        if not file_path.exists():
            raise HTTPException(status_code=404, detail=f"Results not found for job {job_id} in format {format}")
        
        return FileResponse(
            path=str(file_path),
            filename=f"{job_id}_results.{format}",
            media_type="application/octet-stream"
        )
    except Exception as e:
        logger.error(f"Error downloading results: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# Additional Utility Endpoints
# =============================================================================

@app.get("/api/v1/source-jobs")
async def list_source_jobs():
    """List available training jobs that can be analyzed"""
    try:
        models_dir = config.models_dir
        activations_dir = config.activations_dir
        
        source_jobs = []
        
        if models_dir.exists():
            for job_dir in models_dir.iterdir():
                if job_dir.is_dir() and job_dir.name.startswith("train_"):
                    # Check if corresponding activations exist
                    activation_dir = activations_dir / job_dir.name
                    sae_model_path = job_dir / "sae_model.pt"
                    
                    if activation_dir.exists() and sae_model_path.exists():
                        source_jobs.append({
                            "job_id": job_dir.name,
                            "model_path": str(sae_model_path),
                            "activations_path": str(activation_dir),
                            "ready_for_analysis": True
                        })
        
        return {
            "source_jobs": source_jobs,
            "total": len(source_jobs),
            "data_path": str(config.data_path)
        }
    except Exception as e:
        logger.error(f"Error listing source jobs: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/stats")
async def get_service_stats():
    """Get service statistics"""
    try:
        stats = {
            "service": config.service_name,
            "version": config.service_version,
            "uptime_seconds": 0,  # Could be calculated if tracking start time
            "total_jobs_processed": 0,  # Would need to track this
            "active_jobs": len(processing_service.active_jobs) if processing_service else 0,
            "data_path": config.data_path,
            "disk_usage": {
                "models_dir": str(config.models_dir),
                "activations_dir": str(config.activations_dir),
                "results_dir": str(config.results_dir)
            }
        }
        return stats
    except Exception as e:
        logger.error(f"Error getting service stats: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# Application Entry Point
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    
    logger.info(f"🚀 Starting {config.service_name} on {config.api_host}:{config.api_port}")
    logger.info(f"📂 Data path: {config.data_path}")
    logger.info(f"🔗 Points to: {config.data_path_obj.resolve()}")
    
    uvicorn.run(
        app, 
        host=config.api_host, 
        port=config.api_port, 
        log_level=config.log_level.lower(),
        access_log=True
    )
# src/main.py - Unified Configuration Version with Fixed Response Models
"""
Main FastAPI application for miStudioFind service - Standardized Implementation.
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
from fastapi.responses import StreamingResponse, FileResponse, JSONResponse
from contextlib import asynccontextmanager
from datetime import datetime
from pydantic import BaseModel, Field
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
# Unified Configuration - Same Pattern as miStudioTrain
# =============================================================================

class ServiceConfig:
    """Unified configuration for miStudioFind - Environment-first approach"""
    
    def __init__(self):
        # Primary data path - same pattern for all services
        self.data_path = os.getenv("DATA_PATH", "/data")
        self.data_path_obj = Path(self.data_path)
        self.data_path_obj.mkdir(parents=True, exist_ok=True)
        
        # Service metadata
        self.service_name = "miStudioFind"
        self.service_version = "1.0.0"
        
        # API configuration
        self.api_host = os.getenv("API_HOST", "0.0.0.0")
        self.api_port = int(os.getenv("API_PORT", "8002"))
        self.log_level = os.getenv("LOG_LEVEL", "INFO")
        
        # Feature analysis configuration
        self.top_k_selections = int(os.getenv("TOP_K_SELECTIONS", "20"))
        self.coherence_threshold = float(os.getenv("COHERENCE_THRESHOLD", "0.5"))
        self.batch_size = int(os.getenv("BATCH_SIZE", "100"))
        self.memory_optimization = os.getenv("MEMORY_OPTIMIZATION", "true").lower() == "true"
        
        # Performance configuration
        self.max_concurrent_jobs = int(os.getenv("MAX_CONCURRENT_JOBS", "4"))
        self.processing_timeout_minutes = int(os.getenv("PROCESSING_TIMEOUT_MINUTES", "60"))
        self.memory_limit_gb = float(os.getenv("MEMORY_LIMIT_GB", "8.0"))
        
        # Output configuration
        self.save_intermediate_results = os.getenv("SAVE_INTERMEDIATE_RESULTS", "true").lower() == "true"
        self.compress_outputs = os.getenv("COMPRESS_OUTPUTS", "false").lower() == "true"
        self.feature_preview_count = int(os.getenv("FEATURE_PREVIEW_COUNT", "50"))
        
        # Create subdirectories
        self._ensure_directories()
    
    def _ensure_directories(self):
        """Ensure all required directories exist"""
        directories = [
            self.data_path_obj / "models",
            self.data_path_obj / "activations",
            self.data_path_obj / "results" / "find",
            self.data_path_obj / "cache",
            self.data_path_obj / "logs" / "find"
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    @property
    def models_dir(self) -> Path:
        return self.data_path_obj / "models"
    
    @property
    def activations_dir(self) -> Path:
        return self.data_path_obj / "activations"
    
    @property
    def results_dir(self) -> Path:
        return self.data_path_obj / "results" / "find"
    
    @property
    def cache_dir(self) -> Path:
        return self.data_path_obj / "cache"
    
    @property
    def logs_dir(self) -> Path:
        return self.data_path_obj / "logs" / "find"


# Global configuration instance
config = ServiceConfig()


# =============================================================================
# API Models - Fixed Response Models
# =============================================================================

class FindRequest(BaseModel):
    """Request model for feature analysis"""
    source_job_id: str = Field(description="Source training job ID from miStudioTrain")
    top_k: int = Field(default=20, description="Number of top activations per feature")
    coherence_threshold: float = Field(default=0.5, description="Minimum coherence threshold for features")


class AnalysisParameters(BaseModel):
    """Analysis parameters model"""
    top_k: int
    coherence_threshold: float


class NextSteps(BaseModel):
    """Next steps model"""
    check_status: str
    get_results: str


class FindJobResponse(BaseModel):
    """Response model for starting feature analysis"""
    job_id: str
    status: str
    message: str
    source_job_id: str
    parameters: AnalysisParameters
    timestamp: str
    next_steps: NextSteps


class ComponentStatus(BaseModel):
    """Component status model"""
    processing_service: bool
    enhanced_persistence: bool
    advanced_filter: bool


class HealthResponse(BaseModel):
    """Health check response model"""
    status: str
    service: str
    version: str
    data_path: str
    timestamp: str
    components: ComponentStatus


class ConfigResponse(BaseModel):
    """Configuration response model"""
    service_name: str
    service_version: str
    data_path: str
    api_host: str
    api_port: int
    top_k_selections: int
    coherence_threshold: float
    max_concurrent_jobs: int
    processing_timeout_minutes: int
    memory_optimization: bool


class JobStatusResponse(BaseModel):
    """Response model for job status"""
    job_id: str
    status: str
    progress: float
    message: str
    start_time: Optional[str] = None
    estimated_time_remaining: Optional[int] = None


class JobResultResponse(BaseModel):
    """Response model for job results"""
    job_id: str
    status: str
    results_path: Optional[str] = None
    feature_count: Optional[int] = None
    processing_time: Optional[float] = None


class JobSummary(BaseModel):
    """Job summary model"""
    job_id: str
    status: str
    source_job_id: str
    created_at: str
    completed_at: Optional[str] = None


class JobListResponse(BaseModel):
    """Response model for job listing"""
    jobs: List[JobSummary]
    total: int


class CancelJobResponse(BaseModel):
    """Response model for job cancellation"""
    message: str


# =============================================================================
# Service Initialization with Fallback Handling
# =============================================================================

# Import core modules with fallback handling
processing_service = None
enhanced_persistence = None
advanced_filter = None

try:
    from core.simple_processing_service import SimpleProcessingService
    processing_service = SimpleProcessingService(config.data_path)
    logger.info("✅ Successfully initialized processing service")
except ImportError as e:
    logger.error(f"Failed to import SimpleProcessingService: {e}")
    processing_service = None

try:
    from core.result_persistence import EnhancedResultPersistence
    enhanced_persistence = EnhancedResultPersistence(config.data_path)
    logger.info("✅ Successfully initialized enhanced persistence")
except ImportError as e:
    logger.warning(f"Enhanced persistence not available: {e}")
    enhanced_persistence = None

try:
    from core.advanced_filtering import AdvancedFeatureFilter, PatternCategory, QualityTier
    advanced_filter = AdvancedFeatureFilter()
    logger.info("✅ Successfully initialized advanced filter")
except ImportError as e:
    logger.warning(f"Advanced filter not available: {e}")
    advanced_filter = None
    # Create dummy classes for fallback
    class PatternCategory:
        pass
    class QualityTier:
        pass


# =============================================================================
# FastAPI Application Setup
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan - startup and shutdown."""
    logger.info(f"🚀 {config.service_name} API starting up...")
    logger.info(f"Data path: {config.data_path}")
    logger.info(f"Service version: {config.service_version}")
    logger.info(f"✅ Data path accessible")
    yield
    logger.info(f"🛑 {config.service_name} API shutting down...")


app = FastAPI(
    title="miStudioFind API",
    description="Feature analysis and discovery service for sparse autoencoders",
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
# API Endpoints with Proper Response Models
# =============================================================================

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        service=config.service_name,
        version=config.service_version,
        data_path=config.data_path,
        timestamp=datetime.now().isoformat(),
        components=ComponentStatus(
            processing_service=processing_service is not None,
            enhanced_persistence=enhanced_persistence is not None,
            advanced_filter=advanced_filter is not None
        )
    )


@app.get("/api/v1/config", response_model=ConfigResponse)
async def get_config():
    """Get current service configuration"""
    return ConfigResponse(
        service_name=config.service_name,
        service_version=config.service_version,
        data_path=config.data_path,
        api_host=config.api_host,
        api_port=config.api_port,
        top_k_selections=config.top_k_selections,
        coherence_threshold=config.coherence_threshold,
        max_concurrent_jobs=config.max_concurrent_jobs,
        processing_timeout_minutes=config.processing_timeout_minutes,
        memory_optimization=config.memory_optimization
    )


@app.post("/api/v1/find", response_model=FindJobResponse)
async def start_feature_analysis(request: FindRequest, background_tasks: BackgroundTasks):
    """Start feature analysis job"""
    
    if processing_service is None:
        raise HTTPException(
            status_code=503, 
            detail="Processing service not available"
        )
    
    try:
        # Validate source job exists
        source_models_dir = config.models_dir / request.source_job_id
        source_activations_dir = config.activations_dir / request.source_job_id
        
        if not source_models_dir.exists() or not source_activations_dir.exists():
            raise HTTPException(
                status_code=404,
                detail=f"Source job {request.source_job_id} not found. Expected directories: {source_models_dir}, {source_activations_dir}"
            )
        
        # Check for required files
        sae_model_path = source_models_dir / "sae_model.pt"
        feature_activations_path = source_activations_dir / "feature_activations.pt"
        metadata_path = source_activations_dir / "metadata.json"
        
        missing_files = []
        for path in [sae_model_path, feature_activations_path, metadata_path]:
            if not path.exists():
                missing_files.append(str(path))
        
        if missing_files:
            raise HTTPException(
                status_code=400,
                detail=f"Missing required files: {missing_files}"
            )
        
        # Start analysis job
        job_id = await processing_service.start_analysis_job(
            source_job_id=request.source_job_id,
            top_k=request.top_k
        )
        
        # Echo job name to console/logs
        logger.info(f"🔍 Started feature analysis job: {job_id}")
        logger.info(f"📊 Analyzing source job: {request.source_job_id}")
        logger.info(f"⚙️  Parameters: top_k={request.top_k}, coherence_threshold={request.coherence_threshold}")
        
        return FindJobResponse(
            job_id=job_id,
            status="queued",
            message=f"Feature analysis job '{job_id}' started for {request.source_job_id}",
            source_job_id=request.source_job_id,
            parameters=AnalysisParameters(
                top_k=request.top_k,
                coherence_threshold=request.coherence_threshold
            ),
            timestamp=datetime.now().isoformat(),
            next_steps=NextSteps(
                check_status=f"/api/v1/find/{job_id}/status",
                get_results=f"/api/v1/find/{job_id}/results"
            )
        )
        
    except Exception as e:
        logger.error(f"Error starting feature analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/find/{job_id}/status", response_model=JobStatusResponse)
async def get_job_status(job_id: str):
    """Get status of feature analysis job"""
    
    if processing_service is None:
        raise HTTPException(status_code=503, detail="Processing service not available")
    
    try:
        # Call non-async method from SimpleProcessingService
        status_data = processing_service.get_job_status(job_id)
        
        if status_data is None:
            raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
        
        # Convert SimpleProcessingService format to our response model
        return JobStatusResponse(
            job_id=job_id,
            status=status_data.get("status", "unknown"),
            progress=status_data.get("progress", {}).get("features_processed", 0) / max(1, status_data.get("progress", {}).get("total_features", 1)),
            message=f"Processing {status_data.get('progress', {}).get('features_processed', 0)}/{status_data.get('progress', {}).get('total_features', 0)} features",
            start_time=datetime.fromtimestamp(status_data.get("start_time", 0)).isoformat() if status_data.get("start_time") else None,
            estimated_time_remaining=status_data.get("progress", {}).get("estimated_time_remaining")
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting job status: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/find/{job_id}/results", response_model=JobResultResponse)
async def get_job_results(job_id: str):
    """Get results of completed feature analysis job"""
    
    if processing_service is None:
        raise HTTPException(status_code=503, detail="Processing service not available")
    
    try:
        # Call non-async method from SimpleProcessingService
        results_data = processing_service.get_job_results(job_id)
        
        if results_data is None:
            # Check if job exists but isn't completed
            status_data = processing_service.get_job_status(job_id)
            if status_data:
                raise HTTPException(
                    status_code=409,
                    detail=f"Job {job_id} is not completed yet. Current status: {status_data.get('status', 'unknown')}"
                )
            else:
                raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
        
        # Convert SimpleProcessingService format to our response model
        return JobResultResponse(
            job_id=job_id,
            status=results_data.get("status", "unknown"),
            results_path=f"/api/v1/find/{job_id}/download/json",  # Provide download path
            feature_count=len(results_data.get("results", [])),
            processing_time=results_data.get("processing_time", 0)
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting job results: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/jobs", response_model=JobListResponse)
async def list_jobs():
    """List all analysis jobs"""
    
    if processing_service is None:
        raise HTTPException(status_code=503, detail="Processing service not available")
    
    try:
        # Get jobs from both active and completed dictionaries
        job_summaries = []
        
        # Add active jobs
        for job_id, job_data in processing_service.active_jobs.items():
            job_summaries.append(JobSummary(
                job_id=job_id,
                status=job_data.get("status", "unknown"),
                source_job_id=job_data.get("source_job_id", ""),
                created_at=datetime.fromtimestamp(job_data.get("start_time", 0)).isoformat(),
                completed_at=None
            ))
        
        # Add completed jobs
        for job_id, job_data in processing_service.completed_jobs.items():
            completed_time = None
            if job_data.get("completion_time"):
                completed_time = datetime.fromtimestamp(job_data["completion_time"]).isoformat()
            
            job_summaries.append(JobSummary(
                job_id=job_id,
                status=job_data.get("status", "unknown"),
                source_job_id=job_data.get("source_job_id", ""),
                created_at=datetime.fromtimestamp(job_data.get("start_time", 0)).isoformat(),
                completed_at=completed_time
            ))
        
        return JobListResponse(jobs=job_summaries, total=len(job_summaries))
    except Exception as e:
        logger.error(f"Error listing jobs: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/v1/find/{job_id}", response_model=CancelJobResponse)
async def cancel_job(job_id: str):
    """Cancel a running analysis job"""
    
    if processing_service is None:
        raise HTTPException(status_code=503, detail="Processing service not available")
    
    try:
        success = await processing_service.cancel_job(job_id)
        if success:
            return CancelJobResponse(message=f"Job {job_id} cancelled successfully")
        else:
            raise HTTPException(status_code=400, detail=f"Job {job_id} cannot be cancelled")
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    except Exception as e:
        logger.error(f"Error cancelling job: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# File Download Endpoints (if enhanced_persistence is available)
# =============================================================================

@app.get("/api/v1/find/{job_id}/download/{format}")
async def download_results(job_id: str, format: str):
    """Download analysis results in specified format"""
    
    if enhanced_persistence is None:
        raise HTTPException(status_code=503, detail="Enhanced persistence not available")
    
    try:
        file_path = await enhanced_persistence.get_result_file(job_id, format)
        if not file_path.exists():
            raise HTTPException(status_code=404, detail=f"Results not found for job {job_id} in format {format}")
        
        return FileResponse(
            path=str(file_path),
            filename=f"{job_id}_results.{format}",
            media_type="application/octet-stream"
        )
    except Exception as e:
        logger.error(f"Error downloading results: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# Additional Utility Endpoints
# =============================================================================

@app.get("/api/v1/source-jobs")
async def list_source_jobs():
    """List available training jobs that can be analyzed"""
    try:
        models_dir = config.models_dir
        activations_dir = config.activations_dir
        
        source_jobs = []
        
        if models_dir.exists():
            for job_dir in models_dir.iterdir():
                if job_dir.is_dir() and job_dir.name.startswith("train_"):
                    # Check if corresponding activations exist
                    activation_dir = activations_dir / job_dir.name
                    sae_model_path = job_dir / "sae_model.pt"
                    
                    if activation_dir.exists() and sae_model_path.exists():
                        source_jobs.append({
                            "job_id": job_dir.name,
                            "model_path": str(sae_model_path),
                            "activations_path": str(activation_dir),
                            "ready_for_analysis": True
                        })
        
        return {
            "source_jobs": source_jobs,
            "total": len(source_jobs),
            "data_path": str(config.data_path)
        }
    except Exception as e:
        logger.error(f"Error listing source jobs: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/stats")
async def get_service_stats():
    """Get service statistics"""
    try:
        stats = {
            "service": config.service_name,
            "version": config.service_version,
            "uptime_seconds": 0,  # Could be calculated if tracking start time
            "total_jobs_processed": 0,  # Would need to track this
            "active_jobs": len(processing_service.active_jobs) if processing_service else 0,
            "data_path": config.data_path,
            "disk_usage": {
                "models_dir": str(config.models_dir),
                "activations_dir": str(config.activations_dir),
                "results_dir": str(config.results_dir)
            }
        }
        return stats
    except Exception as e:
        logger.error(f"Error getting service stats: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# Application Entry Point
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    
    logger.info(f"🚀 Starting {config.service_name} on {config.api_host}:{config.api_port}")
    logger.info(f"📂 Data path: {config.data_path}")
    logger.info(f"🔗 Points to: {config.data_path_obj.resolve()}")
    
    uvicorn.run(
        app, 
        host=config.api_host, 
        port=config.api_port, 
        log_level=config.log_level.lower(),
        access_log=True
    )