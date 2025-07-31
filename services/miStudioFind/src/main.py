# src/main.py - Fixed Version with Proper Import Order and Data Path Configuration
"""
Main FastAPI application for miStudioFind service - Fixed Implementation.
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
# Unified Configuration - Fixed Data Path Setup
# =============================================================================

class ServiceConfig:
    """Unified configuration for miStudioFind - Environment-first approach"""
    
    def __init__(self):
        # Primary data path - same pattern for all services
        self.data_path = os.getenv("DATA_PATH", "/data")
        self.data_path_obj = Path(self.data_path)
        self.data_path_obj.mkdir(parents=True, exist_ok=True)
        
        # Ensure results directory structure exists
        self.results_path = self.data_path_obj / "results" / "find"
        self.results_path.mkdir(parents=True, exist_ok=True)
        
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
        self.max_concurrent_jobs = int(os.getenv("MAX_CONCURRENT_JOBS", "2"))
        
        logger.info(f"🔧 ServiceConfig initialized:")
        logger.info(f"   Data path: {self.data_path}")
        logger.info(f"   Results path: {self.results_path}")
        logger.info(f"   Service: {self.service_name} v{self.service_version}")
        logger.info(f"   API: {self.api_host}:{self.api_port}")

# Initialize global config
config = ServiceConfig()


# =============================================================================
# Request/Response Models
# =============================================================================

class FindRequest(BaseModel):
    """Request model for feature analysis"""
    source_job_id: str = Field(..., description="Source training job ID from miStudioTrain")
    top_k: int = Field(20, description="Number of top activations per feature", ge=1, le=100)
    coherence_threshold: float = Field(0.5, description="Minimum coherence score for feature inclusion", ge=0.0, le=1.0)

class AnalysisParameters(BaseModel):
    """Analysis parameters included in response"""
    top_k: int
    coherence_threshold: float

class NextSteps(BaseModel):
    """Next steps information"""
    check_status: str
    get_results: str

class FindJobResponse(BaseModel):
    """Response model for starting analysis job"""
    job_id: str
    status: str
    message: str
    source_job_id: str
    parameters: AnalysisParameters
    timestamp: str
    next_steps: NextSteps

class JobProgress(BaseModel):
    """Job progress information"""
    features_processed: int
    total_features: int
    estimated_time_remaining: Optional[int] = None

class JobStatusResponse(BaseModel):
    """Response model for job status"""
    job_id: str
    status: str
    source_job_id: str
    start_time: Optional[str] = None
    completion_time: Optional[str] = None
    processing_time: Optional[float] = None
    progress: Optional[JobProgress] = None
    error: Optional[str] = None
    results_path: Optional[str] = None

class JobResultResponse(BaseModel):
    """Response model for job results"""
    job_id: str
    status: str
    source_job_id: str
    feature_count: Optional[int] = None
    results_summary: Optional[Dict[str, Any]] = None
    download_links: Optional[Dict[str, str]] = None
    processing_time: Optional[float] = None

class HealthResponse(BaseModel):
    """Health check response model"""
    status: str
    service: str
    version: str
    data_path: str
    results_path: str
    timestamp: str
    components: Dict[str, bool]

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
# Service Initialization with Proper Order - Fixed
# =============================================================================

# Initialize variables first
processing_service = None
enhanced_persistence = None
advanced_filter = None

# Import and initialize enhanced persistence first (no dependencies)
try:
    from core.result_persistence import EnhancedResultPersistence
    enhanced_persistence = EnhancedResultPersistence(config.data_path)
    logger.info("✅ Successfully initialized enhanced persistence")
except ImportError as e:
    logger.warning(f"Enhanced persistence not available: {e}")
    enhanced_persistence = None

# Import and initialize processing service (depends on enhanced_persistence)
try:
    from core.simple_processing_service import SimpleProcessingService
    processing_service = SimpleProcessingService(config.data_path, enhanced_persistence)
    logger.info("✅ Successfully initialized processing service")
except ImportError as e:
    logger.error(f"Failed to import SimpleProcessingService: {e}")
    processing_service = None

# Import and initialize advanced filter (no dependencies)
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
    # Startup
    logger.info("🚀 miStudioFind service starting up...")
    logger.info(f"📁 Data path: {config.data_path}")
    logger.info(f"📊 Results path: {config.results_path}")
    
    # Verify directories exist
    config.results_path.mkdir(parents=True, exist_ok=True)
    logger.info(f"✅ Results directory ready: {config.results_path}")
    
    yield
    
    # Shutdown
    logger.info("🛑 miStudioFind service shutting down...")

app = FastAPI(
    title="miStudioFind API",
    description="Feature Analysis Service for Sparse Autoencoder Features",
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
    return HealthResponse(
        status="healthy",
        service=config.service_name,
        version=config.service_version,
        data_path=config.data_path,
        results_path=str(config.results_path),
        timestamp=datetime.now().isoformat(),
        components={
            "processing_service": processing_service is not None,
            "enhanced_persistence": enhanced_persistence is not None,
            "advanced_filter": advanced_filter is not None
        }
    )

@app.post("/api/v1/find/analyze", response_model=FindJobResponse)
async def start_feature_analysis(request: FindRequest):
    """Start feature analysis job"""
    
    if processing_service is None:
        raise HTTPException(status_code=503, detail="Processing service not available")
    
    try:
        # Validate source data exists
        source_models_dir = config.data_path_obj / "models" / request.source_job_id
        source_activations_dir = config.data_path_obj / "activations" / request.source_job_id
        
        if not source_models_dir.exists() or not source_activations_dir.exists():
            raise HTTPException(
                status_code=404,
                detail=f"Source job data not found for {request.source_job_id}. "
                f"Expected directories: {source_models_dir}, {source_activations_dir}"
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
        logger.info(f"💾 Results will be saved to: {config.results_path / job_id}")
        
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
        status_data = processing_service.get_job_status(job_id)
        
        if status_data is None:
            raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
        
        # Convert to response model
        progress_data = status_data.get("progress", {})
        
        return JobStatusResponse(
            job_id=job_id,
            status=status_data.get("status", "unknown"),
            source_job_id=status_data.get("source_job_id"),
            start_time=datetime.fromtimestamp(status_data["start_time"]).isoformat() if "start_time" in status_data else None,
            completion_time=datetime.fromtimestamp(status_data["completion_time"]).isoformat() if "completion_time" in status_data else None,
            processing_time=status_data.get("processing_time"),
            progress=JobProgress(
                features_processed=progress_data.get("features_processed", 0),
                total_features=progress_data.get("total_features", 0),
                estimated_time_remaining=progress_data.get("estimated_time_remaining")
            ) if progress_data else None,
            error=status_data.get("error"),
            results_path=status_data.get("results_path")
        )
        
    except Exception as e:
        logger.error(f"Error getting job status: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/find/{job_id}/results", response_model=JobResultResponse)
async def get_job_results(job_id: str):
    """Get results of completed analysis job"""
    
    if processing_service is None:
        raise HTTPException(status_code=503, detail="Processing service not available")
    
    try:
        status_data = processing_service.get_job_status(job_id)
        
        if status_data is None:
            raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
        
        if status_data.get("status") != "completed":
            raise HTTPException(
                status_code=400, 
                detail=f"Job {job_id} is not completed (status: {status_data.get('status')})"
            )
        
        results = processing_service.get_job_results(job_id)
        
        if results is None:
            raise HTTPException(status_code=404, detail=f"Results not found for job {job_id}")
        
        # Prepare download links for saved files
        download_links = {}
        saved_files = status_data.get("saved_files", {})
        if saved_files:
            for file_type, file_path in saved_files.items():
                download_links[file_type] = f"/api/v1/find/{job_id}/download/{file_type}"
        
        return JobResultResponse(
            job_id=job_id,
            status=status_data["status"],
            source_job_id=status_data["source_job_id"],
            feature_count=len(results) if results else 0,
            results_summary={
                "total_features": len(results) if results else 0,
                "analysis_type": "feature_analysis",
                "results_available": True
            },
            download_links=download_links if download_links else None,
            processing_time=status_data.get("processing_time")
        )
        
    except Exception as e:
        logger.error(f"Error getting job results: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/find/jobs", response_model=JobListResponse)
async def list_jobs():
    """List all analysis jobs"""
    
    if processing_service is None:
        raise HTTPException(status_code=503, detail="Processing service not available")
    
    try:
        jobs_data = processing_service.list_jobs()
        
        jobs = []
        for job_data in jobs_data:
            jobs.append(JobSummary(
                job_id=job_data["job_id"],
                status=job_data["status"],
                source_job_id=job_data["source_job_id"],
                created_at=datetime.fromtimestamp(job_data["start_time"]).isoformat(),
                completed_at=datetime.fromtimestamp(job_data["completion_time"]).isoformat() if job_data.get("completion_time") else None
            ))
        
        return JobListResponse(
            jobs=jobs,
            total=len(jobs)
        )
        
    except Exception as e:
        logger.error(f"Error listing jobs: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/v1/find/{job_id}", response_model=CancelJobResponse)
async def cancel_job(job_id: str):
    """Cancel a running analysis job"""
    
    if processing_service is None:
        raise HTTPException(status_code=503, detail="Processing service not available")
    
    try:
        success = processing_service.cancel_job(job_id)
        
        if success:
            return CancelJobResponse(message=f"Job {job_id} cancelled successfully")
        else:
            raise HTTPException(status_code=404, detail=f"Job {job_id} not found or cannot be cancelled")
            
    except Exception as e:
        logger.error(f"Error cancelling job: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# File Download Endpoints
# =============================================================================

@app.get("/api/v1/find/{job_id}/download/{file_type}")
async def download_result_file(job_id: str, file_type: str):
    """Download a specific result file"""
    
    if processing_service is None:
        raise HTTPException(status_code=503, detail="Processing service not available")
    
    try:
        status_data = processing_service.get_job_status(job_id)
        
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
            filename=f"{job_id}_{file_type}_{file_path.suffix}",
            media_type="application/octet-stream"
        )
        
    except Exception as e:
        logger.error(f"Error downloading file: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/find/{job_id}/download/all")
async def download_all_results(job_id: str):
    """Download all result files as a ZIP archive"""
    
    if processing_service is None:
        raise HTTPException(status_code=503, detail="Processing service not available")
    
    try:
        status_data = processing_service.get_job_status(job_id)
        
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
            
            # Add enhanced persistence files if available
            enhanced_files = status_data.get("enhanced_persistence_files", {})
            for file_type, file_path in enhanced_files.items():
                file_path_obj = Path(file_path)
                if file_path_obj.exists():
                    zip_file.write(file_path_obj, f"enhanced_{file_type}{file_path_obj.suffix}")
        
        zip_buffer.seek(0)
        
        return StreamingResponse(
            io.BytesIO(zip_buffer.read()),
            media_type="application/zip",
            headers={"Content-Disposition": f"attachment; filename={job_id}_results.zip"}
        )
        
    except Exception as e:
        logger.error(f"Error creating ZIP download: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# Application Entry Point
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    
    logger.info(f"🚀 Starting {config.service_name} v{config.service_version}")
    logger.info(f"📁 Data path: {config.data_path}")
    logger.info(f"📊 Results path: {config.results_path}")
    logger.info(f"🌐 API server: {config.api_host}:{config.api_port}")
    
    uvicorn.run(
        app,
        host=config.api_host,
        port=config.api_port,
        log_level=config.log_level.lower()
    )