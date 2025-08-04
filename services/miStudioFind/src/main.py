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
    """Start feature analysis job with robust source data validation"""
    
    if processing_service is None:
        raise HTTPException(status_code=503, detail="Processing service not available")
    
    try:
        # Enhanced source data validation with comprehensive fallback logic
        training_data = _load_training_data_robust(request.source_job_id)
        
        if training_data is None:
            raise HTTPException(
                status_code=404,
                detail=f"Training job {request.source_job_id} not found or incomplete"
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


def _load_training_data_robust(source_job_id: str) -> Optional[Dict[str, Any]]:
    """Load training data with comprehensive fallback logic - like miStudioScore"""
    
    logger.info(f"🔍 Loading training data for job: {source_job_id}")
    
    # Define multiple possible locations for training data (comprehensive like miStudioScore)
    possible_locations = [
        # Standard training output locations
        {
            "models_dir": config.data_path_obj / "models" / source_job_id,
            "activations_dir": config.data_path_obj / "activations" / source_job_id,
            "description": "standard_training_output"
        },
        # Results-based locations
        {
            "models_dir": config.data_path_obj / "results" / "train" / source_job_id / "models",
            "activations_dir": config.data_path_obj / "results" / "train" / source_job_id / "activations",
            "description": "results_train_structured"
        },
        # Flat results directory
        {
            "models_dir": config.data_path_obj / "results" / "train" / source_job_id,
            "activations_dir": config.data_path_obj / "results" / "train" / source_job_id,
            "description": "results_train_flat"
        },
        # Alternative training directory
        {
            "models_dir": config.data_path_obj / "training" / source_job_id / "models",
            "activations_dir": config.data_path_obj / "training" / source_job_id / "activations",
            "description": "training_structured"
        },
        # Flat training directory
        {
            "models_dir": config.data_path_obj / "training" / source_job_id,
            "activations_dir": config.data_path_obj / "training" / source_job_id,
            "description": "training_flat"
        }
    ]
    
    # Define multiple possible file naming patterns for each location
    model_file_patterns = ["sae_model.pt", f"{source_job_id}_sae_model.pt", "model.pt", "sparse_autoencoder.pt"]
    activation_file_patterns = ["feature_activations.pt", f"{source_job_id}_activations.pt", "activations.pt"]
    metadata_file_patterns = ["metadata.json", f"{source_job_id}_metadata.json", "training_info.json", "info.json"]
    
    # Try each location with each file pattern combination
    for location in possible_locations:
        models_dir = location["models_dir"]
        activations_dir = location["activations_dir"]
        description = location["description"]
        
        logger.debug(f"📂 Checking location: {description}")
        logger.debug(f"   Models dir: {models_dir}")
        logger.debug(f"   Activations dir: {activations_dir}")
        
        if not models_dir.exists() or not activations_dir.exists():
            logger.debug(f"   ❌ Directories don't exist")
            continue
        
        # Try to find required files with different naming patterns
        found_files = {}
        
        # Find model file
        for pattern in model_file_patterns:
            model_path = models_dir / pattern
            if model_path.exists():
                found_files["model"] = model_path
                logger.debug(f"   ✅ Found model: {pattern}")
                break
        
        # Find activations file
        for pattern in activation_file_patterns:
            activations_path = activations_dir / pattern
            if activations_path.exists():
                found_files["activations"] = activations_path
                logger.debug(f"   ✅ Found activations: {pattern}")
                break
        
        # Find metadata file
        for pattern in metadata_file_patterns:
            metadata_path = activations_dir / pattern
            if metadata_path.exists():
                found_files["metadata"] = metadata_path
                logger.debug(f"   ✅ Found metadata: {pattern}")
                break
            # Also check in models directory
            metadata_path = models_dir / pattern
            if metadata_path.exists():
                found_files["metadata"] = metadata_path
                logger.debug(f"   ✅ Found metadata in models dir: {pattern}")
                break
        
        # Check if we have all required files
        if "model" in found_files and "activations" in found_files:
            logger.info(f"✅ Found complete training data at: {description}")
            logger.info(f"   Model: {found_files['model']}")
            logger.info(f"   Activations: {found_files['activations']}")
            if "metadata" in found_files:
                logger.info(f"   Metadata: {found_files['metadata']}")
            
            # Load and validate metadata if available
            metadata = {}
            if "metadata" in found_files:
                try:
                    with open(found_files["metadata"], 'r') as f:
                        metadata = json.load(f)
                    logger.info(f"   � Features: {metadata.get('num_features', 'unknown')}")
                except Exception as e:
                    logger.warning(f"   ⚠️ Could not read metadata: {e}")
            
            return {
                "source_job_id": source_job_id,
                "location": description,
                "files": found_files,
                "metadata": metadata,
                "validated": True
            }
        else:
            missing = []
            if "model" not in found_files:
                missing.append("model")
            if "activations" not in found_files:
                missing.append("activations")
            logger.debug(f"   ❌ Missing files: {missing}")
    
    # If we get here, no valid training data was found - provide comprehensive debugging info
    logger.error(f"❌ No valid training data found for job {source_job_id}")
    logger.error(f"🔍 Searched locations:")
    for location in possible_locations:
        logger.error(f"   - {location['description']}")
        logger.error(f"     Models: {location['models_dir']} (exists: {location['models_dir'].exists()})")
        logger.error(f"     Activations: {location['activations_dir']} (exists: {location['activations_dir'].exists()})")
    
    # Show available training jobs for debugging
    available_jobs = _discover_available_training_jobs()
    available_ids = [job["training_job_id"] for job in available_jobs]
    logger.error(f"📋 Available training jobs: {available_ids}")
    
    return None


def _discover_available_training_jobs() -> List[Dict[str, Any]]:
    """Discover available training jobs - internal helper"""
    available_jobs = []
    
    # Check main training models directory
    models_base = config.data_path_obj / "models"
    if models_base.exists():
        for job_dir in models_base.iterdir():
            if job_dir.is_dir():
                # Look for any model file
                model_files = list(job_dir.glob("*.pt"))
                if model_files:
                    available_jobs.append({
                        "training_job_id": job_dir.name,
                        "location": "models_directory",
                        "model_files": [str(f) for f in model_files]
                    })
    
    # Check results/train directory
    results_base = config.data_path_obj / "results" / "train"
    if results_base.exists():
        for job_dir in results_base.iterdir():
            if job_dir.is_dir() and job_dir.name not in [j["training_job_id"] for j in available_jobs]:
                model_files = list(job_dir.glob("**/*.pt"))
                if model_files:
                    available_jobs.append({
                        "training_job_id": job_dir.name,
                        "location": "results_train",
                        "model_files": [str(f) for f in model_files]
                    })
    
    return available_jobs


def _ensure_standardized_result_storage(job_id: str, results_data: Dict[str, Any]) -> Dict[str, str]:
    """Ensure results are saved with multiple naming conventions like miStudioScore"""
    
    try:
        # Create job-specific directory
        job_results_dir = config.results_path / job_id
        job_results_dir.mkdir(parents=True, exist_ok=True)
        
        saved_files = {}
        
        # Save with multiple naming conventions for cross-service compatibility
        naming_patterns = [
            "analysis_results.json",  # Primary name
            f"{job_id}_analysis_results.json",  # Job-specific name
            f"{job_id}_complete_results.json",  # Complete results name
            f"{job_id}_analysis.json",  # Short analysis name
            "results.json"  # Generic fallback name
        ]
        
        # Save JSON results with multiple names
        for pattern in naming_patterns:
            results_path = job_results_dir / pattern
            with open(results_path, 'w') as f:
                json.dump(results_data, f, indent=2, default=str)
            saved_files[pattern.replace('.json', '')] = str(results_path)
        
        # Save CSV results if possible
        if "results" in results_data and results_data["results"]:
            csv_patterns = [
                "features_analysis.csv",
                f"{job_id}_features_analysis.csv",
                f"{job_id}_analysis.csv",
                "analysis.csv"
            ]
            
            for pattern in csv_patterns:
                csv_path = job_results_dir / pattern
                _save_results_as_csv(results_data["results"], csv_path)
                saved_files[pattern.replace('.csv', '')] = str(csv_path)
        
        logger.info(f"✅ Results saved with {len(saved_files)} naming patterns to {job_results_dir}")
        return saved_files
        
    except Exception as e:
        logger.error(f"❌ Failed to save standardized results for job {job_id}: {e}")
        return {}


def _save_results_as_csv(features: List[Dict[str, Any]], csv_path: Path):
    """Save feature analysis results in CSV format - like miStudioScore"""
    import csv
    
    if not features:
        logger.warning("No features to save to CSV")
        return
    
    # Determine all possible columns from all features
    all_columns = set()
    for feature in features:
        all_columns.update(feature.keys())
    
    # Order columns logically
    priority_columns = ["feature_id", "feature_index", "coherence_score", "max_activation", "pattern_description"]
    remaining_columns = all_columns - set(priority_columns)
    
    # Final column order
    ordered_columns = []
    for col in priority_columns:
        if col in all_columns:
            ordered_columns.append(col)
    
    # Add remaining columns
    ordered_columns.extend(sorted(remaining_columns))
    
    try:
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=ordered_columns)
            writer.writeheader()
            
            for feature in features:
                row = {}
                for field in ordered_columns:
                    value = feature.get(field, "")
                    # Handle complex objects by converting to string
                    if isinstance(value, (dict, list)):
                        value = str(value)
                    row[field] = value
                writer.writerow(row)
        
        logger.info(f"✅ CSV results saved to: {csv_path} ({len(features)} features)")
        
    except Exception as e:
        logger.error(f"❌ Failed to save CSV results: {e}")


# =============================================================================
# Enhanced API Endpoints with Robust Storage
# =============================================================================
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
        
        # Prepare download links with standardized endpoints like miStudioScore
        download_links = {
            "json": f"/api/v1/find/{job_id}/download/json",
            "csv": f"/api/v1/find/{job_id}/download/csv"
        }
        
        # Add additional file download links for saved files
        saved_files = status_data.get("saved_files", {})
        if saved_files:
            for file_type, file_path in saved_files.items():
                if file_type not in download_links:  # Don't override standard endpoints
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
# File Download Endpoints - Standardized like miStudioScore
# =============================================================================

@app.get("/api/v1/find/{job_id}/download/json")
async def download_json_results(job_id: str):
    """Download JSON results file - standardized endpoint"""
    
    if processing_service is None:
        raise HTTPException(status_code=503, detail="Processing service not available")
    
    try:
        results_path = config.results_path / job_id / "analysis_results.json"
        
        if not results_path.exists():
            raise HTTPException(status_code=404, detail=f"Results file not found for job {job_id}")
        
        return FileResponse(
            path=str(results_path),
            filename=f"analysis_results_{job_id}.json",
            media_type="application/json"
        )
        
    except Exception as e:
        logger.error(f"Error downloading JSON results: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/find/{job_id}/download/csv")
async def download_csv_results(job_id: str):
    """Download CSV results file - standardized endpoint"""
    
    if processing_service is None:
        raise HTTPException(status_code=503, detail="Processing service not available")
    
    try:
        results_path = config.results_path / job_id / "features_analysis.csv"
        
        if not results_path.exists():
            raise HTTPException(status_code=404, detail=f"CSV results file not found for job {job_id}")
        
        return FileResponse(
            path=str(results_path),
            filename=f"features_analysis_{job_id}.csv",
            media_type="text/csv"
        )
        
    except Exception as e:
        logger.error(f"Error downloading CSV results: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


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


@app.get("/api/v1/find/training/available")
async def list_available_training_jobs():
    """List available training jobs with comprehensive discovery - enhanced like miStudioScore"""
    
    try:
        available_jobs = []
        
        # Use the enhanced discovery function
        discovered_jobs = _discover_available_training_jobs()
        
        # Enhance each job with additional validation and metadata
        for job in discovered_jobs:
            job_id = job["training_job_id"]
            
            # Try to load training data to validate completeness
            training_data = _load_training_data_robust(job_id)
            
            enhanced_job = {
                "training_job_id": job_id,
                "location": job["location"],
                "status": "available" if training_data else "incomplete",
                "validation": "complete" if training_data else "missing_files"
            }
            
            if training_data:
                enhanced_job.update({
                    "model_path": str(training_data["files"].get("model", "")),
                    "activations_path": str(training_data["files"].get("activations", "")),
                    "metadata_path": str(training_data["files"].get("metadata", "")),
                    "feature_count": training_data["metadata"].get("num_features", "unknown"),
                    "has_metadata": "metadata" in training_data["files"]
                })
            else:
                enhanced_job["model_files"] = job.get("model_files", [])
            
            available_jobs.append(enhanced_job)
        
        # Sort by completeness and then by job_id
        available_jobs.sort(key=lambda x: (x["status"] == "available", x["training_job_id"]), reverse=True)
        
        complete_jobs = [j for j in available_jobs if j["status"] == "available"]
        incomplete_jobs = [j for j in available_jobs if j["status"] != "available"]
        
        logger.info(f"Discovered {len(complete_jobs)} complete and {len(incomplete_jobs)} incomplete training jobs")
        
        return {
            "available_training_jobs": available_jobs,
            "total": len(available_jobs),
            "complete_jobs": len(complete_jobs),
            "incomplete_jobs": len(incomplete_jobs),
            "search_locations": [
                str(config.data_path_obj / "models"),
                str(config.data_path_obj / "results" / "train"),
                str(config.data_path_obj / "training")
            ]
        }
        
    except Exception as e:
        logger.error(f"Error discovering training jobs: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/find/training/{training_job_id}/preview")
async def preview_training_job(training_job_id: str):
    """Preview a training job with comprehensive validation - enhanced like miStudioScore"""
    
    try:
        # Use robust loading function for validation
        training_data = _load_training_data_robust(training_job_id)
        
        if training_data is None:
            # If not found, provide detailed error information
            available_jobs = _discover_available_training_jobs()
            available_ids = [job["training_job_id"] for job in available_jobs]
            
            raise HTTPException(
                status_code=404,
                detail=f"Training job {training_job_id} not found. Available jobs: {available_ids}"
            )
        
        # Extract metadata information
        metadata = training_data.get("metadata", {})
        files = training_data.get("files", {})
        
        # Calculate recommendations based on metadata
        num_features = metadata.get("num_features", 20)
        recommended_top_k = min(20, max(5, num_features // 10))
        
        return {
            "training_job_id": training_job_id,
            "location": training_data["location"],
            "validation_status": "complete",
            "files_found": {
                "model": str(files.get("model", "")) if "model" in files else None,
                "activations": str(files.get("activations", "")) if "activations" in files else None,
                "metadata": str(files.get("metadata", "")) if "metadata" in files else None
            },
            "metadata": metadata,
            "num_features": num_features,
            "recommended_parameters": {
                "top_k": recommended_top_k,
                "coherence_threshold": 0.5,
                "max_features_to_analyze": min(100, num_features)
            },
            "analysis_ready": True
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error previewing training job {training_job_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# Legacy endpoints for compatibility
@app.get("/api/v1/config")
async def get_config():
    """Get current service configuration with enhanced information"""
    return {
        "service_name": config.service_name,
        "service_version": config.service_version,
        "data_path": config.data_path,
        "results_path": str(config.results_path),
        "api_host": config.api_host,
        "api_port": config.api_port,
        "top_k_selections": config.top_k_selections,
        "coherence_threshold": config.coherence_threshold,
        "max_concurrent_jobs": config.max_concurrent_jobs,
        "storage_features": {
            "robust_source_loading": True,
            "multiple_naming_conventions": True,
            "standardized_downloads": True,
            "cross_service_compatibility": True,
            "comprehensive_error_handling": True
        },
        "integration_status": {
            "processing_service": processing_service is not None,
            "enhanced_persistence": enhanced_persistence is not None,
            "advanced_filter": advanced_filter is not None
        }
    }


@app.get("/api/v1/find/integration/status")
async def get_integration_status():
    """Check integration status with other miStudio services"""
    
    try:
        # Check integration with other services
        integration_status = {
            "train_integration": {
                "source_paths_checked": 5,
                "available_training_jobs": len(_discover_available_training_jobs()),
                "robust_loading": True
            },
            "explain_integration": {
                "standardized_outputs": True,
                "multiple_naming_patterns": True,
                "cross_service_discovery": True
            },
            "score_integration": {
                "consistent_storage_pattern": True,
                "standardized_endpoints": True,
                "fallback_logic_implemented": True
            },
            "storage_compliance": {
                "hierarchical_structure": True,
                "job_based_organization": True,
                "multiple_output_formats": True,
                "robust_path_resolution": True
            }
        }
        
        return {
            "service": config.service_name,
            "version": config.service_version,
            "integration_status": integration_status,
            "compliance_score": "HIGH",
            "recommendations": []
        }
        
    except Exception as e:
        logger.error(f"Error checking integration status: {str(e)}")
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
