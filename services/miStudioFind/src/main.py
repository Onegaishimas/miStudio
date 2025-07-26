# src/main.py - Clean version with proper imports
"""
Main FastAPI application for miStudioFind service - Complete Implementation.
"""

import os
import sys
import json
import logging
from typing import Dict, Any, Optional

# Add the parent directory to Python path to access core modules
SERVICE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, SERVICE_ROOT)

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse
from contextlib import asynccontextmanager
from datetime import datetime
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

# Configuration
class Config:
    """Service configuration."""
    data_path = os.getenv("DATA_PATH", "/home/sean/app/miStudio/services/miStudioTrain/data")
    api_host = os.getenv("API_HOST", "0.0.0.0")
    api_port = int(os.getenv("API_PORT", "8001"))
    log_level = os.getenv("LOG_LEVEL", "INFO")
    service_version = "1.0.0"

config = Config()

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

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan - startup and shutdown."""
    # Startup
    logger.info("miStudioFind API starting up...")
    logger.info(f"Data path: {config.data_path}")
    logger.info(f"Service version: {config.service_version}")
    
    # Verify data path exists
    if os.path.exists(config.data_path):
        logger.info("✅ Data path accessible")
    else:
        logger.warning(f"⚠️ Data path not found: {config.data_path}")
    
    yield
    
    # Shutdown
    logger.info("miStudioFind API shutting down...")

# FastAPI application
app = FastAPI(
    title="miStudioFind API",
    description="Feature Analysis Service for AI Interpretability - Analyzes SAE features from miStudioTrain",
    version=config.service_version,
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with service information."""
    return {
        "service": "miStudioFind", 
        "status": "running",
        "version": config.service_version,
        "description": "Feature Analysis Service for miStudio AI Interpretability Platform",
        "documentation": "/docs",
        "health_check": "/health",
        "features_available": {
            "basic_analysis": processing_service is not None,
            "file_persistence": enhanced_persistence is not None,
            "advanced_filtering": advanced_filter is not None
        }
    }

# Health check endpoint
@app.get("/health")
async def health_check():
    """Comprehensive health check endpoint."""
    try:
        # Check data path accessibility
        data_path_exists = os.path.exists(config.data_path)
        data_path_writable = False
        
        if data_path_exists:
            try:
                test_file = os.path.join(config.data_path, ".health_check")
                with open(test_file, 'w') as f:
                    f.write("test")
                os.remove(test_file)
                data_path_writable = True
            except (OSError, PermissionError):
                data_path_writable = False
        
        # Check processing service
        processing_service_healthy = processing_service is not None
        
        # Overall health
        healthy = data_path_exists and processing_service_healthy
        
        return {
            "healthy": healthy,
            "service": "miStudioFind",
            "version": config.service_version,
            "timestamp": datetime.now().isoformat(),
            "configuration": {
                "data_path": config.data_path,
                "api_host": config.api_host,
                "api_port": config.api_port,
                "log_level": config.log_level
            },
            "system_checks": {
                "data_path_exists": data_path_exists,
                "data_path_writable": data_path_writable,
                "processing_service_initialized": processing_service_healthy,
                "enhanced_persistence_available": enhanced_persistence is not None,
                "advanced_filtering_available": advanced_filter is not None
            },
            "endpoints": {
                "validation": "/api/v1/validate/{job_id}",
                "start_analysis": "/api/v1/find/start",
                "job_status": "/api/v1/find/{job_id}/status",
                "job_results": "/api/v1/find/{job_id}/results",
                "list_jobs": "/api/v1/find/jobs",
                "export": "/api/v1/find/{job_id}/export" if enhanced_persistence else "Not available",
                "filtering": "/api/v1/find/{job_id}/results/filtered" if advanced_filter else "Not available"
            }
        }
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "healthy": False,
            "service": "miStudioFind",
            "version": config.service_version,
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

# File validation endpoint
@app.get("/api/v1/validate/{job_id}")
async def validate_training_files(job_id: str):
    """Validate that miStudioTrain output files exist and are ready for analysis."""
    try:
        logger.info(f"Validating files for job: {job_id}")
        
        # Define expected file paths from miStudioTrain
        files_to_check = {
            "sae_model": f"{config.data_path}/models/{job_id}/sae_model.pt",
            "feature_activations": f"{config.data_path}/activations/{job_id}/feature_activations.pt", 
            "metadata": f"{config.data_path}/activations/{job_id}/metadata.json"
        }
        
        results = {}
        total_size_mb = 0
        
        # Check each file
        for name, path in files_to_check.items():
            file_exists = os.path.exists(path)
            file_size = 0
            file_readable = False
            
            if file_exists:
                try:
                    file_size = os.path.getsize(path)
                    total_size_mb += file_size / 1e6
                    # Test readability
                    with open(path, 'rb') as f:
                        f.read(1)  # Try to read first byte
                    file_readable = True
                except (OSError, PermissionError):
                    file_readable = False
            
            results[name] = {
                "path": path,
                "exists": file_exists,
                "size_bytes": file_size,
                "size_mb": round(file_size / 1e6, 2),
                "readable": file_readable
            }
        
        # Overall validation
        all_files_present = all(f["exists"] for f in results.values())
        all_files_readable = all(f["readable"] for f in results.values())
        reasonable_sizes = all(f["size_mb"] > 0.1 for f in results.values())
        
        # Try to peek at metadata if available
        metadata_info = {}
        metadata_path = files_to_check["metadata"]
        if os.path.exists(metadata_path):
            try:
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                metadata_info = {
                    "source_service": metadata.get("service", "unknown"),
                    "model_name": metadata.get("model_info", {}).get("model_name", "unknown"),
                    "feature_count": metadata.get("sae_config", {}).get("hidden_dim", 0),
                    "activation_dim": metadata.get("model_info", {}).get("hidden_size", 0),
                    "ready_for_find": metadata.get("ready_for_find_service", False),
                    "training_loss": metadata.get("training_results", {}).get("final_loss", "unknown")
                }
            except (json.JSONDecodeError, KeyError) as e:
                metadata_info = {"error": f"Could not parse metadata: {e}"}
        
        return {
            "job_id": job_id,
            "validation_timestamp": datetime.now().isoformat(),
            "files": results,
            "summary": {
                "all_files_present": all_files_present,
                "all_files_readable": all_files_readable,
                "reasonable_file_sizes": reasonable_sizes,
                "total_size_mb": round(total_size_mb, 2),
                "ready_for_analysis": all_files_present and all_files_readable and reasonable_sizes
            },
            "metadata_info": metadata_info,
            "next_steps": {
                "can_start_analysis": all_files_present and all_files_readable,
                "recommendation": (
                    "Files look good - ready for feature analysis!" 
                    if all_files_present and all_files_readable 
                    else "Please check file accessibility"
                )
            }
        }
        
    except Exception as e:
        logger.error(f"Error validating files for job {job_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Validation error: {str(e)}")

# Start analysis endpoint
@app.post("/api/v1/find/start")
async def start_analysis(request: Dict[str, Any]):
    """Start feature analysis job."""
    try:
        # Extract request parameters
        source_job_id = request.get("source_job_id")
        top_k = request.get("top_k", 20)
        coherence_threshold = request.get("coherence_threshold", 0.7)
        
        # Validate required parameters
        if not source_job_id:
            raise HTTPException(status_code=400, detail="source_job_id is required")
        
        if not 1 <= top_k <= 100:
            raise HTTPException(status_code=400, detail="top_k must be between 1 and 100")
        
        if not 0.0 <= coherence_threshold <= 1.0:
            raise HTTPException(status_code=400, detail="coherence_threshold must be between 0.0 and 1.0")
        
        logger.info(f"Starting analysis for job: {source_job_id} with top_k={top_k}")
        
        # Check if processing service is available
        if processing_service is None:
            raise HTTPException(
                status_code=503, 
                detail="Processing service not available. Check core module imports."
            )
        
        # Start the analysis job
        job_id = await processing_service.start_analysis_job(source_job_id, top_k)
        
        return {
            "job_id": job_id,
            "status": "queued", 
            "message": f"Feature analysis started for {source_job_id}",
            "source_job_id": source_job_id,
            "parameters": {
                "top_k": top_k,
                "coherence_threshold": coherence_threshold
            },
            "timestamp": datetime.now().isoformat(),
            "next_steps": {
                "check_status": f"/api/v1/find/{job_id}/status",
                "get_results": f"/api/v1/find/{job_id}/results"
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to start analysis: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis startup failed: {str(e)}")

# Job status endpoint
@app.get("/api/v1/find/{job_id}/status")
async def get_job_status(job_id: str):
    """Get current status of an analysis job."""
    try:
        if processing_service is None:
            raise HTTPException(status_code=503, detail="Processing service not available")
        
        status = processing_service.get_job_status(job_id)
        if not status:
            raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
        
        # Add computed fields
        status["timestamp"] = datetime.now().isoformat()
        
        # Calculate progress percentage
        progress = status.get("progress", {})
        total = progress.get("total_features", 0)
        processed = progress.get("features_processed", 0)
        
        if total > 0:
            status["progress_percentage"] = round((processed / total) * 100, 1)
        else:
            status["progress_percentage"] = 0.0
        
        return status
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting status for job {job_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Status check failed: {str(e)}")

# Job results endpoint
@app.get("/api/v1/find/{job_id}/results")
async def get_job_results(job_id: str):
    """Get results of a completed analysis job."""
    try:
        if processing_service is None:
            raise HTTPException(status_code=503, detail="Processing service not available")
        
        results = processing_service.get_job_results(job_id)
        if not results:
            # Check if job exists but isn't completed
            status = processing_service.get_job_status(job_id)
            if status:
                raise HTTPException(
                    status_code=409,
                    detail=f"Job {job_id} is not completed yet. Current status: {status['status']}"
                )
            else:
                raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
        
        # Calculate summary statistics
        feature_results = results.get("results", [])
        if feature_results:
            quality_distribution = {"high": 0, "medium": 0, "low": 0}
            coherence_scores = []
            
            for result in feature_results:
                quality_level = result.get("quality_level", "low")
                quality_distribution[quality_level] = quality_distribution.get(quality_level, 0) + 1
                coherence_scores.append(result.get("coherence_score", 0.0))
            
            summary = {
                "total_features_analyzed": len(feature_results),
                "quality_distribution": quality_distribution,
                "mean_coherence_score": round(sum(coherence_scores) / len(coherence_scores), 3),
                "high_quality_features": quality_distribution["high"],
                "interpretable_features": quality_distribution["high"] + quality_distribution["medium"]
            }
        else:
            summary = {
                "total_features_analyzed": 0,
                "quality_distribution": {"high": 0, "medium": 0, "low": 0},
                "mean_coherence_score": 0.0,
                "high_quality_features": 0,
                "interpretable_features": 0
            }
        
        return {
            "job_id": job_id,
            "source_job_id": results.get("source_job_id"),
            "status": results.get("status"),
            "processing_time_seconds": results.get("processing_time", 0),
            "timestamp": datetime.now().isoformat(),
            "summary": summary,
            "results": feature_results,
            "ready_for_explain_service": summary["interpretable_features"] > 0
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting results for job {job_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Results retrieval failed: {str(e)}")

# List jobs endpoint
@app.get("/api/v1/find/jobs")
async def list_jobs():
    """List all analysis jobs (active and completed)."""
    try:
        if processing_service is None:
            return {
                "active_jobs": [],
                "completed_jobs": [],
                "message": "Processing service not available"
            }
        
        active_jobs = list(processing_service.active_jobs.values())
        completed_jobs = list(processing_service.completed_jobs.values())
        
        # Add summary statistics
        total_jobs = len(active_jobs) + len(completed_jobs)
        successful_jobs = len([job for job in completed_jobs if job.get("status") == "completed"])
        
        return {
            "active_jobs": active_jobs,
            "completed_jobs": completed_jobs,
            "summary": {
                "total_jobs": total_jobs,
                "active_count": len(active_jobs),
                "completed_count": len(completed_jobs),
                "success_rate": (successful_jobs / len(completed_jobs) * 100) if completed_jobs else 0.0
            },
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error listing jobs: {e}")
        raise HTTPException(status_code=500, detail=f"Job listing failed: {str(e)}")

# ENHANCED ENDPOINTS (Only available if modules loaded successfully)

@app.get("/api/v1/find/{job_id}/export")
async def export_results(job_id: str, format: str = "json", include_files: bool = True):
    """Export results in specified format with optional file bundle."""
    if enhanced_persistence is None:
        raise HTTPException(status_code=503, detail="Enhanced persistence not available")
    
    try:
        # Get job results first
        if processing_service is None:
            raise HTTPException(status_code=503, detail="Processing service not available")
        
        results = processing_service.get_job_results(job_id)
        if not results:
            raise HTTPException(status_code=404, detail=f"Job {job_id} not found or not completed")
        
        # Save in all formats
        saved_files = enhanced_persistence.save_comprehensive_results(job_id, results)
        
        if format == "all" or include_files:
            # Create zip bundle with all formats
            zip_buffer = io.BytesIO()
            
            with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                for format_name, file_path in saved_files.items():
                    if os.path.exists(file_path):
                        zip_file.write(file_path, f"{job_id}_{format_name}.{format_name}")
            
            zip_buffer.seek(0)
            
            return StreamingResponse(
                io.BytesIO(zip_buffer.read()),
                media_type="application/zip",
                headers={"Content-Disposition": f"attachment; filename={job_id}_analysis_complete.zip"}
            )
        
        elif format in saved_files:
            # Return specific format
            file_path = saved_files[format]
            if os.path.exists(file_path):
                return FileResponse(
                    file_path,
                    media_type="application/octet-stream",
                    filename=f"{job_id}_analysis.{format}"
                )
            else:
                raise HTTPException(status_code=404, detail=f"File not found: {format}")
        
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported format: {format}")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Export failed for job {job_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Export failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    
    logger.info(f"Starting miStudioFind API on {config.api_host}:{config.api_port}")
    logger.info(f"Data path configured: {config.data_path}")
    logger.info(f"Service version: {config.service_version}")
    logger.info("Ready to analyze your Phi-4 breakthrough results!")
    
    uvicorn.run(
        app,
        host=config.api_host,
        port=config.api_port,
        log_level=config.log_level.lower()
    )
