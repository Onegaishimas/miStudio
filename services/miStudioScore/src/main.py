# src/main.py - Fixed version with proper _initialize_scorers method placement
"""
miStudioScore Service - Enhanced scoring service integrated with Find/Explain results
Fixed: Moved _initialize_scorers method inside IntegratedScoreService class
"""

import asyncio
import json
import logging
import os
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# =============================================================================
# Service Configuration
# =============================================================================

class ServiceConfig:
    """Service configuration"""
    
    def __init__(self, data_path: str = "/data"):
        self.data_path = Path(data_path)
        self.find_results_dir = self.data_path / "results" / "find"
        self.explain_results_dir = self.data_path / "results" / "explain"
        self.score_results_dir = self.data_path / "results" / "score"
        
        # Ensure directories exist
        for directory in [self.find_results_dir, self.explain_results_dir, self.score_results_dir]:
            directory.mkdir(parents=True, exist_ok=True)
        
        self.service_name = "miStudioScore"
        self.service_version = "v1.0.0"

# Initialize configuration
config = ServiceConfig()

logger.info("🔧 ServiceConfig initialized:")
logger.info(f"   Data path: {config.data_path}")
logger.info(f"   Find results: {config.find_results_dir}")
logger.info(f"   Explain results: {config.explain_results_dir}")
logger.info(f"   Score results: {config.score_results_dir}")
logger.info(f"   Service: {config.service_name} {config.service_version}")

# =============================================================================
# Request/Response Models
# =============================================================================

class ScoreParameters(BaseModel):
    """Parameters for scoring job"""
    source_type: str
    source_job_id: str
    scoring_jobs: List[Dict[str, Any]]
    output_formats: List[str] = ["json", "csv"]

class NextSteps(BaseModel):
    """Next steps information"""
    check_status: str
    get_results: str

class ScoreRequest(BaseModel):
    """Request model for starting scoring job"""
    source_type: str  # "find" or "explain"
    source_job_id: str
    scoring_config: Dict[str, Any]
    output_formats: List[str] = ["json", "csv"]

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
        """Initialize available scoring modules (safe version)"""
        # Start with empty list
        self.available_scorers = []
        
        # Only try to load working scorers with proper error handling
        try:
            # Check if relevance scorer exists
            scorer_file = Path(__file__).parent / "scorers" / "relevance_scorer.py"
            if scorer_file.exists():
                from src.scorers.relevance_scorer import RelevanceScorer
                self.available_scorers.append("relevance_scorer")
                logger.info("✅ Loaded RelevanceScorer")
            else:
                logger.warning("RelevanceScorer file not found")
        except Exception as e:
            logger.warning(f"RelevanceScorer not available: {e}")
        
        try:
            # Check if ablation scorer exists
            scorer_file = Path(__file__).parent / "scorers" / "ablation_scorer.py" 
            if scorer_file.exists():
                from src.scorers.ablation_scorer import AblationScorer
                self.available_scorers.append("ablation_scorer")
                logger.info("✅ Loaded AblationScorer")
            else:
                logger.warning("AblationScorer file not found")
        except Exception as e:
            logger.warning(f"AblationScorer not available: {e}")
        
        if not self.available_scorers:
            logger.warning("No scorers available - service will have limited functionality")
        else:
            logger.info(f"Working scorers loaded: {self.available_scorers}")
    
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
                                    job_data = json.load(f)
                                available_jobs.append({
                                    "job_id": job_dir.name,
                                    "timestamp": job_data.get("timestamp", "unknown"),
                                    "features_count": len(job_data.get("features", [])),
                                    "status": "completed"
                                })
                            except Exception as e:
                                logger.warning(f"Error reading {results_file}: {e}")
        
        elif source_type == "explain":
            source_dir = self.config.explain_results_dir
            if source_dir.exists():
                for job_dir in source_dir.iterdir():
                    if job_dir.is_dir():
                        results_file = job_dir / "explanation_results.json"
                        if results_file.exists():
                            try:
                                with open(results_file, 'r') as f:
                                    job_data = json.load(f)
                                available_jobs.append({
                                    "job_id": job_dir.name,
                                    "timestamp": job_data.get("timestamp", "unknown"),
                                    "features_count": len(job_data.get("features", [])),
                                    "status": "completed"
                                })
                            except Exception as e:
                                logger.warning(f"Error reading {results_file}: {e}")
        
        return available_jobs
    
    async def start_scoring_job(self, request: ScoreRequest) -> str:
        """Start a new scoring job"""
        # Generate job ID
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        job_id = f"score_{timestamp}_{hash(request.source_job_id) % 100000000:08x}"
        
        # Create job info
        job_info = {
            "job_id": job_id,
            "status": "queued",
            "source_type": request.source_type,
            "source_job_id": request.source_job_id,
            "scoring_config": request.scoring_config,
            "output_formats": request.output_formats,
            "start_time": datetime.now().isoformat(),
            "progress": {
                "scoring_jobs_processed": 0,
                "total_scoring_jobs": len(request.scoring_config.get("scoring_jobs", [])),
                "current_job": None
            }
        }
        
        # Store job info
        self.active_jobs[job_id] = job_info
        
        # Start processing in background
        asyncio.create_task(self._process_scoring_job(job_id))
        
        return job_id
    
    async def _process_scoring_job(self, job_id: str):
        """Process scoring job in background"""
        try:
            job_info = self.active_jobs[job_id]
            job_info["status"] = "processing"
            
            # Load source data
            source_data = self._load_source_data(
                job_info["source_type"], 
                job_info["source_job_id"]
            )
            
            if source_data is None:
                raise Exception(f"Could not load source data for {job_info['source_type']} job {job_info['source_job_id']}")
            
            # Process scoring jobs
            features = source_data.get("features", [])
            scoring_jobs = job_info["scoring_config"].get("scoring_jobs", [])
            added_scores = []
            
            for i, scoring_job in enumerate(scoring_jobs):
                job_name = scoring_job.get("name", f"job_{i}")
                scorer_name = scoring_job.get("scorer", "")
                job_params = scoring_job.get("parameters", {})
                
                # Update progress
                job_info["progress"]["current_job"] = job_name
                job_info["progress"]["scoring_jobs_processed"] = i
                
                if scorer_name not in self.available_scorers:
                    logger.warning(f"Scorer '{scorer_name}' not available. Skipping job '{job_name}'")
                    continue
                
                try:
                    # Process scoring job here (simplified for demo)
                    logger.info(f"Processing scoring job: {job_name} using {scorer_name}")
                    added_scores.append(job_name)
                    
                except Exception as e:
                    logger.error(f"Scoring job '{job_name}' failed: {e}")
                    continue
            
            # Save results
            results = {
                "job_id": job_id,
                "source_type": job_info["source_type"],
                "source_job_id": job_info["source_job_id"],
                "timestamp": datetime.now().isoformat(),
                "features": features,
                "scores_added": added_scores,
                "scoring_config": job_info["scoring_config"]
            }
            
            # Create output directory
            output_dir = self.config.score_results_dir / job_id
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Save results
            with open(output_dir / "scoring_results.json", 'w') as f:
                json.dump(results, f, indent=2)
            
            # Mark job as completed
            job_info["status"] = "completed"
            job_info["completion_time"] = datetime.now().isoformat()
            job_info["results"] = results
            job_info["results_path"] = str(output_dir)
            
            # Move to completed jobs
            self.completed_jobs[job_id] = job_info
            del self.active_jobs[job_id]
            
            logger.info(f"✅ Scoring job {job_id} completed successfully")
            
        except Exception as e:
            logger.error(f"Error processing scoring job {job_id}: {e}")
            job_info["status"] = "failed"
            job_info["error"] = str(e)
            job_info["completion_time"] = datetime.now().isoformat()
            
            # Move to completed jobs
            self.completed_jobs[job_id] = job_info
            if job_id in self.active_jobs:
                del self.active_jobs[job_id]
    
    def _load_source_data(self, source_type: str, source_job_id: str) -> Optional[Dict[str, Any]]:
        """Load source data from Find or Explain results"""
        try:
            if source_type == "find":
                source_file = self.config.find_results_dir / source_job_id / "analysis_results.json"
            elif source_type == "explain":
                source_file = self.config.explain_results_dir / source_job_id / "explanation_results.json"
            else:
                logger.error(f"Unknown source type: {source_type}")
                return None
            
            if not source_file.exists():
                logger.error(f"Source file not found: {source_file}")
                return None
            
            with open(source_file, 'r') as f:
                return json.load(f)
                
        except Exception as e:
            logger.error(f"Error loading source data: {e}")
            return None
    
    def get_job_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get status of scoring job"""
        if job_id in self.active_jobs:
            return self.active_jobs[job_id]
        elif job_id in self.completed_jobs:
            return self.completed_jobs[job_id]
        else:
            return None
    
    def get_job_results(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get results of completed scoring job"""
        if job_id in self.completed_jobs:
            job_info = self.completed_jobs[job_id]
            if job_info["status"] == "completed":
                return job_info.get("results")
        return None
    
    def list_jobs(self) -> List[Dict[str, Any]]:
        """List all scoring jobs"""
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
        data_path=str(config.data_path),
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
            status=status_data["status"],
            source_type=status_data["source_type"],
            source_job_id=status_data["source_job_id"],
            start_time=status_data.get("start_time"),
            completion_time=status_data.get("completion_time"),
            processing_time=processing_time,
            progress=JobProgress(
                scoring_jobs_processed=progress_data.get("scoring_jobs_processed", 0),
                total_scoring_jobs=progress_data.get("total_scoring_jobs", 0),
                current_job=progress_data.get("current_job")
            ) if progress_data else None,
            error=status_data.get("error"),
            results_path=status_data.get("results_path")
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting job status: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/score/{job_id}/results", response_model=JobResultResponse)
async def get_job_results(job_id: str):
    """Get results of completed scoring job"""
    
    try:
        results = score_service.get_job_results(job_id)
        status_data = score_service.get_job_status(job_id)
        
        if results is None or status_data is None:
            raise HTTPException(status_code=404, detail=f"Job {job_id} results not found")
        
        if status_data["status"] != "completed":
            raise HTTPException(status_code=400, detail=f"Job {job_id} is not completed yet")
        
        # Calculate processing time
        processing_time = None
        if status_data.get("start_time") and status_data.get("completion_time"):
            start = datetime.fromisoformat(status_data["start_time"])
            end = datetime.fromisoformat(status_data["completion_time"])
            processing_time = (end - start).total_seconds()
        
        return JobResultResponse(
            job_id=job_id,
            status=status_data["status"],
            source_type=status_data["source_type"],
            source_job_id=status_data["source_job_id"],
            scores_added=results.get("scores_added", []),
            results_summary={
                "total_features": len(results.get("features", [])),
                "scores_added": len(results.get("scores_added", [])),
                "processing_time": processing_time
            },
            download_links={
                "json": f"/api/v1/score/{job_id}/download/json",
                "csv": f"/api/v1/score/{job_id}/download/csv"
            },
            processing_time=processing_time
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting job results: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/score/jobs", response_model=JobListResponse)
async def list_jobs():
    """List all scoring jobs"""
    
    try:
        jobs = score_service.list_jobs()
        
        job_summaries = []
        for job in jobs:
            job_summaries.append(JobSummary(
                job_id=job["job_id"],
                status=job["status"],
                source_type=job["source_type"],
                source_job_id=job["source_job_id"],
                created_at=job["start_time"],
                completed_at=job.get("completion_time")
            ))
        
        return JobListResponse(
            jobs=job_summaries,
            total=len(job_summaries)
        )
        
    except Exception as e:
        logger.error(f"Error listing jobs: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# =============================================================================
# Development Server Entry Point
# =============================================================================

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8004,
        reload=True,
        log_level="info"
    )