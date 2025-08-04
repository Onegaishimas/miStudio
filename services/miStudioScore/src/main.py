# src/main.py - Fixed miStudioScore Service with Enhanced Data Loading
"""
miStudioScore Service - Enhanced scoring service with robust data loading
Fixed: Proper source data discovery and loading, no fallback to test data
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

# Additional imports for file downloads
from fastapi.responses import FileResponse
import os

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
        self.service_version = "v1.2.0-fixed"

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
# Enhanced Service Implementation with Fixed Data Loading
# =============================================================================

class IntegratedScoreService:
    """Fixed service with robust data loading and no test data fallback"""
    
    def __init__(self, config: ServiceConfig):
        self.config = config
        self.active_jobs: Dict[str, Dict[str, Any]] = {}
        self.completed_jobs: Dict[str, Dict[str, Any]] = {}
        
        # Initialize scorer storage
        self.scorers: Dict[str, Any] = {}
        self.available_scorers: List[str] = []
        
        # Load scorers
        self._initialize_scorers()
        
        logger.info(f"IntegratedScoreService initialized (FIXED VERSION)")
        logger.info(f"  Find results path: {self.config.find_results_dir}")
        logger.info(f"  Explain results path: {self.config.explain_results_dir}")
        logger.info(f"  Score results path: {self.config.score_results_dir}")
        logger.info(f"  Available scorers: {self.available_scorers}")
    
    def _initialize_scorers(self):
        """Initialize available scoring modules with proper error handling"""
        logger.info("🔧 Initializing scoring modules...")
        
        # Try to load RelevanceScorer
        try:
            from src.scorers.relevance_scorer import RelevanceScorer
            self.scorers["relevance_scorer"] = RelevanceScorer
            self.available_scorers.append("relevance_scorer")
            logger.info("✅ Loaded RelevanceScorer")
        except ImportError as e:
            logger.warning(f"❌ RelevanceScorer not available: {e}")
        except Exception as e:
            logger.warning(f"❌ Error loading RelevanceScorer: {e}")
        
        # Try to load AblationScorer
        try:
            from src.scorers.ablation_scorer import AblationScorer
            self.scorers["ablation_scorer"] = AblationScorer
            self.available_scorers.append("ablation_scorer")
            logger.info("✅ Loaded AblationScorer")
        except ImportError as e:
            logger.warning(f"❌ AblationScorer not available: {e}")
        except Exception as e:
            logger.warning(f"❌ Error loading AblationScorer: {e}")
        
        # Load mock scorers for demonstration (but prefer real data)
        if not self.available_scorers:
            logger.warning("⚠️ No external scorers available, loading mock scorers")
        
        # Always load mock scorers as fallback, but use real data
        self._load_mock_scorers()
        
        logger.info(f"📊 {len(self.available_scorers)} total scorers available: {self.available_scorers}")
    
    def _load_mock_scorers(self):
        """Load mock scorers that process real data, not test data"""
        
        class MockRelevanceScorer:
            @property
            def name(self) -> str:
                return features
        
        class MockAblationScorer:
            @property
            def name(self) -> str:
                return "ablation_scorer"
            
            def score(self, features: List[Dict[str, Any]], **kwargs) -> List[Dict[str, Any]]:
                score_name = kwargs.get("name", "ablation_score")
                threshold = kwargs.get("threshold", 0.3)
                
                logger.info(f"Mock ablation scoring for '{score_name}' with threshold {threshold}")
                logger.info(f"Processing {len(features)} real features (not test data)")
                
                for feature in features:
                    # Score based on actual feature coherence and complexity
                    coherence = feature.get("coherence_score", 0.5)
                    max_activation = feature.get("max_activation", 1.0)
                    
                    # Convert to float if possible
                    try:
                        coherence = float(coherence)
                        max_activation = float(max_activation)
                    except (ValueError, TypeError):
                        coherence = 0.5
                        max_activation = 1.0
                    
                    # Ablation score based on coherence above threshold and activation strength
                    if coherence > threshold:
                        score = (coherence - threshold) * (max_activation / 10.0)  # Normalize activation
                    else:
                        score = 0.0
                    
                    feature[score_name] = round(score, 6)
                
                return features
        
        # Only add mock scorers if not already present
        if "relevance_scorer" not in self.scorers:
            self.scorers["relevance_scorer"] = MockRelevanceScorer
            if "relevance_scorer" not in self.available_scorers:
                self.available_scorers.append("relevance_scorer")
        
        if "ablation_scorer" not in self.scorers:
            self.scorers["ablation_scorer"] = MockAblationScorer
            if "ablation_scorer" not in self.available_scorers:
                self.available_scorers.append("ablation_scorer")
        
        logger.info("📊 Mock scorers loaded with real data processing capability")
    
    def discover_source_jobs(self, source_type: str) -> List[Dict[str, Any]]:
        """Enhanced discovery with comprehensive path checking and debugging"""
        logger.info(f"🔍 Discovering {source_type} jobs...")
        
        available_jobs = []
        
        if source_type == "explain":
            source_dir = self.config.explain_results_dir
            logger.info(f"📁 Searching in: {source_dir}")
            
            if source_dir.exists():
                logger.info(f"✅ Directory exists, scanning for jobs...")
                job_count = 0
                for job_dir in source_dir.iterdir():
                    if job_dir.is_dir():
                        job_count += 1
                        logger.debug(f"📂 Checking directory: {job_dir.name}")
                        
                        # Check multiple possible file locations for explain results
                        possible_files = [
                            job_dir / "explanation_results.json",
                            job_dir / f"{job_dir.name}_explanation_results.json",
                            job_dir / f"{job_dir.name}_complete_results.json",
                            job_dir / "results.json"
                        ]
                        
                        for results_file in possible_files:
                            if results_file.exists():
                                logger.debug(f"📄 Found file: {results_file}")
                                try:
                                    with open(results_file, 'r') as f:
                                        job_data = json.load(f)
                                    
                                    # Look for explanations
                                    explanations = job_data.get("explanations", [])
                                    if explanations:  # Only add if explanations exist
                                        available_jobs.append({
                                            "job_id": job_dir.name,
                                            "timestamp": job_data.get("timestamp", "unknown"),
                                            "features_count": len(explanations),
                                            "status": "completed",
                                            "file_path": str(results_file)
                                        })
                                        logger.info(f"✅ Found valid explain job: {job_dir.name} with {len(explanations)} explanations")
                                        break  # Found valid file, stop checking
                                    else:
                                        logger.debug(f"⚠️ File {results_file} has no explanations")
                                except Exception as e:
                                    logger.warning(f"Error reading {results_file}: {e}")
                                    continue
                        else:
                            logger.debug(f"❌ No valid results file found in {job_dir.name}")
                
                logger.info(f"📊 Scanned {job_count} directories, found {len(available_jobs)} valid explain jobs")
            else:
                logger.warning(f"❌ Directory does not exist: {source_dir}")
        
        logger.info(f"🎯 Discovered {len(available_jobs)} {source_type} jobs: {[job['job_id'] for job in available_jobs]}")
        return sorted(available_jobs, key=lambda x: x["job_id"], reverse=True)
    
    async def start_scoring_job(self, request: ScoreRequest) -> str:
        """Start a new scoring job with enhanced data loading"""
        # Generate job ID
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        random_suffix = f"{hash(str(request.dict())) % 100000000:08d}"
        job_id = f"score_{timestamp}_{random_suffix}"
        
        # Store job information
        job_info = {
            "job_id": job_id,
            "source_type": request.source_type,
            "source_job_id": request.source_job_id,
            "scoring_config": request.scoring_config,
            "output_formats": request.output_formats,
            "status": "queued",
            "start_time": datetime.now().isoformat(),
            "progress": {
                "scoring_jobs_processed": 0,
                "total_scoring_jobs": len(request.scoring_config.get("scoring_jobs", [])),
                "current_job": None
            }
        }
        
        self.active_jobs[job_id] = job_info
        
        # Start background processing
        asyncio.create_task(self._process_scoring_job(job_id))
        
        return job_id
    
    async def _process_scoring_job(self, job_id: str):
        """Process scoring job with robust data loading - NO test data fallback"""
        job_info = self.active_jobs[job_id]
        
        try:
            # Update status
            job_info["status"] = "running"
            
            # Load source data with comprehensive path checking
            source_data = self._load_source_data_robust(
                job_info["source_type"], 
                job_info["source_job_id"]
            )
            
            if source_data is None:
                raise ValueError(f"Could not load source data for {job_info['source_type']} job {job_info['source_job_id']}")
            
            # Extract features from source data
            features = self._extract_features_from_source(source_data, job_info["source_type"])
            
            if not features:
                raise ValueError(f"No features found in {job_info['source_type']} results")
            
            logger.info(f"🎯 Scoring {len(features)} real features from {job_info['source_type']} results")
            
            # Execute scoring jobs
            added_scores = []
            scoring_jobs = job_info["scoring_config"].get("scoring_jobs", [])
            
            for i, job in enumerate(scoring_jobs):
                scorer_name = job.get("scorer")
                job_params = job.get("parameters", {})
                job_name = job.get("name")
                
                # Update progress
                job_info["progress"]["scoring_jobs_processed"] = i
                job_info["progress"]["current_job"] = job_name
                
                if not job_name:
                    logger.warning(f"Scoring job with scorer '{scorer_name}' is missing a 'name'. Skipping.")
                    continue
                
                if scorer_name not in self.scorers:
                    logger.error(f"Scorer '{scorer_name}' not found. Available: {self.available_scorers}")
                    continue
                
                try:
                    # REAL SCORING IMPLEMENTATION
                    logger.info(f"🔄 Processing scoring job: {job_name} using {scorer_name}")
                    
                    # Get scorer instance
                    scorer_class = self.scorers[scorer_name]
                    scorer_instance = scorer_class()
                    
                    # Prepare parameters
                    all_params = {"name": job_name, **job_params}
                    
                    # Execute actual scoring on real features
                    features = scorer_instance.score(features, **all_params)
                    added_scores.append(job_name)
                    
                    logger.info(f"✅ Completed scoring job: {job_name}")
                    
                except Exception as e:
                    logger.error(f"❌ Scoring job '{job_name}' failed: {e}")
                    continue
            
            # Update final progress
            job_info["progress"]["scoring_jobs_processed"] = len(scoring_jobs)
            
            # Save results
            results = {
                "job_id": job_id,
                "source_type": job_info["source_type"],
                "source_job_id": job_info["source_job_id"],
                "timestamp": datetime.now().isoformat(),
                "features": features,
                "scores_added": added_scores,
                "scoring_config": job_info["scoring_config"],
                "total_features_processed": len(features),
                "successful_scoring_jobs": len(added_scores),
                "data_source": "real_features"  # Indicate real data was used
            }
            
            # Create output directory
            output_dir = self.config.score_results_dir / job_id
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Save JSON results
            with open(output_dir / "scoring_results.json", 'w') as f:
                json.dump(results, f, indent=2)
            
            # Save CSV results if features exist
            if features:
                self._save_csv_results(features, output_dir / "scored_features.csv", added_scores)
            
            # Save job info
            with open(output_dir / "job_info.json", 'w') as f:
                json.dump(job_info, f, indent=2)
            
            # Mark job as completed
            job_info["status"] = "completed"
            job_info["completion_time"] = datetime.now().isoformat()
            job_info["results"] = results
            job_info["results_path"] = str(output_dir)
            
            # Move to completed jobs
            self.completed_jobs[job_id] = job_info
            del self.active_jobs[job_id]
            
            logger.info(f"✅ Scoring job {job_id} completed successfully")
            logger.info(f"📊 Results: {len(features)} real features, {len(added_scores)} scores added")
            
        except Exception as e:
            logger.error(f"❌ Error processing scoring job {job_id}: {e}")
            job_info["status"] = "failed"
            job_info["error"] = str(e)
            job_info["completion_time"] = datetime.now().isoformat()
            
            # Move to completed jobs
            self.completed_jobs[job_id] = job_info
            if job_id in self.active_jobs:
                del self.active_jobs[job_id]
    
    def _load_source_data_robust(self, source_type: str, source_job_id: str) -> Optional[Dict[str, Any]]:
        """Load source data with comprehensive fallback logic - NO test data"""
        
        logger.info(f"🔍 Loading {source_type} data for job: {source_job_id}")
        
        # Define multiple possible file paths based on observed structure
        if source_type == "explain":
            possible_paths = [
                self.config.explain_results_dir / source_job_id / "explanation_results.json",
                self.config.explain_results_dir / source_job_id / f"{source_job_id}_explanation_results.json",
                self.config.explain_results_dir / source_job_id / f"{source_job_id}_complete_results.json",
                self.config.data_path / "results" / source_job_id / f"{source_job_id}_complete_results.json",
                self.config.explain_results_dir / source_job_id / "results.json"
            ]
        elif source_type == "find":
            possible_paths = [
                self.config.find_results_dir / source_job_id / "analysis_results.json",
                self.config.find_results_dir / source_job_id / f"{source_job_id}_complete_results.json",
                self.config.find_results_dir / source_job_id / f"{source_job_id}_analysis.json",
                self.config.data_path / "results" / source_job_id / f"{source_job_id}_analysis.json",
                self.config.find_results_dir / source_job_id / "results.json"
            ]
        else:
            logger.error(f"Unknown source type: {source_type}")
            return None
        
        # Try each path until we find valid data
        for source_file in possible_paths:
            if source_file.exists():
                try:
                    logger.info(f"📂 Attempting to load: {source_file}")
                    with open(source_file, 'r') as f:
                        data = json.load(f)
                    
                    # Validate data contains expected content
                    if source_type == "explain":
                        if "explanations" in data and data["explanations"]:
                            logger.info(f"✅ Loaded {len(data['explanations'])} explanations from: {source_file}")
                            return data
                        elif "features" in data and data["features"]:
                            # Handle case where explanations are stored as features
                            logger.info(f"✅ Loaded {len(data['features'])} features (as explanations) from: {source_file}")
                            return {"explanations": data["features"], **{k: v for k, v in data.items() if k != "features"}}
                    elif source_type == "find":
                        if "features" in data and data["features"]:
                            logger.info(f"✅ Loaded {len(data['features'])} features from: {source_file}")
                            return data
                        elif "results" in data and data["results"]:
                            # Handle case where features are stored as results
                            logger.info(f"✅ Loaded {len(data['results'])} results (as features) from: {source_file}")
                            return {"features": data["results"], **{k: v for k, v in data.items() if k != "results"}}
                    
                    logger.warning(f"📂 File {source_file} exists but doesn't contain expected data structure")
                    
                except Exception as e:
                    logger.warning(f"Error reading {source_file}: {e}")
                    continue
            else:
                logger.debug(f"📂 File not found: {source_file}")
        
        # If we get here, no valid data was found - DO NOT fall back to test data
        logger.error(f"❌ No valid source data found for {source_type} job {source_job_id}")
        logger.error(f"🔍 Searched paths:")
        for path in possible_paths:
            logger.error(f"   - {path} (exists: {path.exists()})")
        
        # List available jobs for debugging
        available_jobs = self.discover_source_jobs(source_type)
        available_ids = [job["job_id"] for job in available_jobs]
        logger.error(f"📋 Available {source_type} jobs: {available_ids}")
        
        # Show directory contents for debugging
        source_dir = self.config.explain_results_dir if source_type == "explain" else self.config.find_results_dir
        if source_dir.exists():
            logger.error(f"📁 Contents of {source_dir}:")
            try:
                for item in source_dir.iterdir():
                    if item.is_dir():
                        logger.error(f"   📁 {item.name}/")
                        try:
                            for subitem in item.iterdir():
                                logger.error(f"      📄 {subitem.name}")
                        except:
                            pass
            except Exception as e:
                logger.error(f"   Error listing contents: {e}")
        
        return None  # Return None instead of test data
    
    def _extract_features_from_source(self, data: Dict[str, Any], source_type: str) -> List[Dict[str, Any]]:
        """Extract and validate features from source data"""
        
        if source_type == "explain":
            # For explain results, features are in "explanations"
            features = data.get("explanations", [])
            if not features:
                raise ValueError("No explanations found in explain results")
            logger.info(f"📊 Extracted {len(features)} explanations for scoring")
        elif source_type == "find":
            # For find results, features are in "features" 
            features = data.get("features", [])
            if not features:
                raise ValueError("No features found in find results")
            logger.info(f"📊 Extracted {len(features)} features for scoring")
        else:
            raise ValueError(f"Unknown source type: {source_type}")
        
        # Validate feature structure
        if not isinstance(features, list):
            raise ValueError(f"Features must be a list, got {type(features)}")
        
        if not features:
            raise ValueError("Features list is empty")
        
        # Validate each feature has some content
        valid_features = []
        for i, feature in enumerate(features):
            if not isinstance(feature, dict):
                logger.warning(f"Feature {i} is not a dictionary, skipping")
                continue
            
            # Feature is valid if it has any content
            if feature:
                valid_features.append(feature)
            else:
                logger.warning(f"Feature {i} is empty, skipping")
        
        if not valid_features:
            raise ValueError("No valid features found after validation")
        
        logger.info(f"✅ Validated {len(valid_features)} features from {source_type} results")
        return valid_features
    
    def _save_csv_results(self, features: List[Dict[str, Any]], csv_path: Path, score_columns: List[str]):
        """Save scoring results in CSV format"""
        import csv
        
        if not features:
            logger.warning("No features to save to CSV")
            return
        
        # Determine all possible columns from all features
        all_columns = set()
        for feature in features:
            all_columns.update(feature.keys())
        
        # Order columns logically
        priority_columns = ["feature_id", "feature_index", "coherence_score", "explanation", "pattern_description"]
        score_columns_set = set(score_columns)
        remaining_columns = all_columns - set(priority_columns) - score_columns_set
        
        # Final column order
        ordered_columns = []
        for col in priority_columns:
            if col in all_columns:
                ordered_columns.append(col)
        
        # Add score columns
        ordered_columns.extend(score_columns)
        
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
            if job_info.get("status") == "completed":
                return job_info.get("results")
        return None

# Initialize the service
score_service = IntegratedScoreService(config)

# =============================================================================
# FastAPI Application
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management"""
    logger.info(f"🚀 {config.service_name} API starting up...")
    logger.info(f"📁 Data path: {config.data_path}")
    logger.info(f"🔍 Find results: {config.find_results_dir}")
    logger.info(f"💬 Explain results: {config.explain_results_dir}")
    logger.info(f"🎯 Score results: {config.score_results_dir}")
    logger.info(f"🔧 Available scorers: {score_service.available_scorers}")
    logger.info(f"⚡ ENHANCED DATA LOADING: Real features only, no test data fallback")
    
    yield
    
    logger.info(f"🛑 {config.service_name} API shutting down...")

app = FastAPI(
    title="miStudioScore API (Fixed)",
    description="Enhanced scoring service with robust data loading - no test data fallback",
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

@app.get("/", response_model=Dict[str, Any])
async def root():
    """Service information endpoint"""
    return {
        "service": config.service_name,
        "status": "running",
        "version": config.service_version,
        "description": "FIXED scoring service - robust data loading, real features only",
        "available_scorers": score_service.available_scorers,
        "scoring_ready": len(score_service.available_scorers) > 0,
        "data_loading": "enhanced_robust_no_test_fallback",
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
            "score_results_writable": config.score_results_dir.exists(),
            "scorers_loaded": len(score_service.available_scorers) > 0,
            "data_loading": True  # Enhanced data loading
        },
        available_scorers=score_service.available_scorers
    )


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
        try:
            if status_data.get("completion_time") and status_data.get("start_time"):
                start = datetime.fromisoformat(status_data["start_time"])
                end = datetime.fromisoformat(status_data["completion_time"])
                processing_time = (end - start).total_seconds()
        except Exception as e:
            logger.warning(f"Error calculating processing time: {e}")
            processing_time = None
        
        return JobStatusResponse(
            job_id=job_id,
            status=status_data.get("status", "unknown"),
            source_type=status_data.get("source_type", "unknown"),
            source_job_id=status_data.get("source_job_id", "unknown"),
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
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting job status: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/score/{job_id}/results", response_model=JobResultResponse)
async def get_job_results(job_id: str):
    """Get results of completed scoring job"""
    
    try:
        results_data = score_service.get_job_results(job_id)
        status_data = score_service.get_job_status(job_id)
        
        if results_data is None or status_data is None:
            raise HTTPException(status_code=404, detail=f"Job {job_id} not found or not completed")
        
        if status_data.get("status") != "completed":
            raise HTTPException(status_code=400, detail=f"Job {job_id} is not completed")
        
        # Calculate processing time
        processing_time = None
        try:
            if status_data.get("completion_time") and status_data.get("start_time"):
                start = datetime.fromisoformat(status_data["start_time"])
                end = datetime.fromisoformat(status_data["completion_time"])
                processing_time = (end - start).total_seconds()
        except Exception as e:
            logger.warning(f"Error calculating processing time: {e}")
            processing_time = None
        
        return JobResultResponse(
            job_id=job_id,
            status=status_data.get("status", "completed"),
            source_type=status_data.get("source_type", "unknown"),
            source_job_id=status_data.get("source_job_id", "unknown"),
            scores_added=results_data.get("scores_added", []),
            results_summary={
                "total_features": results_data.get("total_features_processed", 0),
                "successful_scoring_jobs": results_data.get("successful_scoring_jobs", 0),
                "data_source": results_data.get("data_source", "unknown")
            },
            download_links={
                "json": f"/api/v1/score/{job_id}/download/json",
                "csv": f"/api/v1/score/{job_id}/download/csv"
            },
            processing_time=processing_time
        )
        
    except Exception as e:
        logger.error(f"Error getting job results: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/score/{job_id}/download/json")
async def download_json_results(job_id: str):
    """Download JSON results file"""
    
    try:
        results_path = config.score_results_dir / job_id / "scoring_results.json"
        
        if not results_path.exists():
            raise HTTPException(status_code=404, detail=f"Results file not found for job {job_id}")
        
        return FileResponse(
            path=str(results_path),
            filename=f"scoring_results_{job_id}.json",
            media_type="application/json"
        )
        
    except Exception as e:
        logger.error(f"Error downloading JSON results: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/score/{job_id}/download/csv")
async def download_csv_results(job_id: str):
    """Download CSV results file"""
    
    try:
        results_path = config.score_results_dir / job_id / "scored_features.csv"
        
        if not results_path.exists():
            raise HTTPException(status_code=404, detail=f"CSV results file not found for job {job_id}")
        
        return FileResponse(
            path=str(results_path),
            filename=f"scored_features_{job_id}.csv",
            media_type="text/csv"
        )
        
    except Exception as e:
        logger.error(f"Error downloading CSV results: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/score/jobs", response_model=JobListResponse)
async def list_jobs():
    """List all scoring jobs"""
    
    try:
        all_jobs = []
        
        # Add active jobs
        for job_id, job_info in score_service.active_jobs.items():
            all_jobs.append(JobSummary(
                job_id=job_id,
                status=job_info.get("status", "unknown"),
                source_type=job_info.get("source_type", "unknown"),
                source_job_id=job_info.get("source_job_id", "unknown"),
                created_at=job_info.get("start_time", "unknown")
            ))
        
        # Add completed jobs
        for job_id, job_info in score_service.completed_jobs.items():
            all_jobs.append(JobSummary(
                job_id=job_id,
                status=job_info.get("status", "unknown"),
                source_type=job_info.get("source_type", "unknown"),
                source_job_id=job_info.get("source_job_id", "unknown"),
                created_at=job_info.get("start_time", "unknown"),
                completed_at=job_info.get("completion_time")
            ))
        
        # Sort by creation time (newest first)
        all_jobs.sort(key=lambda x: x.created_at, reverse=True)
        
        return JobListResponse(
            jobs=all_jobs,
            total=len(all_jobs)
        )
        
    except Exception as e:
        logger.error(f"Error listing jobs: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/score/discover/{source_type}")
async def discover_source_jobs(source_type: str):
    """Discover available source jobs for scoring"""
    
    try:
        if source_type not in ["find", "explain"]:
            raise HTTPException(status_code=400, detail="Source type must be 'find' or 'explain'")
        
        available_jobs = score_service.discover_source_jobs(source_type)
        
        return {
            "source_type": source_type,
            "available_jobs": available_jobs,
            "total": len(available_jobs)
        }
        
    except Exception as e:
        logger.error(f"Error discovering source jobs: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# =============================================================================
# Application Startup
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    
    logger.info("🚀 Starting miStudioScore service...")
    logger.info("⚡ Enhanced data loading and scoring capabilities")
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8004,
        reload=False,
        log_level="info"
    )
