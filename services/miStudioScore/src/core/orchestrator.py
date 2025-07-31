# src/core/orchestrator.py - Updated with Shared Storage Integration
"""
Enhanced orchestrator for miStudioScore with shared storage integration.
"""

import logging
import json
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


class IntegratedOrchestrator:
    """Enhanced orchestrator with shared storage integration"""
    
    def __init__(self, data_path: str = "/data"):
        """Initialize orchestrator with shared data path"""
        self.data_path = Path(data_path)
        self.find_results_dir = self.data_path / "results" / "find"
        self.explain_results_dir = self.data_path / "results" / "explain"
        self.score_results_dir = self.data_path / "results" / "score"
        
        # Ensure directories exist
        for directory in [self.find_results_dir, self.explain_results_dir, self.score_results_dir]:
            directory.mkdir(parents=True, exist_ok=True)
        
        # Initialize available scorers
        self.scorers = {}
        self._load_scorers()
        
        logger.info(f"IntegratedOrchestrator initialized with data path: {self.data_path}")
        logger.info(f"Available scorers: {list(self.scorers.keys())}")
    
    def _load_scorers(self):
        """Load available scoring modules"""
        try:
            from core.scoring.relevance_scorer import RelevanceScorer
            self.scorers["relevance_scorer"] = RelevanceScorer
            logger.info("✅ Loaded RelevanceScorer")
        except ImportError:
            logger.warning("RelevanceScorer not available")
        
        try:
            from core.scoring.ablation_scorer import AblationScorer
            self.scorers["ablation_scorer"] = AblationScorer
            logger.info("✅ Loaded AblationScorer")
        except ImportError:
            logger.warning("AblationScorer not available")
        
        # Add built-in pattern scorer
        self.scorers["pattern_scorer"] = PatternScorer
        logger.info("✅ Loaded PatternScorer (built-in)")
    
    def discover_source_jobs(self, source_type: str) -> List[Dict[str, Any]]:
        """Discover available source jobs for scoring"""
        available_jobs = []
        
        if source_type == "find":
            # Check main Find results directory
            if self.find_results_dir.exists():
                for job_dir in self.find_results_dir.iterdir():
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
                                    "results_path": str(results_file),
                                    "location": "main"
                                })
                            except Exception as e:
                                logger.warning(f"Could not read Find job {job_dir.name}: {e}")
            
            # Check enhanced persistence directories
            results_base = self.data_path / "results"
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
                                    "results_path": str(results_file),
                                    "location": "enhanced"
                                })
                            except Exception as e:
                                logger.warning(f"Could not read enhanced Find job {job_dir.name}: {e}")
        
        elif source_type == "explain":
            if self.explain_results_dir.exists():
                for job_dir in self.explain_results_dir.iterdir():
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
                                    "results_path": str(results_file),
                                    "location": "main"
                                })
                            except Exception as e:
                                logger.warning(f"Could not read Explain job {job_dir.name}: {e}")
        
        logger.info(f"Discovered {len(available_jobs)} available {source_type} jobs")
        return available_jobs
    
    def load_source_data(self, source_type: str, source_job_id: str) -> List[Dict[str, Any]]:
        """Load source data and convert to features format"""
        if source_type == "find":
            # Try main results directory first
            main_path = self.find_results_dir / source_job_id / "analysis_results.json"
            
            if main_path.exists():
                with open(main_path, 'r') as f:
                    data = json.load(f)
                features = data.get('results', [])
                logger.info(f"Loaded {len(features)} features from Find job: {main_path}")
                return features
            
            # Try enhanced persistence directory
            enhanced_path = self.data_path / "results" / source_job_id / f"{source_job_id}_complete_results.json"
            
            if enhanced_path.exists():
                with open(enhanced_path, 'r') as f:
                    data = json.load(f)
                features = data.get('results', [])
                logger.info(f"Loaded {len(features)} features from enhanced Find job: {enhanced_path}")
                return features
        
        elif source_type == "explain":
            explain_path = self.explain_results_dir / source_job_id / "explanation_results.json"
            
            if explain_path.exists():
                with open(explain_path, 'r') as f:
                    data = json.load(f)
                explanations = data.get('explanations', [])
                
                # Convert explanations to features format
                features = []
                for exp in explanations:
                    feature = {
                        "feature_id": exp.get("feature_id"),
                        "coherence_score": exp.get("coherence_score", 0),
                        "pattern_keywords": exp.get("pattern_keywords", []),
                        "explanation": exp.get("explanation", ""),
                        "confidence": exp.get("confidence", "unknown"),
                        **exp  # Include all other fields
                    }
                    features.append(feature)
                
                logger.info(f"Loaded {len(features)} features from Explain job: {explain_path}")
                return features
        
        # List available jobs for error message
        available_jobs = self.discover_source_jobs(source_type)
        available_ids = [job["source_job_id"] for job in available_jobs]
        
        raise FileNotFoundError(f"{source_type.title()} job {source_job_id} not found. Available: {available_ids}")
    
    def execute_scoring_pipeline(self, source_type: str, source_job_id: str, 
                                config: Dict[str, Any]) -> Tuple[Optional[str], List[str]]:
        """Execute the complete scoring pipeline with shared storage integration"""
        try:
            # Load source features
            features = self.load_source_data(source_type, source_job_id)
            if not features:
                logger.error(f"No features found in {source_type} job {source_job_id}")
                return None, []
            
            logger.info(f"Starting scoring pipeline with {len(features)} features")
            
            # Process scoring jobs
            added_scores = []
            scoring_jobs = config.get("scoring_jobs", [])
            
            if not scoring_jobs:
                logger.warning("No scoring jobs found in configuration")
                return None, []
            
            for job in scoring_jobs:
                scorer_name = job.get("scorer")
                job_params = job.get("params", {})
                job_name = job.get("name")
                
                if not job_name:
                    logger.warning(f"Scoring job with scorer '{scorer_name}' is missing a 'name'. Skipping.")
                    continue
                
                if scorer_name in self.scorers:
                    logger.info(f"Executing job '{job_name}' with scorer '{scorer_name}'...")
                    scorer_class = self.scorers[scorer_name]
                    scorer_instance = scorer_class()
                    
                    # Pass all params to the scorer instance, including the job name
                    all_params = {"name": job_name, **job_params}
                    features = scorer_instance.score(features, **all_params)
                    added_scores.append(job_name)
                else:
                    logger.warning(f"Scorer '{scorer_name}' not found. Skipping job.")
            
            # Generate unique job ID for this scoring run
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            job_id = f"score_{timestamp}_{source_job_id}"
            
            # Save the enriched data to shared storage
            output_path = self._save_results_to_shared_storage(
                job_id, source_type, source_job_id, features, added_scores, config
            )
            
            logger.info(f"✅ Scoring pipeline completed. Added scores: {added_scores}")
            return output_path, added_scores
            
        except Exception as e:
            logger.error(f"Scoring pipeline failed: {e}")
            raise
    
    def _save_results_to_shared_storage(self, job_id: str, source_type: str, 
                                      source_job_id: str, features: List[Dict[str, Any]], 
                                      added_scores: List[str], config: Dict[str, Any]) -> str:
        """Save results to shared storage with comprehensive format support"""
        
        # Create job-specific directory
        job_dir = self.score_results_dir / job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        
        # Create comprehensive results structure
        results_data = {
            "job_id": job_id,
            "source_type": source_type,
            "source_job_id": source_job_id,
            "status": "completed",
            "timestamp": datetime.now().isoformat(),
            "processing_summary": {
                "total_features": len(features),
                "scores_added": added_scores,
                "scoring_config": config
            },
            "features": features,
            "metadata": {
                "service": "miStudioScore",
                "version": "1.0.0",
                "source_type": source_type,
                "source_job_id": source_job_id
            }
        }
        
        # Save main JSON results
        json_path = job_dir / "scoring_results.json"
        with open(json_path, 'w') as f:
            json.dump(results_data, f, indent=2, default=str)
        
        # Save CSV format
        csv_path = job_dir / "scored_features.csv"
        self._save_csv_format(features, added_scores, csv_path)
        
        # Save summary report
        summary_path = job_dir / "summary_report.txt"
        self._save_summary_report(job_id, source_type, source_job_id, 
                                features, added_scores, summary_path)
        
        # Save job metadata
        metadata_path = job_dir / "job_info.json"
        job_metadata = {
            "job_id": job_id,
            "source_type": source_type,
            "source_job_id": source_job_id,
            "timestamp": datetime.now().isoformat(),
            "files_created": {
                "json": str(json_path),
                "csv": str(csv_path),
                "summary": str(summary_path),
                "metadata": str(metadata_path)
            },
            "processing_summary": results_data["processing_summary"]
        }
        
        with open(metadata_path, 'w') as f:
            json.dump(job_metadata, f, indent=2, default=str)
        
        logger.info(f"✅ Results saved to shared storage: {job_dir}")
        return str(json_path)
    
    def _save_csv_format(self, features: List[Dict[str, Any]], 
                        added_scores: List[str], csv_path: Path):
        """Save features in CSV format"""
        import csv
        
        # Determine columns
        base_columns = ["feature_id", "coherence_score"]
        score_columns = added_scores
        optional_columns = ["quality_level", "pattern_keywords", "explanation", "confidence"]
        
        # Check what columns are available
        all_columns = base_columns + score_columns
        for col in optional_columns:
            if any(col in feature for feature in features):
                all_columns.append(col)
        
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=all_columns)
            writer.writeheader()
            
            for feature in features:
                row = {}
                for col in all_columns:
                    if col == "pattern_keywords" and col in feature:
                        # Join keywords with semicolons
                        row[col] = "; ".join(str(k) for k in feature[col])
                    else:
                        row[col] = feature.get(col, "")
                writer.writerow(row)
    
    def _save_summary_report(self, job_id: str, source_type: str, source_job_id: str,
                           features: List[Dict[str, Any]], added_scores: List[str], 
                           summary_path: Path):
        """Save human-readable summary report"""
        with open(summary_path, 'w') as f:
            f.write(f"miStudioScore Summary Report\n")
            f.write(f"============================\n\n")
            f.write(f"Job ID: {job_id}\n")
            f.write(f"Source Type: {source_type}\n")
            f.write(f"Source Job ID: {source_job_id}\n")
            f.write(f"Generated: {datetime.now().isoformat()}\n\n")
            f.write(f"Processing Summary:\n")
            f.write(f"Total Features: {len(features)}\n")
            f.write(f"Scores Added: {len(added_scores)}\n")
            f.write(f"Score Types: {', '.join(added_scores)}\n\n")
            
            # Score statistics
            f.write(f"Score Statistics:\n")
            f.write(f"================\n")
            for score_name in added_scores:
                scores = [f.get(score_name, 0) for f in features if score_name in f]
                if scores:
                    f.write(f"{score_name}:\n")
                    f.write(f"  Count: {len(scores)}\n")
                    f.write(f"  Average: {sum(scores)/len(scores):.3f}\n")
                    f.write(f"  Max: {max(scores):.3f}\n")
                    f.write(f"  Min: {min(scores):.3f}\n\n")
            
            # Top features
            f.write(f"Top 10 Features by Average Score:\n")
            f.write(f"=================================\n")
            
            # Calculate average scores for ranking
            scored_features = []
            for feature in features:
                avg_score = sum(feature.get(score, 0) for score in added_scores) / len(added_scores) if added_scores else 0
                scored_features.append((feature, avg_score))
            
            # Sort by average score
            scored_features.sort(key=lambda x: x[1], reverse=True)
            
            for i, (feature, avg_score) in enumerate(scored_features[:10]):
                f.write(f"{i+1:2d}. Feature {feature.get('feature_id', 'unknown'):>3} ")
                f.write(f"(avg: {avg_score:.3f}) ")
                
                # Show individual scores
                score_details = []
                for score_name in added_scores:
                    if score_name in feature:
                        score_details.append(f"{score_name}={feature[score_name]:.3f}")
                
                if score_details:
                    f.write(f"[{', '.join(score_details)}]")
                f.write("\n")


class PatternScorer:
    """Built-in pattern-based scorer"""
    
    def score(self, features: List[Dict[str, Any]], **params) -> List[Dict[str, Any]]:
        """Apply pattern-based scoring"""
        score_name = params.get("name", "pattern_score")
        
        for feature in features:
            # Calculate pattern score based on coherence and keyword diversity
            coherence = feature.get("coherence_score", 0)
            keywords = feature.get("pattern_keywords", [])
            keyword_diversity = min(len(keywords) / 10.0, 1.0)  # Normalize to max 10 keywords
            
            # Combine coherence and keyword diversity
            pattern_score = (coherence * 0.7) + (keyword_diversity * 0.3)
            feature[score_name] = min(1.0, pattern_score)
        
        return features


# Legacy compatibility function
def execute_scoring(features_path: str, config_path: str, output_dir: str) -> Tuple[Optional[str], List[str]]:
    """Legacy function for backward compatibility"""
    logger.warning("Using legacy execute_scoring function")
    logger.warning("Consider migrating to IntegratedOrchestrator for better shared storage integration")
    
    try:
        # Try to determine if this is a shared storage path
        path_obj = Path(features_path)
        
        if "/data/results/find/" in str(path_obj) or "/data/results/explain/" in str(path_obj):
            logger.info("Detected shared storage path, attempting to use integrated orchestrator")
            
            # Extract job ID and type from path
            if "/find/" in str(path_obj):
                source_type = "find"
                source_job_id = path_obj.parent.name
            elif "/explain/" in str(path_obj):
                source_type = "explain"
                source_job_id = path_obj.parent.name
            else:
                raise ValueError("Could not determine source type from path")
            
            # Load config
            import yaml
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            # Use integrated orchestrator
            orchestrator = IntegratedOrchestrator()
            return orchestrator.execute_scoring_pipeline(source_type, source_job_id, config)
        
        else:
            # Fall back to original legacy behavior
            from utils.file_handler import load_json_file, load_yaml_config, save_json_file
            
            features = load_json_file(features_path)
            config = load_yaml_config(config_path)
            
            added_scores = []
            scoring_jobs = config.get("scoring_jobs", [])
            
            # Use built-in scorers for legacy mode
            available_scorers = {"pattern_scorer": PatternScorer}
            
            for job in scoring_jobs:
                scorer_name = job.get("scorer")
                job_params = job.get("params", {})
                job_name = job.get("name")
                
                if not job_name:
                    continue
                
                if scorer_name in available_scorers:
                    scorer_class = available_scorers[scorer_name]
                    scorer_instance = scorer_class()
                    all_params = {"name": job_name, **job_params}
                    features = scorer_instance.score(features, **all_params)
                    added_scores.append(job_name)
            
            # Save using legacy format
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            output_filename = f"scores_{timestamp}.json"
            output_path = f"{output_dir}/{output_filename}"
            
            save_json_file(features, output_path)
            return output_path, added_scores
            
    except Exception as e:
        logger.error(f"Legacy scoring execution failed: {e}")
        raise