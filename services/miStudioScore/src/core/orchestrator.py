# src/core/orchestrator.py - Updated with Pattern Scorer Removed
"""
Enhanced orchestrator for miStudioScore with only working scorers.
Pattern scorer has been completely removed as it was broken and provided no analytical value.
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
        
        # Initialize available scorers (pattern_scorer removed)
        self.scorers = {}
        self._load_scorers()
        
        logger.info(f"IntegratedOrchestrator initialized with data path: {self.data_path}")
        logger.info(f"Available scorers: {list(self.scorers.keys())}")
    
    def _load_scorers(self):
        """Load available scoring modules (working scorers only)"""
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
        
        # Pattern scorer has been completely removed - it was broken and provided no value
        logger.info(f"Total working scorers loaded: {len(self.scorers)}")
    
    def discover_source_jobs(self, source_type: str) -> List[Dict[str, Any]]:
        """Discover available source jobs for scoring"""
        if source_type == "find":
            source_dir = self.find_results_dir
        elif source_type == "explain":
            source_dir = self.explain_results_dir
        else:
            raise ValueError(f"Unknown source type: {source_type}")
        
        jobs = []
        for job_dir in source_dir.iterdir():
            if job_dir.is_dir():
                # Look for key files to determine job validity
                if source_type == "find":
                    key_file = job_dir / "features.json"
                elif source_type == "explain":
                    key_file = job_dir / "explanation_results.json"
                
                if key_file.exists():
                    # Get job metadata
                    info_file = job_dir / "job_info.json"
                    if info_file.exists():
                        try:
                            with open(info_file, 'r') as f:
                                job_info = json.load(f)
                        except:
                            job_info = {}
                    else:
                        job_info = {}
                    
                    jobs.append({
                        "job_id": job_dir.name,
                        "source_type": source_type,
                        "job_directory": str(job_dir),
                        "key_file": str(key_file),
                        "metadata": job_info
                    })
        
        return sorted(jobs, key=lambda x: x["job_id"], reverse=True)
    
    def execute_scoring_pipeline(self, source_type: str, source_job_id: str, scoring_config: Dict[str, Any]) -> Tuple[Optional[str], List[str]]:
        """Execute scoring pipeline with integrated storage"""
        logger.info(f"Starting scoring pipeline for {source_type} job {source_job_id}")
        
        # Validate source job exists
        source_jobs = self.discover_source_jobs(source_type)
        source_job = next((j for j in source_jobs if j["job_id"] == source_job_id), None)
        
        if not source_job:
            raise ValueError(f"Source job {source_job_id} not found in {source_type} results")
        
        # Load source data
        source_data_path = source_job["key_file"]
        logger.info(f"Loading source data from: {source_data_path}")
        
        try:
            with open(source_data_path, 'r') as f:
                features = json.load(f)
            
            # Handle different source formats
            if source_type == "explain" and isinstance(features, dict):
                # Extract features from explanation results
                if "explanations" in features:
                    features = features["explanations"]
                elif "features" in features:
                    features = features["features"]
                else:
                    raise ValueError("Could not extract features from explanation results")
            
            if not isinstance(features, list):
                raise ValueError(f"Expected features to be a list, got {type(features)}")
            
            logger.info(f"Loaded {len(features)} features from source")
            
        except Exception as e:
            logger.error(f"Failed to load source data: {e}")
            raise
        
        # Execute scoring jobs
        added_scores = []
        scoring_jobs = scoring_config.get("scoring_jobs", [])
        
        for job in scoring_jobs:
            scorer_name = job.get("scorer")
            job_params = job.get("params", {})
            job_name = job.get("name")
            
            if not job_name:
                logger.warning(f"Scoring job with scorer '{scorer_name}' is missing a 'name'. Skipping.")
                continue
            
            if scorer_name not in self.scorers:
                logger.error(f"Scorer '{scorer_name}' not found. Available scorers: {list(self.scorers.keys())}")
                continue
            
            try:
                logger.info(f"Executing scoring job: {job_name} using {scorer_name}")
                scorer_class = self.scorers[scorer_name]
                scorer_instance = scorer_class()
                
                # Prepare parameters
                all_params = {"name": job_name, **job_params}
                
                # Execute scoring
                features = scorer_instance.score(features, **all_params)
                added_scores.append(job_name)
                
                logger.info(f"✅ Completed scoring job: {job_name}")
                
            except Exception as e:
                logger.error(f"Scoring job '{job_name}' failed: {e}", exc_info=True)
                continue
        
        if not added_scores:
            logger.warning("No scoring jobs completed successfully")
            return None, []
        
        # Generate output path
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        job_id = f"score_{timestamp}_{hash(source_job_id) % 100000000:08x}"
        
        output_dir = self.score_results_dir / job_id
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save results in multiple formats
        results_json_path = output_dir / "scoring_results.json"
        scored_features_csv_path = output_dir / "scored_features.csv"
        job_info_path = output_dir / "job_info.json"
        summary_path = output_dir / "summary_report.txt"
        
        # Save JSON results
        with open(results_json_path, 'w') as f:
            json.dump(features, f, indent=2)
        
        # Save CSV results
        self._save_csv_results(features, scored_features_csv_path, added_scores)
        
        # Save job info
        job_info = {
            "job_id": job_id,
            "source_type": source_type,
            "source_job_id": source_job_id,
            "timestamp": datetime.now().isoformat(),
            "total_features": len(features),
            "scores_added": added_scores,
            "scoring_config": scoring_config
        }
        
        with open(job_info_path, 'w') as f:
            json.dump(job_info, f, indent=2)
        
        # Generate summary report
        self._generate_summary_report(features, added_scores, summary_path, job_info)
        
        logger.info(f"✅ Scoring pipeline completed. Results saved to: {output_dir}")
        
        return str(results_json_path), added_scores
    
    def _save_csv_results(self, features: List[Dict[str, Any]], csv_path: Path, added_scores: List[str]):
        """Save results in CSV format"""
        try:
            import csv
            
            with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
                # Determine all possible fields
                all_fields = set()
                for feature in features:
                    all_fields.update(feature.keys())
                
                # Order fields: feature_id, coherence_score, then added scores, then rest
                ordered_fields = []
                if "feature_id" in all_fields:
                    ordered_fields.append("feature_id")
                if "coherence_score" in all_fields:
                    ordered_fields.append("coherence_score")
                
                # Add scoring results
                for score in added_scores:
                    if score in all_fields:
                        ordered_fields.append(score)
                
                # Add remaining fields
                remaining_fields = sorted(all_fields - set(ordered_fields))
                ordered_fields.extend(remaining_fields)
                
                writer = csv.DictWriter(csvfile, fieldnames=ordered_fields)
                writer.writeheader()
                
                for feature in features:
                    row = {}
                    for field in ordered_fields:
                        value = feature.get(field, "")
                        # Handle complex objects by converting to string
                        if isinstance(value, (dict, list)):
                            value = str(value)
                        row[field] = value
                    writer.writerow(row)
                    
            logger.info(f"CSV results saved to: {csv_path}")
            
        except Exception as e:
            logger.error(f"Failed to save CSV results: {e}")
    
    def _generate_summary_report(self, features: List[Dict[str, Any]], added_scores: List[str], 
                                summary_path: Path, job_info: Dict[str, Any]):
        """Generate human-readable summary report"""
        try:
            with open(summary_path, 'w') as f:
                f.write("miStudioScore Summary Report\n")
                f.write("============================\n")
                f.write(f"Job ID: {job_info['job_id']}\n")
                f.write(f"Source Type: {job_info['source_type']}\n")
                f.write(f"Source Job ID: {job_info['source_job_id']}\n")
                f.write(f"Generated: {job_info['timestamp']}\n")
                f.write(f"Processing Time: {job_info.get('processing_time', 'Unknown')}\n\n")
                
                f.write(f"Results Summary:\n")
                f.write(f"Total Features: {len(features)}\n")
                f.write(f"Scores Added: {len(added_scores)}\n")
                f.write(f"Score Types: {', '.join(added_scores)}\n\n")
                
                # Score statistics
                f.write(f"Score Statistics:\n")
                for score_name in added_scores:
                    scores = [f.get(score_name, 0) for f in features if score_name in f]
                    if scores:
                        f.write(f"{score_name}:\n")
                        f.write(f"  Average: {sum(scores)/len(scores):.3f}\n")
                        f.write(f"  Max: {max(scores):.3f}\n")
                        f.write(f"  Min: {min(scores):.3f}\n")
                
                # Top features
                f.write(f"\nTop 5 Features by Average Score:\n")
                f.write(f"--------------------------------\n")
                
                # Calculate average scores for ranking
                scored_features = []
                for feature in features:
                    avg_score = sum(feature.get(score, 0) for score in added_scores) / len(added_scores) if added_scores else 0
                    scored_features.append((feature, avg_score))
                
                # Sort by average score
                scored_features.sort(key=lambda x: x[1], reverse=True)
                
                for i, (feature, avg_score) in enumerate(scored_features[:5]):
                    feature_id = feature.get('feature_id', feature.get('feature_index', 'unknown'))
                    f.write(f"{i+1}. Feature {feature_id} (avg: {avg_score:.3f})\n")
                    
            logger.info(f"Summary report saved to: {summary_path}")
            
        except Exception as e:
            logger.error(f"Failed to generate summary report: {e}")


# Legacy compatibility function (pattern_scorer removed)
def execute_scoring(features_path: str, config_path: str, output_dir: str) -> Tuple[Optional[str], List[str]]:
    """Legacy function for backward compatibility (pattern_scorer removed)"""
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
            # Fall back to original legacy behavior (with working scorers only)
            from utils.file_handler import load_json_file, load_yaml_config, save_json_file
            
            features = load_json_file(features_path)
            config = load_yaml_config(config_path)
            
            added_scores = []
            scoring_jobs = config.get("scoring_jobs", [])
            
            # Use only working scorers for legacy mode (pattern_scorer removed)
            available_scorers = {}
            
            # Try to import working scorers
            try:
                from core.scoring.relevance_scorer import RelevanceScorer
                available_scorers["relevance_scorer"] = RelevanceScorer
            except ImportError:
                logger.warning("RelevanceScorer not available in legacy mode")
            
            try:
                from core.scoring.ablation_scorer import AblationScorer
                available_scorers["ablation_scorer"] = AblationScorer
            except ImportError:
                logger.warning("AblationScorer not available in legacy mode")
            
            if not available_scorers:
                logger.error("No working scorers available")
                return None, []
            
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
                else:
                    logger.warning(f"Scorer '{scorer_name}' not available. Available: {list(available_scorers.keys())}")
            
            # Save using legacy format
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            output_filename = f"scores_{timestamp}.json"
            output_path = f"{output_dir}/{output_filename}"
            
            save_json_file(features, output_path)
            return output_path, added_scores
            
    except Exception as e:
        logger.error(f"Legacy scoring execution failed: {e}")
        raise