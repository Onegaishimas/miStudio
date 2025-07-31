# src/utils/file_handler.py - Updated with Shared Storage Integration
"""
Enhanced file handling utilities with shared storage integration.
"""
import json
import yaml
import logging
from typing import Dict, Any, List, Optional
from pathlib import Path

logger = logging.getLogger(__name__)


class IntegratedFileHandler:
    """Enhanced file handler with shared storage integration"""
    
    def __init__(self, data_path: str = "/data"):
        """Initialize with shared data path"""
        self.data_path = Path(data_path)
        self.find_results_dir = self.data_path / "results" / "find"
        self.explain_results_dir = self.data_path / "results" / "explain"
        self.score_results_dir = self.data_path / "results" / "score"
        
        # Ensure directories exist
        for directory in [self.find_results_dir, self.explain_results_dir, self.score_results_dir]:
            directory.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"IntegratedFileHandler initialized with data path: {self.data_path}")
    
    def discover_find_jobs(self) -> List[Dict[str, Any]]:
        """Discover available Find jobs"""
        jobs = []
        
        # Check main Find results directory
        if self.find_results_dir.exists():
            for job_dir in self.find_results_dir.iterdir():
                if job_dir.is_dir():
                    results_file = job_dir / "analysis_results.json"
                    if results_file.exists():
                        try:
                            with open(results_file, 'r') as f:
                                data = json.load(f)
                            
                            jobs.append({
                                "job_id": job_dir.name,
                                "path": str(results_file),
                                "feature_count": len(data.get('results', [])),
                                "location": "main"
                            })
                        except Exception as e:
                            logger.warning(f"Could not read Find job {job_dir.name}: {e}")
        
        # Check enhanced persistence directories
        results_base = self.data_path / "results"
        if results_base.exists():
            for job_dir in results_base.glob("find_*"):
                if job_dir.is_dir() and job_dir.name not in [j["job_id"] for j in jobs]:
                    results_file = job_dir / f"{job_dir.name}_complete_results.json"
                    if results_file.exists():
                        try:
                            with open(results_file, 'r') as f:
                                data = json.load(f)
                            
                            jobs.append({
                                "job_id": job_dir.name,
                                "path": str(results_file),
                                "feature_count": len(data.get('results', [])),
                                "location": "enhanced"
                            })
                        except Exception as e:
                            logger.warning(f"Could not read enhanced Find job {job_dir.name}: {e}")
        
        return jobs
    
    def discover_explain_jobs(self) -> List[Dict[str, Any]]:
        """Discover available Explain jobs"""
        jobs = []
        
        if self.explain_results_dir.exists():
            for job_dir in self.explain_results_dir.iterdir():
                if job_dir.is_dir():
                    results_file = job_dir / "explanation_results.json"
                    if results_file.exists():
                        try:
                            with open(results_file, 'r') as f:
                                data = json.load(f)
                            
                            jobs.append({
                                "job_id": job_dir.name,
                                "path": str(results_file),
                                "explanation_count": len(data.get('explanations', [])),
                                "location": "main"
                            })
                        except Exception as e:
                            logger.warning(f"Could not read Explain job {job_dir.name}: {e}")
        
        return jobs
    
    def load_find_results(self, find_job_id: str) -> Dict[str, Any]:
        """Load Find results from shared storage"""
        # Try main results directory first
        main_path = self.find_results_dir / find_job_id / "analysis_results.json"
        
        if main_path.exists():
            with open(main_path, 'r') as f:
                data = json.load(f)
            logger.info(f"Loaded Find results from: {main_path}")
            return data
        
        # Try enhanced persistence directory
        enhanced_path = self.data_path / "results" / find_job_id / f"{find_job_id}_complete_results.json"
        
        if enhanced_path.exists():
            with open(enhanced_path, 'r') as f:
                data = json.load(f)
            logger.info(f"Loaded Find results from: {enhanced_path}")
            return data
        
        # List available for error
        available_jobs = self.discover_find_jobs()
        available_ids = [job["job_id"] for job in available_jobs]
        
        raise FileNotFoundError(f"Find job {find_job_id} not found. Available: {available_ids}")
    
    def load_explain_results(self, explain_job_id: str) -> Dict[str, Any]:
        """Load Explain results from shared storage"""
        explain_path = self.explain_results_dir / explain_job_id / "explanation_results.json"
        
        if explain_path.exists():
            with open(explain_path, 'r') as f:
                data = json.load(f)
            logger.info(f"Loaded Explain results from: {explain_path}")
            return data
        
        # List available for error
        available_jobs = self.discover_explain_jobs()
        available_ids = [job["job_id"] for job in available_jobs]
        
        raise FileNotFoundError(f"Explain job {explain_job_id} not found. Available: {available_ids}")
    
    def save_score_results(self, job_id: str, data: Dict[str, Any]) -> str:
        """Save scoring results to shared storage"""
        job_dir = self.score_results_dir / job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        
        output_path = job_dir / "scoring_results.json"
        
        try:
            with open(output_path, 'w') as f:
                json.dump(data, f, indent=2, default=str)
            logger.info(f"Saved scoring results to: {output_path}")
            return str(output_path)
        except IOError as e:
            logger.error(f"Could not write to file: {output_path} - {e}")
            raise
    
    def save_score_csv(self, job_id: str, features: List[Dict[str, Any]], 
                      score_columns: List[str]) -> str:
        """Save scoring results in CSV format"""
        import csv
        
        job_dir = self.score_results_dir / job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        
        csv_path = job_dir / "scored_features.csv"
        
        # Determine columns
        base_columns = ["feature_id", "coherence_score"]
        all_columns = base_columns + score_columns
        
        # Add optional columns if present
        optional_columns = ["quality_level", "pattern_keywords", "explanation"]
        for col in optional_columns:
            if any(col in feature for feature in features):
                all_columns.append(col)
        
        try:
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
            
            logger.info(f"Saved CSV results to: {csv_path}")
            return str(csv_path)
        except IOError as e:
            logger.error(f"Could not write CSV file: {csv_path} - {e}")
            raise


# Legacy compatibility functions
def load_json_file(file_path: str) -> List[Dict[str, Any]]:
    """Legacy function - now redirects to integrated handler"""
    logger.warning(f"Using legacy load_json_file for: {file_path}")
    logger.warning("Consider migrating to IntegratedFileHandler for better integration")
    
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        logger.info(f"Successfully loaded JSON from {file_path}")
        return data
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        raise
    except json.JSONDecodeError:
        logger.error(f"Error decoding JSON from {file_path}")
        raise


def save_json_file(data: List[Dict[str, Any]], file_path: str):
    """Legacy function - now redirects to integrated handler"""
    logger.warning(f"Using legacy save_json_file for: {file_path}")
    logger.warning("Consider migrating to IntegratedFileHandler for better integration")
    
    try:
        # Ensure directory exists
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        logger.info(f"Successfully saved JSON to {file_path}")
    except IOError:
        logger.error(f"Could not write to file: {file_path}")
        raise


def load_yaml_config(file_path: str) -> Dict[str, Any]:
    """Load YAML configuration file"""
    try:
        with open(file_path, 'r') as f:
            config = yaml.safe_load(f)
        logger.info(f"Successfully loaded YAML config from {file_path}")
        return config
    except FileNotFoundError:
        logger.error(f"Config file not found: {file_path}")
        raise
    except yaml.YAMLError:
        logger.error(f"Error parsing YAML from {file_path}")
        raise


# Modern integrated functions
def create_integrated_handler(data_path: str = "/data") -> IntegratedFileHandler:
    """Create an integrated file handler"""
    return IntegratedFileHandler(data_path)


def discover_available_sources(data_path: str = "/data") -> Dict[str, List[Dict[str, Any]]]:
    """Discover all available source jobs"""
    handler = IntegratedFileHandler(data_path)
    
    return {
        "find_jobs": handler.discover_find_jobs(),
        "explain_jobs": handler.discover_explain_jobs()
    }