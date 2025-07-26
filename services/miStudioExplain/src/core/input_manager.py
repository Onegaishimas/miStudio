"""
Input Manager for miStudioExplain Service

Manages loading and validation of miStudioFind outputs with comprehensive
error handling, input validation, and structured data extraction.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
from datetime import datetime
import re

logger = logging.getLogger(__name__)


class ValidationError(Exception):
    """Raised when input validation fails."""
    pass


class FileNotFoundError(Exception):
    """Raised when required input files are not found."""
    pass


@dataclass
class FeatureData:
    """Structure for individual feature data from miStudioFind."""
    feature_id: int
    coherence_score: float
    quality_level: str
    pattern_category: str
    pattern_keywords: List[str]
    top_activations: List[Dict[str, Any]]
    activation_statistics: Dict[str, float]
    
    def __post_init__(self):
        """Validate feature data after initialization."""
        if not isinstance(self.feature_id, int) or self.feature_id < 0:
            raise ValidationError(f"Invalid feature_id: {self.feature_id}")
        
        if not isinstance(self.coherence_score, (int, float)) or not (0.0 <= self.coherence_score <= 1.0):
            raise ValidationError(f"Invalid coherence_score: {self.coherence_score}")
        
        if not isinstance(self.pattern_keywords, list):
            raise ValidationError("pattern_keywords must be a list")
        
        if not isinstance(self.top_activations, list):
            raise ValidationError("top_activations must be a list")


@dataclass
class JobMetadata:
    """Metadata for miStudioFind job."""
    job_id: str
    source_training_job: str
    model_name: str
    total_features: int
    processing_time: str
    service_version: str = "1.0.0"
    completion_timestamp: Optional[str] = None
    
    def __post_init__(self):
        """Validate job metadata after initialization."""
        if not self.job_id or not isinstance(self.job_id, str):
            raise ValidationError("job_id must be a non-empty string")
        
        if not self.source_training_job or not isinstance(self.source_training_job, str):
            raise ValidationError("source_training_job must be a non-empty string")
        
        if not isinstance(self.total_features, int) or self.total_features <= 0:
            raise ValidationError("total_features must be a positive integer")


class InputManager:
    """
    Manages loading and validation of miStudioFind outputs.
    
    Provides comprehensive input validation, error handling, and structured
    data extraction from miStudioFind JSON results.
    """
    
    # Required keys in miStudioFind JSON
    REQUIRED_ROOT_KEYS = ['job_metadata', 'features', 'summary_insights']
    REQUIRED_METADATA_KEYS = ['job_id', 'source_training_job', 'total_features_processed']
    REQUIRED_FEATURE_KEYS = ['feature_id', 'coherence_score', 'pattern_category']
    
    def __init__(self, data_path: str = "./data/input"):
        """
        Initialize InputManager.
        
        Args:
            data_path: Path to directory containing input files
            
        Raises:
            ValidationError: If data_path is invalid
        """
        if not data_path or not isinstance(data_path, str):
            raise ValidationError("data_path must be a non-empty string")
        
        self.data_path = Path(data_path)
        self.data_path.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        self.logger.info(f"InputManager initialized with data path: {self.data_path}")
    
    def load_mistudio_find_results(self, job_id: str) -> Dict[str, Any]:
        """
        Load and validate miStudioFind JSON results.
        
        Args:
            job_id: Unique identifier for the miStudioFind job
            
        Returns:
            Dictionary containing validated miStudioFind results
            
        Raises:
            ValidationError: If job_id is invalid or data structure is malformed
            FileNotFoundError: If results file cannot be found
            json.JSONDecodeError: If file contains invalid JSON
            
        Example:
            >>> manager = InputManager("./data")
            >>> results = manager.load_mistudio_find_results("find_20250726_123456")
            >>> print(f"Loaded {len(results['features'])} features")
        """
        # Validate input parameters
        if not job_id or not isinstance(job_id, str):
            raise ValidationError("job_id must be a non-empty string")
        
        job_id = job_id.strip()
        if not job_id:
            raise ValidationError("job_id cannot be empty or whitespace")
        
        # Construct file paths (try multiple naming conventions)
        possible_filenames = [
            f"{job_id}_results.json",
            f"{job_id}.json",
            f"mistudio_find_{job_id}.json",
            f"feature_explanations_{job_id}.json"
        ]
        
        file_path = None
        for filename in possible_filenames:
            candidate_path = self.data_path / filename
            if candidate_path.exists():
                file_path = candidate_path
                break
        
        if file_path is None:
            available_files = list(self.data_path.glob("*.json"))
            raise FileNotFoundError(
                f"Results file not found for job_id '{job_id}'. "
                f"Searched for: {possible_filenames}. "
                f"Available files: {[f.name for f in available_files]}"
            )
        
        # Load and parse JSON
        try:
            self.logger.info(f"Loading results from: {file_path}")
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            self.logger.info(f"Successfully loaded JSON data ({len(str(data))} characters)")
            
        except json.JSONDecodeError as e:
            raise json.JSONDecodeError(
                f"Invalid JSON format in {file_path}: {e.msg}",
                e.doc, e.pos
            )
        except Exception as e:
            self.logger.error(f"Failed to load results for {job_id}: {e}")
            raise
        
        # Validate data structure
        self._validate_data_structure(data, job_id)
        
        self.logger.info(f"Successfully loaded and validated results for job {job_id}")
        return data
    
    def validate_input_data(self, data: Dict[str, Any]) -> bool:
        """
        Validate input data structure and completeness.
        
        Args:
            data: Dictionary containing miStudioFind results
            
        Returns:
            True if data is valid and complete
            
        Raises:
            ValidationError: If data structure is invalid
            TypeError: If data is not a dictionary
            
        Example:
            >>> manager = InputManager()
            >>> is_valid = manager.validate_input_data(results)
            >>> if is_valid:
            ...     print("Data is ready for processing")
        """
        if not isinstance(data, dict):
            raise TypeError("Input data must be a dictionary")
        
        if not data:
            raise ValidationError("Input data cannot be empty")
        
        # Validate root structure
        missing_keys = [key for key in self.REQUIRED_ROOT_KEYS if key not in data]
        if missing_keys:
            raise ValidationError(f"Missing required root keys: {missing_keys}")
        
        # Validate metadata
        metadata = data.get('job_metadata', {})
        if not isinstance(metadata, dict):
            raise ValidationError("job_metadata must be a dictionary")
        
        missing_metadata = [key for key in self.REQUIRED_METADATA_KEYS if key not in metadata]
        if missing_metadata:
            raise ValidationError(f"Missing required metadata keys: {missing_metadata}")
        
        # Validate features array
        features = data.get('features', [])
        if not isinstance(features, list):
            raise ValidationError("features must be a list")
        
        if not features:
            raise ValidationError("features list cannot be empty")
        
        # Validate feature structure (sample first few features)
        sample_size = min(5, len(features))
        for i, feature in enumerate(features[:sample_size]):
            if not isinstance(feature, dict):
                raise ValidationError(f"Feature {i} must be a dictionary")
            
            missing_feature_keys = [key for key in self.REQUIRED_FEATURE_KEYS if key not in feature]
            if missing_feature_keys:
                raise ValidationError(f"Feature {i} missing required keys: {missing_feature_keys}")
        
        # Validate data consistency
        declared_feature_count = metadata.get('total_features_processed', 0)
        actual_feature_count = len(features)
        
        if declared_feature_count != actual_feature_count:
            self.logger.warning(
                f"Feature count mismatch: metadata says {declared_feature_count}, "
                f"but found {actual_feature_count} features"
            )
        
        self.logger.info(f"Input data validation passed: {actual_feature_count} features")
        return True
    
    def extract_features(self, data: Dict[str, Any]) -> List[FeatureData]:
        """
        Extract and structure feature data from miStudioFind results.
        
        Args:
            data: Validated miStudioFind results dictionary
            
        Returns:
            List of structured FeatureData objects
            
        Raises:
            ValidationError: If feature data is malformed
            
        Example:
            >>> features = manager.extract_features(results)
            >>> high_quality = [f for f in features if f.coherence_score > 0.7]
            >>> print(f"Found {len(high_quality)} high-quality features")
        """
        if not self.validate_input_data(data):
            raise ValidationError("Data validation failed before feature extraction")
        
        features_data = data['features']
        extracted_features = []
        
        self.logger.info(f"Extracting {len(features_data)} features...")
        
        for i, feature_dict in enumerate(features_data):
            try:
                # Extract required fields with defaults
                feature_data = FeatureData(
                    feature_id=int(feature_dict['feature_id']),
                    coherence_score=float(feature_dict.get('coherence_score', 0.0)),
                    quality_level=str(feature_dict.get('quality_level', 'unknown')),
                    pattern_category=str(feature_dict.get('pattern_category', 'general')),
                    pattern_keywords=list(feature_dict.get('pattern_keywords', [])),
                    top_activations=list(feature_dict.get('top_activations', [])),
                    activation_statistics=dict(feature_dict.get('activation_statistics', {}))
                )
                
                extracted_features.append(feature_data)
                
            except (ValueError, TypeError, KeyError) as e:
                self.logger.warning(f"Skipping malformed feature {i}: {e}")
                continue
            except ValidationError as e:
                self.logger.warning(f"Skipping invalid feature {i}: {e}")
                continue
        
        if not extracted_features:
            raise ValidationError("No valid features could be extracted")
        
        success_rate = len(extracted_features) / len(features_data) * 100
        self.logger.info(
            f"Successfully extracted {len(extracted_features)}/{len(features_data)} "
            f"features ({success_rate:.1f}% success rate)"
        )
        
        return extracted_features
    
    def get_job_metadata(self, data: Dict[str, Any]) -> JobMetadata:
        """
        Extract job metadata from miStudioFind results.
        
        Args:
            data: Validated miStudioFind results dictionary
            
        Returns:
            Structured JobMetadata object
            
        Raises:
            ValidationError: If metadata is missing or invalid
            
        Example:
            >>> metadata = manager.get_job_metadata(results)
            >>> print(f"Job {metadata.job_id} processed {metadata.total_features} features")
        """
        if not isinstance(data, dict) or 'job_metadata' not in data:
            raise ValidationError("Missing job_metadata in input data")
        
        metadata_dict = data['job_metadata']
        
        try:
            # Extract and validate metadata
            job_metadata = JobMetadata(
                job_id=str(metadata_dict['job_id']),
                source_training_job=str(metadata_dict.get('source_training_job', 'unknown')),
                model_name=str(metadata_dict.get('model_name', 'unknown')),
                total_features=int(metadata_dict.get('total_features_processed', 0)),
                processing_time=str(metadata_dict.get('processing_time', 'unknown')),
                service_version=str(metadata_dict.get('service_version', '1.0.0')),
                completion_timestamp=metadata_dict.get('completion_timestamp')
            )
            
            self.logger.info(f"Extracted metadata for job: {job_metadata.job_id}")
            return job_metadata
            
        except (KeyError, ValueError, TypeError) as e:
            raise ValidationError(f"Failed to extract job metadata: {e}")
    
    def _validate_data_structure(self, data: Dict[str, Any], job_id: str) -> None:
        """
        Internal method to validate the overall data structure.
        
        Args:
            data: Raw data loaded from JSON
            job_id: Job ID for error reporting
            
        Raises:
            ValidationError: If structure is invalid
        """
        try:
            # Basic structure validation
            self.validate_input_data(data)
            
            # Additional consistency checks
            metadata = data['job_metadata']
            features = data['features']
            
            # Check job_id consistency
            file_job_id = metadata.get('job_id', '')
            if file_job_id and file_job_id != job_id:
                self.logger.warning(
                    f"Job ID mismatch: requested '{job_id}', "
                    f"file contains '{file_job_id}'"
                )
            
            # Check for reasonable feature counts
            feature_count = len(features)
            if feature_count > 10000:
                self.logger.warning(f"Very large feature count: {feature_count}")
            elif feature_count < 10:
                self.logger.warning(f"Very small feature count: {feature_count}")
            
            # Validate feature IDs are unique
            feature_ids = [f.get('feature_id') for f in features if 'feature_id' in f]
            if len(set(feature_ids)) != len(feature_ids):
                raise ValidationError("Duplicate feature IDs found in data")
            
        except ValidationError:
            raise
        except Exception as e:
            raise ValidationError(f"Data structure validation failed: {e}")
    
    def list_available_jobs(self) -> List[str]:
        """
        List all available job IDs in the input directory.
        
        Returns:
            List of job IDs that have result files available
            
        Example:
            >>> manager = InputManager()
            >>> jobs = manager.list_available_jobs()
            >>> print(f"Available jobs: {jobs}")
        """
        json_files = list(self.data_path.glob("*.json"))
        job_ids = []
        
        for file_path in json_files:
            # Extract job ID from various filename patterns
            filename = file_path.stem
            
            # Pattern matching for different naming conventions
            patterns = [
                r'(.+)_results$',           # job_id_results.json
                r'mistudio_find_(.+)$',     # mistudio_find_job_id.json
                r'feature_explanations_(.+)$',  # feature_explanations_job_id.json
                r'^(.+)$'                   # job_id.json
            ]
            
            for pattern in patterns:
                match = re.match(pattern, filename)
                if match:
                    job_ids.append(match.group(1))
                    break
        
        # Remove duplicates and sort
        unique_job_ids = sorted(set(job_ids))
        self.logger.info(f"Found {len(unique_job_ids)} available job(s): {unique_job_ids}")
        
        return unique_job_ids
    
    def get_job_summary(self, job_id: str) -> Dict[str, Any]:
        """
        Get a quick summary of a job without loading full data.
        
        Args:
            job_id: Job identifier
            
        Returns:
            Dictionary with job summary information
            
        Example:
            >>> summary = manager.get_job_summary("find_20250726_123456")
            >>> print(f"Features: {summary['feature_count']}")
        """
        try:
            data = self.load_mistudio_find_results(job_id)
            metadata = self.get_job_metadata(data)
            features = data.get('features', [])
            
            # Calculate basic statistics
            coherence_scores = [
                f.get('coherence_score', 0.0) for f in features 
                if isinstance(f.get('coherence_score'), (int, float))
            ]
            
            summary = {
                'job_id': metadata.job_id,
                'source_training_job': metadata.source_training_job,
                'model_name': metadata.model_name,
                'feature_count': len(features),
                'processing_time': metadata.processing_time,
                'avg_coherence_score': sum(coherence_scores) / len(coherence_scores) if coherence_scores else 0.0,
                'max_coherence_score': max(coherence_scores) if coherence_scores else 0.0,
                'high_quality_features': len([s for s in coherence_scores if s >= 0.6]),
                'completion_timestamp': metadata.completion_timestamp
            }
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Failed to get summary for job {job_id}: {e}")
            return {'job_id': job_id, 'error': str(e)}