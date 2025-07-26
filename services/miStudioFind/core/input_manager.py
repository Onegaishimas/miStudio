# core/input_manager.py
"""
Input file management for miStudioFind service - Production version.
"""

import json
import logging
import torch
import os
from pathlib import Path
from typing import Dict, Any, Tuple

logger = logging.getLogger(__name__)


class InputValidationError(Exception):
    """Exception raised when input validation fails."""
    pass


class InputManager:
    """Manages loading and validation of input files from miStudioTrain."""
    
    def __init__(self, data_path: str = None):
        """Initialize InputManager with data path."""
        self.data_path = Path(data_path) if data_path else Path("/data")
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    def load_all_inputs(self, source_job_id: str) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
        """
        Load and validate all input files for a training job.
        
        Args:
            source_job_id: Job ID from miStudioTrain
            
        Returns:
            Tuple of (metadata, sae_model, activation_data)
        """
        self.logger.info(f"Loading inputs for job: {source_job_id}")
        
        # Construct file paths
        models_dir = self.data_path / "models" / source_job_id
        activations_dir = self.data_path / "activations" / source_job_id
        
        sae_model_path = models_dir / "sae_model.pt"
        feature_activations_path = activations_dir / "feature_activations.pt"
        metadata_path = activations_dir / "metadata.json"
        
        # Validate files exist
        for path in [sae_model_path, feature_activations_path, metadata_path]:
            if not path.exists():
                raise InputValidationError(f"Required file not found: {path}")
        
        # Load metadata
        self.logger.info("Loading metadata...")
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        # Load SAE model
        self.logger.info("Loading SAE model...")
        sae_model = torch.load(sae_model_path, map_location='cpu')
        
        # Load activation data
        self.logger.info("Loading activation data...")
        activation_data = torch.load(feature_activations_path, map_location='cpu')
        
        self.logger.info(f"Successfully loaded all inputs for job: {source_job_id}")
        return metadata, sae_model, activation_data
