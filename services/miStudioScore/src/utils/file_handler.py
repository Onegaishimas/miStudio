# src/utils/file_handler.py
"""
Utility functions for reading and writing files (JSON, YAML).
"""
import json
import yaml
import logging
from typing import Dict, Any, List

logger = logging.getLogger(__name__)

def load_json_file(file_path: str) -> List[Dict[str, Any]]:
    """
    Loads a JSON file containing a list of feature objects.

    Args:
        file_path: The path to the JSON file.

    Returns:
        A list of dictionaries, where each dictionary represents a feature.
    """
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
    """
    Saves a list of feature objects to a JSON file.

    Args:
        data: The list of feature data to save.
        file_path: The path where the JSON file will be saved.
    """
    try:
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)
        logger.info(f"Successfully saved JSON to {file_path}")
    except IOError:
        logger.error(f"Could not write to file: {file_path}")
        raise

def load_yaml_config(file_path: str) -> Dict[str, Any]:
    """
    Loads a YAML configuration file.

    Args:
        file_path: The path to the YAML file.

    Returns:
        A dictionary containing the configuration.
    """
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
