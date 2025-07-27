# src/core/orchestrator.py
"""
The main workflow orchestrator for the miStudioScore service.
"""
import logging
import importlib
from typing import Dict, Any, List, Type
from src.utils.file_handler import load_json_file, save_json_file, load_yaml_config
from src.scorers.base_scorer import BaseScorer

logger = logging.getLogger(__name__)

class ScoringOrchestrator:
    """
    Manages the entire scoring workflow from loading data to executing
    scorers and saving the results.
    """

    def __init__(self, config_path: str):
        self.config = load_yaml_config(config_path)
        self.scorers: Dict[str, Type[BaseScorer]] = self._load_scorers()

    def _load_scorers(self) -> Dict[str, Type[BaseScorer]]:
        """
        Dynamically imports and loads all available scorer classes from the
        'src.scorers' module.
        """
        scorers_map = {}
        # Add the new scorer to the list of modules to be loaded
        scorer_modules = ["relevance_scorer", "ablation_scorer"]
        for module_name in scorer_modules:
            try:
                module = importlib.import_module(f"src.scorers.{module_name}")
                for attr_name in dir(module):
                    attr = getattr(module, attr_name)
                    if isinstance(attr, type) and issubclass(attr, BaseScorer) and attr is not BaseScorer:
                        instance = attr()
                        scorers_map[instance.name] = attr
                        logger.info(f"Successfully loaded scorer: {instance.name}")
            except ImportError as e:
                logger.error(f"Could not import scorer module {module_name}: {e}")
        return scorers_map

    def run(self, features_path: str, output_dir: str) -> (str, List[str]):
        """
        Executes the full scoring pipeline based on the configuration.

        Args:
            features_path: Path to the input features.json.
            output_dir: Directory to save the output file.

        Returns:
            A tuple containing the output file path and a list of added score names.
        """
        features = load_json_file(features_path)
        added_scores = []

        scoring_jobs = self.config.get("scoring_jobs", [])
        if not scoring_jobs:
            logger.warning("No scoring jobs found in the configuration file.")
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

        # Save the enriched data
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        output_filename = f"scores_{timestamp}.json"
        output_path = f"{output_dir}/{output_filename}"
        
        save_json_file(features, output_path)

        return output_path, added_scores
