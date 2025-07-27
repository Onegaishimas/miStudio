# src/scorers/ablation_scorer.py
"""
Implementation of the Task-Based Utility Scorer using feature ablation.
"""
import torch
import logging
import importlib.util
from typing import List, Dict, Any, Callable
from tqdm import tqdm

from .base_scorer import BaseScorer
from src.utils.model_loader import load_huggingface_model, load_sae_model

logger = logging.getLogger(__name__)

# Global variable to hold the hook and the feature index to ablate
ablation_hook = None
feature_to_ablate_idx = -1

def get_target_layer(model: torch.nn.Module, layer_name: str):
    """
    Retrieves a specific layer from a model using its string name.
    """
    modules = layer_name.split('.')
    for module_name in modules:
        model = getattr(model, module_name)
    return model

def ablation_forward_hook(module: torch.nn.Module, inputs: tuple, outputs: torch.Tensor):
    """
    The hook function that performs the ablation.
    It zeroes out the activation of the specified feature.
    """
    global feature_to_ablate_idx
    if feature_to_ablate_idx != -1:
        # Clone the output to avoid in-place modification errors
        modified_output = outputs.clone()
        modified_output[:, :, feature_to_ablate_idx] = 0
        return modified_output
    return outputs


class AblationScorer(BaseScorer):
    """
    Scores features by measuring the drop in model performance (utility)
    when a feature is "ablated" or removed from the model's computation.
    """

    @property
    def name(self) -> str:
        return "ablation_scorer"

    def _run_benchmark(self, benchmark_fn: Callable, model, tokenizer, device) -> float:
        """Helper to run the benchmark and return a score."""
        try:
            return benchmark_fn(model, tokenizer, device)
        except Exception as e:
            logger.error(f"Error executing benchmark function: {e}", exc_info=True)
            raise

    def score(self, features: List[Dict[str, Any]], **kwargs) -> List[Dict[str, Any]]:
        global ablation_hook, feature_to_ablate_idx

        # --- 1. Extract parameters and validate ---
        score_name = kwargs.get("name")
        benchmark_path = kwargs.get("benchmark_dataset_path")
        model_name = kwargs.get("target_model_name")
        layer_name = kwargs.get("target_model_layer")
        device = kwargs.get("device", "cpu")

        if not all([score_name, benchmark_path, model_name, layer_name]):
            raise ValueError("AblationScorer requires 'name', 'benchmark_dataset_path', 'target_model_name', and 'target_model_layer'.")

        # --- 2. Load benchmark function dynamically ---
        logger.info(f"Loading benchmark function from {benchmark_path}")
        spec = importlib.util.spec_from_file_location("benchmark", benchmark_path)
        benchmark_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(benchmark_module)
        run_benchmark_fn = benchmark_module.run_benchmark

        # --- 3. Load models ---
        model, tokenizer = load_huggingface_model(model_name, device)
        # The SAE model is not directly used here but its existence is a prerequisite
        
        # --- 4. Get baseline performance ---
        logger.info("Calculating baseline model performance...")
        baseline_score = self._run_benchmark(run_benchmark_fn, model, tokenizer, device)
        logger.info(f"Baseline performance score: {baseline_score:.4f}")

        # --- 5. Register hook and run ablation for each feature ---
        target_layer = get_target_layer(model, layer_name)
        ablation_hook = target_layer.register_forward_hook(ablation_forward_hook)
        
        logger.info(f"Starting ablation scoring for {len(features)} features...")
        for feature in tqdm(features, desc=f"Ablation Scoring for '{score_name}'"):
            feature_idx = feature.get("feature_index")
            if feature_idx is None:
                feature[score_name] = 0.0
                continue

            feature_to_ablate_idx = feature_idx
            
            ablated_score = self._run_benchmark(run_benchmark_fn, model, tokenizer, device)
            
            # The utility score is the change from baseline
            utility_score = ablated_score - baseline_score
            feature[score_name] = round(utility_score, 6)

        # --- 6. Cleanup ---
        ablation_hook.remove()
        feature_to_ablate_idx = -1
        logger.info(f"Ablation scoring for '{score_name}' complete.")
        
        return features
