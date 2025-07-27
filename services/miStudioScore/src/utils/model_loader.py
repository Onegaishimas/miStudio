# src/utils/model_loader.py
"""
Utility functions for loading Hugging Face and custom SAE models.
"""
import torch
import logging
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, PreTrainedModel, PreTrainedTokenizer
from src.models.sae_model import SparseAutoencoder

logger = logging.getLogger(__name__)

def load_huggingface_model(model_name: str, device: str) -> (PreTrainedModel, PreTrainedTokenizer):
    """
    Loads a Hugging Face model and tokenizer for a benchmark task.
    Currently supports Question Answering models.

    Args:
        model_name: The name of the model from the Hugging Face Hub.
        device: The device to load the model onto ("cpu" or "cuda").

    Returns:
        A tuple containing the loaded model and tokenizer.
    """
    try:
        logger.info(f"Loading tokenizer: {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        logger.info(f"Loading model: {model_name} onto device: {device}")
        model = AutoModelForQuestionAnswering.from_pretrained(model_name)
        model.to(device)
        model.eval() # Set model to evaluation mode
        
        logger.info("Model and tokenizer loaded successfully.")
        return model, tokenizer
    except Exception as e:
        logger.error(f"Failed to load Hugging Face model '{model_name}': {e}", exc_info=True)
        raise

def load_sae_model(sae_path: str, device: str) -> SparseAutoencoder:
    """
    Loads a trained Sparse Autoencoder (SAE) model from a file.

    Args:
        sae_path: The path to the saved .pt or .pth SAE model file.
        device: The device to load the model onto ("cpu" or "cuda").

    Returns:
        The loaded SparseAutoencoder model.
    """
    try:
        logger.info(f"Loading SAE model from: {sae_path} onto device: {device}")
        # The SAE model file should contain the state_dict and model args
        checkpoint = torch.load(sae_path, map_location=device)
        
        # It's good practice to save model args with the state dict
        input_dim = checkpoint.get('input_dim')
        feature_dim = checkpoint.get('feature_dim')
        
        if not all([input_dim, feature_dim]):
            raise ValueError("SAE checkpoint must contain 'input_dim' and 'feature_dim'")

        model = SparseAutoencoder(input_dim=input_dim, feature_dim=feature_dim)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval() # Set model to evaluation mode

        logger.info("SAE model loaded successfully.")
        return model
    except FileNotFoundError:
        logger.error(f"SAE model file not found: {sae_path}")
        raise
    except Exception as e:
        logger.error(f"Failed to load SAE model from '{sae_path}': {e}", exc_info=True)
        raise
