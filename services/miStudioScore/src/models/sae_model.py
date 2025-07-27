# src/models/sae_model.py
"""
Defines the structure of the Sparse Autoencoder (SAE) model.
This should be consistent with the model saved by the miStudioTrain service.
"""
import torch
import torch.nn as nn

class SparseAutoencoder(nn.Module):
    """
    A simple Sparse Autoencoder model.
    """
    def __init__(self, input_dim: int, feature_dim: int):
        super().__init__()
        self.encoder = nn.Linear(input_dim, feature_dim)
        self.relu = nn.ReLU()
        self.decoder = nn.Linear(feature_dim, input_dim)

    def forward(self, x: torch.Tensor):
        """
        Forward pass through the SAE.

        Args:
            x: The input tensor from the base model's activation layer.

        Returns:
            A tuple containing the reconstructed activations and the sparse features.
        """
        features = self.relu(self.encoder(x))
        reconstructed_x = self.decoder(features)
        return reconstructed_x, features
