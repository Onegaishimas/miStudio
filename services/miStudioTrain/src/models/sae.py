# =============================================================================
# models/sae.py - Memory Optimized Version
# =============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Any


class SparseAutoencoder(nn.Module):
    """Memory-optimized Sparse Autoencoder for learning interpretable features"""

    def __init__(self, input_dim: int, hidden_dim: int, sparsity_coeff: float = 1e-3):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.sparsity_coeff = sparsity_coeff

        # Encoder: input_dim -> hidden_dim
        self.encoder = nn.Linear(input_dim, hidden_dim, bias=True)

        # Decoder: hidden_dim -> input_dim
        self.decoder = nn.Linear(hidden_dim, input_dim, bias=True)

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize weights to prevent dead neurons"""
        nn.init.xavier_uniform_(self.encoder.weight)
        nn.init.constant_(self.encoder.bias, -0.1)
        nn.init.constant_(self.decoder.bias, 0.0)

        with torch.no_grad():
            self.decoder.weight.data = self.encoder.weight.data.t()

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass through SAE with memory optimization"""
        # Use in-place operations where possible to save memory
        hidden = F.relu(self.encoder(x))
        reconstruction = self.decoder(hidden)

        # Compute losses efficiently
        recon_loss = F.mse_loss(reconstruction, x)
        sparsity_loss = self.sparsity_coeff * torch.mean(torch.abs(hidden))
        total_loss = recon_loss + sparsity_loss

        return reconstruction, hidden, total_loss

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to sparse features"""
        return F.relu(self.encoder(x))

    def decode(self, hidden: torch.Tensor) -> torch.Tensor:
        """Decode hidden features back to input space"""
        return self.decoder(hidden)

    def get_feature_stats(self, dataloader) -> Dict[str, Any]:
        """Compute statistics about learned features with memory management"""
        self.eval()
        all_features = []
        device = next(self.parameters()).device

        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                try:
                    if isinstance(batch, (list, tuple)):
                        batch_data = batch[0]
                    else:
                        batch_data = batch
                    
                    batch_data = batch_data.to(device)
                    features = self.encode(batch_data)
                    all_features.append(features.cpu())
                    
                    # Clear GPU memory periodically
                    if batch_idx % 50 == 0 and device.type == "cuda":
                        torch.cuda.empty_cache()
                        
                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        torch.cuda.empty_cache()
                        continue
                    else:
                        raise e

        if not all_features:
            # Return empty stats if no features were computed
            return {
                "total_features": self.hidden_dim,
                "active_features_count": 0,
                "dead_features_count": self.hidden_dim,
                "avg_activation_frequency": 0.0,
                "feature_activation_rates": [0.0] * self.hidden_dim,
                "feature_mean_activations": [0.0] * self.hidden_dim,
                "feature_max_activations": [0.0] * self.hidden_dim,
                "error": "No features computed due to memory constraints"
            }

        # Concatenate all features
        all_features = torch.cat(all_features, dim=0)

        # Compute statistics efficiently
        feature_means = torch.mean(all_features, dim=0)
        feature_maxes = torch.max(all_features, dim=0)[0]
        active_features = torch.sum(all_features > 0, dim=0)
        total_samples = all_features.shape[0]

        # Convert to lists for JSON serialization
        stats = {
            "total_features": self.hidden_dim,
            "active_features_count": int(torch.sum(active_features > 0)),
            "dead_features_count": int(torch.sum(active_features == 0)),
            "avg_activation_frequency": float(torch.mean(active_features.float() / total_samples)),
            "feature_activation_rates": (active_features.float() / total_samples).tolist(),
            "feature_mean_activations": feature_means.tolist(),
            "feature_max_activations": feature_maxes.tolist(),
            "total_samples_processed": total_samples,
        }

        # Clear memory after computation
        del all_features, feature_means, feature_maxes, active_features
        if device.type == "cuda":
            torch.cuda.empty_cache()

        return stats

    def get_dead_features(self, dataloader, threshold: float = 1e-6) -> torch.Tensor:
        """Identify dead features (features that rarely activate)"""
        self.eval()
        feature_activity = torch.zeros(self.hidden_dim)
        total_samples = 0
        device = next(self.parameters()).device

        with torch.no_grad():
            for batch in dataloader:
                try:
                    if isinstance(batch, (list, tuple)):
                        batch_data = batch[0]
                    else:
                        batch_data = batch
                    
                    batch_data = batch_data.to(device)
                    features = self.encode(batch_data)
                    
                    # Count activations above threshold
                    batch_activity = torch.sum(features > threshold, dim=0).cpu()
                    feature_activity += batch_activity
                    total_samples += features.shape[0]
                    
                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        torch.cuda.empty_cache()
                        continue
                    else:
                        raise e

        # Compute activation frequency
        activation_frequency = feature_activity / total_samples
        dead_features = activation_frequency < 0.01  # Less than 1% activation
        
        return dead_features

    def reinitialize_dead_features(self, dataloader, threshold: float = 1e-6):
        """Reinitialize dead features to prevent feature collapse"""
        dead_features = self.get_dead_features(dataloader, threshold)
        num_dead = torch.sum(dead_features).item()
        
        if num_dead > 0:
            print(f"Reinitializing {num_dead} dead features...")
            
            with torch.no_grad():
                # Reinitialize encoder weights for dead features
                dead_indices = torch.where(dead_features)[0]
                for idx in dead_indices:
                    nn.init.xavier_uniform_(self.encoder.weight[idx:idx+1])
                    self.encoder.bias[idx] = -0.1
                    
                    # Update corresponding decoder weights
                    nn.init.xavier_uniform_(self.decoder.weight[:, idx:idx+1])

    def compute_reconstruction_error(self, dataloader) -> float:
        """Compute average reconstruction error across dataset"""
        self.eval()
        total_error = 0.0
        total_samples = 0
        device = next(self.parameters()).device

        with torch.no_grad():
            for batch in dataloader:
                try:
                    if isinstance(batch, (list, tuple)):
                        batch_data = batch[0]
                    else:
                        batch_data = batch
                    
                    batch_data = batch_data.to(device)
                    reconstruction, _, _ = self.forward(batch_data)
                    
                    error = F.mse_loss(reconstruction, batch_data, reduction='sum')
                    total_error += error.item()
                    total_samples += batch_data.shape[0]
                    
                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        torch.cuda.empty_cache()
                        continue
                    else:
                        raise e

        return total_error / total_samples if total_samples > 0 else float('inf')

    def save_checkpoint(self, path: str, optimizer=None, epoch: int = 0, loss: float = 0.0):
        """Save model checkpoint with training state"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.state_dict(),
            'loss': loss,
            'input_dim': self.input_dim,
            'hidden_dim': self.hidden_dim,
            'sparsity_coeff': self.sparsity_coeff,
        }
        
        if optimizer is not None:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()
        
        torch.save(checkpoint, path)

    @classmethod
    def load_checkpoint(cls, path: str, device: torch.device = None):
        """Load model from checkpoint"""
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        checkpoint = torch.load(path, map_location=device)
        
        model = cls(
            input_dim=checkpoint['input_dim'],
            hidden_dim=checkpoint['hidden_dim'],
            sparsity_coeff=checkpoint['sparsity_coeff']
        )
        
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        
        return model, checkpoint