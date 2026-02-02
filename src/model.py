"""
Transformer Autoencoder for anomaly detection.
"""
import math
import numpy as np
import torch
import torch.nn as nn
from typing import Optional, Tuple


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for transformer."""

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create positional encoding matrix
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))

        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Register as buffer (not a parameter)
        self.register_buffer('pe', pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (batch, seq_len, d_model)
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TransformerAutoencoder(nn.Module):
    """
    Transformer-based autoencoder for time series anomaly detection.

    Architecture:
        Input → Linear Embedding → Positional Encoding →
        Transformer Encoder → Latent →
        Transformer Decoder → Output Projection → Reconstruction
    """

    def __init__(
        self,
        input_dim: int = 16,
        d_model: int = 64,
        nhead: int = 4,
        num_encoder_layers: int = 2,
        num_decoder_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        max_seq_len: int = 120
    ):
        super().__init__()

        self.input_dim = input_dim
        self.d_model = d_model
        self.max_seq_len = max_seq_len

        # Input embedding
        self.input_projection = nn.Linear(input_dim, d_model)

        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, max_seq_len, dropout)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        # Transformer decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)

        # Output projection
        self.output_projection = nn.Linear(d_model, input_dim)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize model weights."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode input sequence to latent representation.

        Args:
            x: Input tensor of shape (batch, seq_len, input_dim)

        Returns:
            Latent tensor of shape (batch, seq_len, d_model)
        """
        # Project input to d_model dimensions
        x = self.input_projection(x)  # (batch, seq_len, d_model)

        # Add positional encoding
        x = self.pos_encoder(x)

        # Encode
        latent = self.encoder(x)  # (batch, seq_len, d_model)

        return latent

    def decode(self, latent: torch.Tensor, target: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Decode latent representation to reconstruction.

        Args:
            latent: Latent tensor of shape (batch, seq_len, d_model)
            target: Target tensor for decoder input (uses latent if None)

        Returns:
            Reconstruction of shape (batch, seq_len, input_dim)
        """
        if target is None:
            target = latent

        # Decode
        decoded = self.decoder(target, latent)  # (batch, seq_len, d_model)

        # Project to output dimensions
        output = self.output_projection(decoded)  # (batch, seq_len, input_dim)

        return output

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: encode then decode.

        Args:
            x: Input tensor of shape (batch, seq_len, input_dim)

        Returns:
            Reconstruction of shape (batch, seq_len, input_dim)
        """
        latent = self.encode(x)
        reconstruction = self.decode(latent)
        return reconstruction

    def compute_reconstruction_error(
        self,
        x: torch.Tensor,
        reduction: str = 'none'
    ) -> torch.Tensor:
        """
        Compute reconstruction error (MSE).

        Args:
            x: Input tensor of shape (batch, seq_len, input_dim)
            reduction: 'none' returns per-feature error, 'mean' averages over features

        Returns:
            Error tensor of shape (batch, seq_len, input_dim) or (batch, seq_len)
        """
        reconstruction = self.forward(x)
        error = (x - reconstruction) ** 2

        if reduction == 'mean':
            error = error.mean(dim=-1)  # Average over features

        return error

    def compute_anomaly_score(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute per-timestep anomaly score.

        Args:
            x: Input tensor of shape (batch, seq_len, input_dim)

        Returns:
            Anomaly scores of shape (batch, seq_len)
        """
        return self.compute_reconstruction_error(x, reduction='mean')

    def get_feature_errors(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get per-feature reconstruction errors.

        Args:
            x: Input tensor of shape (batch, seq_len, input_dim)

        Returns:
            Feature errors of shape (batch, seq_len, input_dim)
        """
        return self.compute_reconstruction_error(x, reduction='none')


def get_anomaly_threshold(
    validation_errors: np.ndarray,
    percentile: float = 95
) -> float:
    """
    Compute anomaly threshold from validation data.

    Args:
        validation_errors: Array of validation reconstruction errors
        percentile: Percentile for threshold (95 = top 5% are anomalies)

    Returns:
        Threshold value
    """
    return float(np.percentile(validation_errors, percentile))


def load_model(
    model_path: str,
    device: Optional[torch.device] = None
) -> Tuple[TransformerAutoencoder, dict]:
    """
    Load a trained model from checkpoint.

    Args:
        model_path: Path to the model checkpoint
        device: Device to load model on

    Returns:
        Tuple of (model, metadata)
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    # Get model config
    config = checkpoint.get('config', {})

    # Create model with saved config
    model = TransformerAutoencoder(
        input_dim=config.get('input_dim', 16),
        d_model=config.get('d_model', 64),
        nhead=config.get('nhead', 4),
        num_encoder_layers=config.get('num_encoder_layers', 2),
        num_decoder_layers=config.get('num_decoder_layers', 2),
        dim_feedforward=config.get('dim_feedforward', 256),
        dropout=config.get('dropout', 0.1),
        max_seq_len=config.get('max_seq_len', 120)
    )

    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    metadata = {
        'threshold': checkpoint.get('threshold', 0.1),
        'config': config,
        'epoch': checkpoint.get('epoch', 0),
        'train_loss': checkpoint.get('train_loss', None),
        'val_loss': checkpoint.get('val_loss', None)
    }

    return model, metadata
