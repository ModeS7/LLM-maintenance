"""
Training script for Transformer Autoencoder.
"""
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
from tqdm import tqdm
from typing import Tuple, Optional

from .data_loader import VesselDataLoader, MODEL_FEATURES
from .model import TransformerAutoencoder, get_anomaly_threshold


# Training configuration
TRAIN_CONFIG = {
    'window_size': 120,        # 10 minutes at 5s sampling
    'stride': 12,              # 1 minute stride
    'batch_size': 512,
    'epochs': 50,
    'learning_rate': 1e-4,
    'weight_decay': 1e-5,
    'train_split': 0.7,
    'val_split': 0.15,
}

# Model configuration
MODEL_CONFIG = {
    'input_dim': len(MODEL_FEATURES),
    'd_model': 64,
    'nhead': 4,
    'num_encoder_layers': 2,
    'num_decoder_layers': 2,
    'dim_feedforward': 256,
    'dropout': 0.1,
    'max_seq_len': 120,
}


def create_dataloaders(
    data_loader: VesselDataLoader,
    config: dict
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create train/val/test dataloaders."""

    # Load and split data
    train_data, val_data, test_data = data_loader.get_train_val_test_split(
        train_ratio=config['train_split'],
        val_ratio=config['val_split']
    )

    print(f"Train samples: {len(train_data.features)}")
    print(f"Val samples: {len(val_data.features)}")
    print(f"Test samples: {len(test_data.features)}")

    # Fit scaler on training data
    data_loader.fit_scaler(train_data.features)

    # Normalize all splits
    train_features = data_loader.normalize(train_data.features)
    val_features = data_loader.normalize(val_data.features)
    test_features = data_loader.normalize(test_data.features)

    # Create sequences
    train_sequences = data_loader.get_sequences(
        train_features,
        window_size=config['window_size'],
        stride=config['stride']
    )
    val_sequences = data_loader.get_sequences(
        val_features,
        window_size=config['window_size'],
        stride=config['stride']
    )
    test_sequences = data_loader.get_sequences(
        test_features,
        window_size=config['window_size'],
        stride=config['stride']
    )

    print(f"Train sequences: {len(train_sequences)}")
    print(f"Val sequences: {len(val_sequences)}")
    print(f"Test sequences: {len(test_sequences)}")

    # Convert to tensors
    train_tensor = torch.FloatTensor(train_sequences)
    val_tensor = torch.FloatTensor(val_sequences)
    test_tensor = torch.FloatTensor(test_sequences)

    # Create datasets and dataloaders
    train_dataset = TensorDataset(train_tensor)
    val_dataset = TensorDataset(val_tensor)
    test_dataset = TensorDataset(test_tensor)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=0
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=0
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=0
    )

    return train_loader, val_loader, test_loader


def train_epoch(
    model: TransformerAutoencoder,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
    total_epochs: int
) -> float:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{total_epochs} [Train]", leave=False)
    for batch in pbar:
        x = batch[0].to(device)

        optimizer.zero_grad()

        # Forward pass
        reconstruction = model(x)

        # Compute loss
        loss = criterion(reconstruction, x)

        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        pbar.set_postfix(loss=f"{loss.item():.6f}")

    return total_loss / len(train_loader)


def validate(
    model: TransformerAutoencoder,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    desc: str = "Validation"
) -> Tuple[float, np.ndarray]:
    """Validate the model."""
    model.eval()
    total_loss = 0.0
    all_errors = []

    with torch.no_grad():
        pbar = tqdm(val_loader, desc=f"[{desc}]", leave=False)
        for batch in pbar:
            x = batch[0].to(device)

            # Forward pass
            reconstruction = model(x)

            # Compute loss
            loss = criterion(reconstruction, x)
            total_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.6f}")

            # Collect reconstruction errors for threshold computation
            errors = model.compute_anomaly_score(x)
            all_errors.append(errors.cpu().numpy())

    all_errors = np.concatenate(all_errors, axis=0).flatten()

    return total_loss / len(val_loader), all_errors


def train(
    data_path: str,
    model_save_path: str,
    scaler_save_path: str,
    train_config: Optional[dict] = None,
    model_config: Optional[dict] = None,
    device: Optional[torch.device] = None
) -> TransformerAutoencoder:
    """
    Train the autoencoder model.

    Args:
        data_path: Path to the data file
        model_save_path: Path to save the trained model
        scaler_save_path: Path to save the fitted scaler
        train_config: Training configuration (uses defaults if None)
        model_config: Model configuration (uses defaults if None)
        device: Device to train on

    Returns:
        Trained model
    """
    if train_config is None:
        train_config = TRAIN_CONFIG
    if model_config is None:
        model_config = MODEL_CONFIG
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"Training on device: {device}")

    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    # Load data
    print("Loading data...")
    data_loader = VesselDataLoader(data_path)
    data_loader.load_data()

    # Create dataloaders
    print("Creating dataloaders...")
    train_loader, val_loader, test_loader = create_dataloaders(data_loader, train_config)

    # Save scaler
    data_loader.save_scaler(scaler_save_path)
    print(f"Scaler saved to {scaler_save_path}")

    # Create model
    print("Creating model...")
    model = TransformerAutoencoder(**model_config)
    model.to(device)

    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_config['learning_rate'],
        weight_decay=train_config['weight_decay']
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )

    # Training loop
    print("Starting training...")
    best_val_loss = float('inf')
    best_epoch = 0
    total_epochs = train_config['epochs']
    train_losses = []
    val_losses = []

    for epoch in range(total_epochs):
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device, epoch, total_epochs)

        # Validate
        val_loss, val_errors = validate(model, val_loader, criterion, device, desc="Val")

        # Update scheduler
        scheduler.step(val_loss)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        # Build epoch summary line
        epoch_summary = f"Epoch {epoch+1}/{total_epochs}: Train Loss = {train_loss:.6f}, Val Loss = {val_loss:.6f}"

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch

            # Compute threshold from validation errors
            threshold = get_anomaly_threshold(val_errors, percentile=95)

            # Save checkpoint
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'threshold': threshold,
                'config': model_config,
                'train_losses': train_losses,
                'val_losses': val_losses,
            }
            torch.save(checkpoint, model_save_path)
            epoch_summary += f" [BEST - threshold={threshold:.6f}]"

        print(epoch_summary)
        print()  # Blank line between epochs

    print(f"\nTraining complete. Best epoch: {best_epoch + 1}, Best val loss: {best_val_loss:.6f}")

    # Save full loss history alongside the model
    loss_history_path = str(Path(model_save_path).parent / 'loss_history.npz')
    np.savez(loss_history_path,
             train_losses=np.array(train_losses),
             val_losses=np.array(val_losses),
             best_epoch=best_epoch)
    print(f"Loss history saved to {loss_history_path}")

    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_loss, test_errors = validate(model, test_loader, criterion, device, desc="Test")
    print(f"Test Loss: {test_loss:.6f}")

    # Compute test set metrics
    threshold = get_anomaly_threshold(val_errors, percentile=95)
    n_anomalies = (test_errors > threshold).sum()
    print(f"Test anomalies (above threshold): {n_anomalies} ({100*n_anomalies/len(test_errors):.2f}%)")

    return model


def main():
    parser = argparse.ArgumentParser(description='Train Transformer Autoencoder')
    parser.add_argument('--data', type=str, default='data/Data_Pwr_All_S5.txt',
                        help='Path to data file')
    parser.add_argument('--model-output', type=str, default='models/autoencoder.pt',
                        help='Path to save model')
    parser.add_argument('--scaler-output', type=str, default='models/scaler.pkl',
                        help='Path to save scaler')
    parser.add_argument('--epochs', type=int, default=TRAIN_CONFIG['epochs'],
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=TRAIN_CONFIG['batch_size'],
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=TRAIN_CONFIG['learning_rate'],
                        help='Learning rate')

    args = parser.parse_args()

    # Update config with CLI args
    config = TRAIN_CONFIG.copy()
    config['epochs'] = args.epochs
    config['batch_size'] = args.batch_size
    config['learning_rate'] = args.lr

    # Ensure output directory exists
    Path(args.model_output).parent.mkdir(parents=True, exist_ok=True)

    # Train
    train(
        data_path=args.data,
        model_save_path=args.model_output,
        scaler_save_path=args.scaler_output,
        train_config=config
    )


if __name__ == '__main__':
    main()
