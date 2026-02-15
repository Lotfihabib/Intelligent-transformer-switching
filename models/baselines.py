"""
Baseline Models Module

Implements simple baseline models for comparison with TFT:
- PersistenceModel: Naive persistence (repeat last value)
- SimpleLSTM: Basic 2-layer LSTM with linear decoder

All models provide TFT-compatible interface for drop-in comparison.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from pathlib import Path

from config import CONFIG


class PersistenceModel:
    """
    Naive persistence baseline: repeats last observed value for entire horizon.

    This is the simplest possible forecast and serves as a baseline to
    demonstrate that more complex models add value.

    Provides TFT-compatible interface for drop-in comparison.
    """

    def __init__(self, horizon=24):
        """
        Initialize persistence model.

        Args:
            horizon: Forecast horizon length (default: 24 steps = 4 hours)
        """
        self.horizon = horizon
        self.model_type = 'Persistence'
        self.device = 'cpu'

    def eval(self):
        """Set to evaluation mode (no-op for persistence)"""
        pass

    def train(self):
        """Set to training mode (no-op for persistence)"""
        pass

    def to(self, device):
        """Move to device (no-op for persistence)"""
        self.device = device
        return self

    def forward(self, features, hist_target):
        """
        Generate persistence forecast.

        Args:
            features: torch.Tensor [batch, seq_len, n_features] (ignored)
            hist_target: torch.Tensor [batch, seq_len]

        Returns:
            torch.Tensor [batch, horizon, 1] predictions (point forecast, no quantiles)
        """
        # Take last observed value
        last_value = hist_target[:, -1]  # [batch]

        # Repeat for entire horizon
        # Shape: [batch] → [batch, 1, 1] → [batch, horizon, 1]
        forecast = last_value.unsqueeze(1).unsqueeze(2).repeat(1, self.horizon, 1)

        return forecast

    def __call__(self, features, hist_target):
        """Make model callable"""
        return self.forward(features, hist_target)


class SimpleLSTM(nn.Module):
    """
    Simple LSTM baseline for point forecasting (no quantiles).

    Architecture:
        - 2-layer LSTM encoder processes historical sequence
        - Dropout for regularization
        - Linear decoder projects last hidden state to horizon

    Serves as a reasonable baseline showing value of attention mechanism
    and quantile prediction in TFT.
    """

    def __init__(self, input_dim, hidden_dim=64, num_layers=2, horizon=24, dropout=0.1):
        """
        Initialize SimpleLSTM model.

        Args:
            input_dim: Number of input features
            hidden_dim: Hidden dimension size (default: 64)
            num_layers: Number of LSTM layers (default: 2)
            horizon: Forecast horizon length (default: 24)
            dropout: Dropout rate (default: 0.1)
        """
        super().__init__()
        self.horizon = horizon
        self.model_type = 'LSTM'
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # LSTM encoder
        # Input: features + target concatenated
        self.encoder = nn.LSTM(
            input_dim + 1,  # +1 for target
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # Dropout layer
        self.dropout = nn.Dropout(dropout)

        # Linear decoder: hidden → horizon
        self.decoder = nn.Linear(hidden_dim, horizon)

    def forward(self, features, hist_target):
        """
        Generate LSTM forecast.

        Args:
            features: torch.Tensor [batch, seq_len, input_dim]
            hist_target: torch.Tensor [batch, seq_len]

        Returns:
            torch.Tensor [batch, horizon, 1] point forecasts
        """
        # Concatenate target as additional feature
        # hist_target: [batch, seq_len] → [batch, seq_len, 1]
        target_expanded = hist_target.unsqueeze(-1)

        # Concatenate along feature dimension
        # [batch, seq_len, input_dim] + [batch, seq_len, 1] → [batch, seq_len, input_dim+1]
        x = torch.cat([features, target_expanded], dim=-1)

        # Encode sequence through LSTM
        # output: [batch, seq_len, hidden_dim]
        # h_n: [num_layers, batch, hidden_dim]
        # c_n: [num_layers, batch, hidden_dim]
        output, (h_n, c_n) = self.encoder(x)

        # Use last hidden state from top layer
        last_hidden = h_n[-1]  # [batch, hidden_dim]

        # Apply dropout
        last_hidden = self.dropout(last_hidden)

        # Decode to horizon
        forecast = self.decoder(last_hidden)  # [batch, horizon]

        # Reshape to [batch, horizon, 1] for consistency with TFT
        forecast = forecast.unsqueeze(-1)

        return forecast


def train_baseline_lstm(train_loader, val_loader, input_dim, epochs=5, device='cpu',
                        metadata=None, outdir='./outputs'):
    """
    Train SimpleLSTM baseline model with MSE loss.

    Args:
        train_loader: Training DataLoader
        val_loader: Validation DataLoader
        input_dim: Number of input features
        epochs: Number of training epochs (default: 20)
        device: torch device ('cuda' or 'cpu')
        metadata: Preprocessing metadata dict (for denormalization)
        outdir: Output directory for saving model

    Returns:
        Trained SimpleLSTM model
    """
    print(f"\n[TRAINING] SimpleLSTM Baseline")
    print(f"  Input features: {input_dim}")
    print(f"  Hidden dim: 64")
    print(f"  Horizon: {CONFIG['H']}")
    print(f"  Epochs: {epochs}")
    print(f"  Device: {device}")

    # Create model
    model = SimpleLSTM(
        input_dim=input_dim,
        hidden_dim=64,
        num_layers=2,
        horizon=CONFIG['H'],
        dropout=0.1
    ).to(device)

    # Optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=0.001,  # Higher LR than TFT since simpler model
        weight_decay=0.0001
    )

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5
        # Note: verbose parameter removed in newer PyTorch versions
    )

    # Loss function: Mean Squared Error
    criterion = nn.MSELoss()

    # Training history
    history = {
        'train_losses': [],
        'val_losses': [],
        'best_val_loss': float('inf'),
        'best_epoch': 0
    }

    # Early stopping
    patience = 10
    patience_counter = 0
    best_model_state = None

    # Training loop
    print("\nTraining progress:")
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0

        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs} [Train]', leave=False)
        for features, hist_target, future_target in train_pbar:
            features = features.to(device)
            hist_target = hist_target.to(device)
            future_target = future_target.to(device)

            # Forward pass
            predictions = model(features, hist_target)  # [batch, horizon, 1]
            predictions = predictions.squeeze(-1)  # [batch, horizon]

            # Compute loss
            loss = criterion(predictions, future_target)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()

            # Gradient clipping (prevent exploding gradients)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            train_loss += loss.item()
            train_pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        train_loss /= len(train_loader)

        # Validation phase
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{epochs} [Val]', leave=False)
            for features, hist_target, future_target in val_pbar:
                features = features.to(device)
                hist_target = hist_target.to(device)
                future_target = future_target.to(device)

                # Forward pass
                predictions = model(features, hist_target)
                predictions = predictions.squeeze(-1)

                # Compute loss
                loss = criterion(predictions, future_target)
                val_loss += loss.item()
                val_pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        val_loss /= len(val_loader)

        # Learning rate scheduling
        scheduler.step(val_loss)

        # Store history
        history['train_losses'].append(train_loss)
        history['val_losses'].append(val_loss)

        # Print epoch summary
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1}/{epochs} - "
              f"Train Loss: {train_loss:.4f}, "
              f"Val Loss: {val_loss:.4f}, "
              f"LR: {current_lr:.2e}")

        # Early stopping check
        if val_loss < history['best_val_loss'] - 0.001:
            history['best_val_loss'] = val_loss
            history['best_epoch'] = epoch + 1
            patience_counter = 0
            best_model_state = model.state_dict().copy()
            print(f"  → New best validation loss: {val_loss:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n[EARLY STOP] No improvement for {patience} epochs")
                break

    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"\n[OK] Loaded best model from epoch {history['best_epoch']}")

    # Save model
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    model_path = outdir / 'baseline_lstm.pt'
    torch.save(model.state_dict(), model_path)
    print(f"[OK] LSTM baseline saved: {model_path}")

    # Save model config
    lstm_config = {
        'input_dim': input_dim,
        'hidden_dim': 64,
        'num_layers': 2,
        'horizon': CONFIG['H'],
        'dropout': 0.1
    }
    config_path = outdir / 'baseline_lstm_config.json'
    import json
    with open(config_path, 'w') as f:
        json.dump(lstm_config, f, indent=2)
    print(f"[OK] LSTM config saved: {config_path}")

    # Save training history
    history_path = outdir / 'baseline_lstm_history.json'
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"[OK] LSTM history saved: {history_path}")

    print(f"\n[SUMMARY] LSTM Training Complete")
    print(f"  Best validation loss: {history['best_val_loss']:.4f}")
    print(f"  Best epoch: {history['best_epoch']}/{epochs}")
    print(f"  Final train loss: {history['train_losses'][-1]:.4f}")
    print(f"  Final val loss: {history['val_losses'][-1]:.4f}")

    return model


def load_baseline_lstm(model_path, config_path, device='cpu'):
    """
    Load saved SimpleLSTM model from disk.

    Args:
        model_path: Path to saved model weights (.pt file)
        config_path: Path to model config (.json file)
        device: torch device

    Returns:
        Loaded SimpleLSTM model
    """
    import json

    # Load config
    with open(config_path, 'r') as f:
        config = json.load(f)

    # Create model
    model = SimpleLSTM(
        input_dim=config['input_dim'],
        hidden_dim=config['hidden_dim'],
        num_layers=config['num_layers'],
        horizon=config['horizon'],
        dropout=config.get('dropout', 0.1)
    )

    # Load weights
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()

    return model
