"""
Model Training Module

This module handles the complete training pipeline for the Temporal Fusion Transformer model.
Provides a clean, modular interface with helper functions for dataset/model creation.
"""

import torch
import torch.optim as optim
from tqdm import tqdm
from pathlib import Path
import json
import numpy as np

from config import CONFIG
from .loss import pinball_loss
from .dataset import SimpleTimeSeriesDataset
from .networks import TemporalFusionTransformer
from torch.utils.data import DataLoader

# Try to import evaluation metrics
try:
    from evaluation.metrics import compute_evaluation_metrics
except ImportError:
    compute_evaluation_metrics = None


# JSON Encoder for numpy types
class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for numpy types"""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def create_datasets_and_loaders(df_processed, target_col='S_TOTAL', split=None):
    """
    Create PyTorch datasets and dataloaders from preprocessed DataFrame.

    Args:
        df_processed: Preprocessed DataFrame with 'split' column
        target_col: Name of target column (default: 'S_TOTAL')
        split: If specified, only create loader for this split ('train', 'val', or 'test').
               If None, creates train and val loaders (default behavior).

    Returns:
        tuple: If split is None: (train_loader, val_loader, feature_columns)
               If split is specified: (None, None, single_loader) where single_loader is for the specified split
    """
    # Get feature columns
    feature_cols = [col for col in df_processed.columns if col not in [target_col, 'split']]

    # If specific split requested, create only that loader
    if split is not None:
        dataset = SimpleTimeSeriesDataset(
            df_processed,
            target_col=target_col,
            seq_len=CONFIG['sequence_length'],
            horizon=CONFIG['H'],
            split=split
        )

        loader = DataLoader(
            dataset,
            batch_size=CONFIG['batch_size'],
            shuffle=(split == 'train'),
            drop_last=(split == 'train')
        )

        return None, None, loader

    # Default behavior: create train and val datasets
    train_dataset = SimpleTimeSeriesDataset(
        df_processed,
        target_col=target_col,
        seq_len=CONFIG['sequence_length'],
        horizon=CONFIG['H'],
        split='train'
    )

    val_dataset = SimpleTimeSeriesDataset(
        df_processed,
        target_col=target_col,
        seq_len=CONFIG['sequence_length'],
        horizon=CONFIG['H'],
        split='val'
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=CONFIG['batch_size'],
        shuffle=True,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=CONFIG['batch_size'],
        shuffle=False
    )

    return train_loader, val_loader, feature_cols


def create_tft_model(n_features, feature_cols=None, device='cpu'):
    """
    Create and initialize Temporal Fusion Transformer model.

    Args:
        n_features: Number of input features
        feature_cols: List of feature column names (optional)
        device: Device to place model on ('cpu' or 'cuda')

    Returns:
        TemporalFusionTransformer model on specified device
    """
    model = TemporalFusionTransformer(
        input_dim=n_features,
        d_model=128,
        num_heads=4,
        num_layers=2,
        horizon=CONFIG['H'],
        num_quantiles=3
    )

    # Initialize with feature names if provided
    if feature_cols is not None:
        model.initialize_from_data(feature_cols)

    # Move to device
    model = model.to(device)

    return model


def save_model_and_history(model, outdir, training_history=None):
    """
    Save model weights, architecture config, and training history.

    Args:
        model: Trained PyTorch model
        outdir: Output directory path
        training_history: Optional dict with training metrics
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Save model weights
    model_path = outdir / 'model.pt'
    torch.save(model.state_dict(), model_path)
    print(f"[OK] Model saved: {model_path}")

    # Save model configuration
    from config import CONFIG
    model_config = {
        'input_dim': model.input_dim,
        'd_model': model.d_model,
        'num_heads': model.num_heads,
        'num_layers': model.num_layers,
        'horizon': model.horizon,
        'num_quantiles': model.num_quantiles,
        'dropout_rate': CONFIG['dropout_rate']  # Get from CONFIG since model doesn't store it
    }
    config_path = outdir / 'model_config.json'
    with open(config_path, 'w') as f:
        json.dump(model_config, f, indent=2)
    print(f"[OK] Model config saved: {config_path}")

    # Save training history
    if training_history is not None:
        history_path = outdir / 'training_history.json'
        history_data = {
            'train_losses': training_history.get('train_losses', []),
            'val_losses': training_history.get('val_losses', []),
            'train_metrics': training_history.get('train_metrics', []),
            'val_metrics': training_history.get('val_metrics', []),
            # NEW: Add multistep metrics if available
            'val_multistep_point': training_history.get('val_multistep_point', []),
            'val_multistep_prob': training_history.get('val_multistep_prob', []),
            'epochs': training_history.get('epochs', 0)
        }
        with open(history_path, 'w') as f:
            json.dump(history_data, f, indent=2, cls=NumpyEncoder)
        print(f"[OK] Training history saved: {history_path}")


# =============================================================================
# CORE TRAINING FUNCTION
# =============================================================================

def train_model(model, train_loader, val_loader, epochs=20, device='cpu', metadata=None):
    """
    Core training loop with early stopping and metric tracking.

    Args:
        model: PyTorch model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        epochs: Number of training epochs
        device: Device ('cpu' or 'cuda')
        metadata: Dict with normalization stats for denormalization

    Returns:
        Trained model with training_history attribute
    """
    # Extract denormalization parameters
    target_mean = metadata.get('normalization', {}).get('target_mean', 0.0) if metadata else 0.0
    target_std = metadata.get('normalization', {}).get('target_std', 1.0) if metadata else 1.0

    # Setup optimizer and scheduler
    optimizer = optim.Adam(model.parameters(), lr=CONFIG['learning_rate'], weight_decay=CONFIG['weight_decay'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=8, min_lr=1e-6)

    # Early stopping
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None

    # Training history
    history = {
        'train_losses': [],
        'val_losses': [],
        'train_metrics': [],
        'val_metrics': []
    }

    # Learning rate warmup
    warmup_epochs = min(5, epochs // 4)
    base_lr = CONFIG['learning_rate']

    # Training loop
    epoch_pbar = tqdm(range(epochs), desc="Training", unit="epoch")
    for epoch in epoch_pbar:
        # Warmup
        if epoch < warmup_epochs:
            lr = base_lr * (epoch + 1) / warmup_epochs
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

        # === TRAINING PHASE ===
        model.train()
        train_loss = 0.0
        train_preds, train_targs = [], []

        for features, hist_target, targets in train_loader:
            features, hist_target, targets = features.to(device), hist_target.to(device), targets.to(device)

            optimizer.zero_grad()
            predictions = model(features, hist_target)
            loss = pinball_loss(predictions, targets, CONFIG['quantiles'],
                              smoothing=0.05, quantile_weights=CONFIG['quantile_weights'])

            if not (torch.isnan(loss) or torch.isinf(loss)):
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
                optimizer.step()
                train_loss += loss.item()
                train_preds.append(predictions.detach())
                train_targs.append(targets.detach())

        train_loss /= len(train_loader)

        # === VALIDATION PHASE ===
        model.eval()
        val_loss = 0.0
        val_preds, val_targs = [], []

        with torch.no_grad():
            for features, hist_target, targets in val_loader:
                features, hist_target, targets = features.to(device), hist_target.to(device), targets.to(device)
                predictions = model(features, hist_target)
                loss = pinball_loss(predictions, targets, CONFIG['quantiles'],
                                  smoothing=0.0, quantile_weights=CONFIG['quantile_weights'])
                val_loss += loss.item()
                val_preds.append(predictions.detach())
                val_targs.append(targets.detach())

        val_loss /= len(val_loader)
        scheduler.step(val_loss)

        # Store losses
        history['train_losses'].append(train_loss)
        history['val_losses'].append(val_loss)

        # Compute metrics if available
        if compute_evaluation_metrics and train_preds and val_preds:
            # Denormalize for metrics
            train_pred_cat = torch.cat(train_preds, dim=0) * target_std + target_mean
            train_targ_cat = torch.cat(train_targs, dim=0) * target_std + target_mean
            val_pred_cat = torch.cat(val_preds, dim=0) * target_std + target_mean
            val_targ_cat = torch.cat(val_targs, dim=0) * target_std + target_mean

            history['train_metrics'].append(
                compute_evaluation_metrics(train_pred_cat, train_targ_cat, CONFIG['quantiles'])
            )
            history['val_metrics'].append(
                compute_evaluation_metrics(val_pred_cat, val_targ_cat, CONFIG['quantiles'])
            )

            # NEW: Optional multi-step metrics tracking (controlled by config)
            if CONFIG.get('compute_multistep_metrics', False):
                try:
                    from evaluation.multistep_metrics import (
                        compute_multistep_point_metrics,
                        compute_multistep_probabilistic_metrics
                    )

                    # Compute for key horizons only to save time
                    key_horizons = CONFIG.get('evaluation_horizons', [1, 6, 12, 24])
                    # Filter to valid horizons based on actual horizon length
                    horizon_len = val_pred_cat.shape[1]
                    key_horizons = [h for h in key_horizons if h <= horizon_len]

                    val_multistep_point = compute_multistep_point_metrics(
                        val_pred_cat, val_targ_cat, horizons=key_horizons, quantiles=CONFIG['quantiles']
                    )
                    val_multistep_prob = compute_multistep_probabilistic_metrics(
                        val_pred_cat, val_targ_cat, horizons=key_horizons, quantiles=CONFIG['quantiles']
                    )

                    # Store in history
                    if 'val_multistep_point' not in history:
                        history['val_multistep_point'] = []
                        history['val_multistep_prob'] = []

                    history['val_multistep_point'].append(val_multistep_point)
                    history['val_multistep_prob'].append(val_multistep_prob)

                except ImportError:
                    pass  # Silently skip if multistep_metrics not available

        # Update progress
        current_lr = optimizer.param_groups[0]['lr']
        status = {
            'TrLoss': f'{train_loss:.4f}',
            'VaLoss': f'{val_loss:.4f}',
            'Best': f'{best_val_loss:.4f}',
            'LR': f'{current_lr:.2e}'
        }

        if history['val_metrics']:
            vm = history['val_metrics'][-1]
            status.update({
                'MAE': f'{vm["mae"]:.2f}',
                'Cov': f'{vm["coverage_80"]:.0f}%'
            })

        epoch_pbar.set_postfix(status)

        # Early stopping
        if val_loss < best_val_loss - 0.01:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
            if patience_counter >= CONFIG['patience']:
                print(f"\n[OK] Early stopping at epoch {epoch+1}")
                if best_model_state:
                    model.load_state_dict(best_model_state)
                break

    # Store history in model
    history['epochs'] = len(history['train_losses'])
    model.training_history = history

    # Print final summary
    if history['val_metrics']:
        print(f"\n{'='*60}")
        print("TRAINING COMPLETED")
        print(f"{'='*60}")
        final = history['val_metrics'][-1]
        print(f"Final Validation Metrics:")
        print(f"  MAE: {final['mae']:.3f} MVA")
        print(f"  RMSE: {final['rmse']:.3f} MVA")
        print(f"  Coverage 80%: {final['coverage_80']:.1f}%")
        print(f"  CRPS: {final['crps']:.4f}")
        print(f"{'='*60}\n")

    return model


# =============================================================================
# HIGH-LEVEL WRAPPER
# =============================================================================

def train_model_from_dataframe(df_processed, epochs=20, outdir='./outputs', metadata=None):
    """
    Complete training pipeline from DataFrame to trained model.

    This is the main entry point for training. It handles:
    - Dataset creation
    - Model instantiation
    - Training
    - Model saving

    Args:
        df_processed: Preprocessed DataFrame with features and 'split' column
        epochs: Number of training epochs
        outdir: Output directory for model and plots
        metadata: Metadata dict from preprocessing (for denormalization)

    Returns:
        tuple: (trained_model, val_loader, device)
            - trained_model: Trained TemporalFusionTransformer model
            - val_loader: Validation data loader
            - device: Torch device ('cuda' or 'cpu')

    Example:
        >>> df_processed, metadata = preprocess_data(df)
        >>> model, val_loader, device = train_model_from_dataframe(df_processed, epochs=20, metadata=metadata)
    """
    print(f"\n{'='*60}")
    print("INITIALIZING TRAINING PIPELINE")
    print(f"{'='*60}")

    # Step 1: Create datasets and loaders
    print("Creating datasets and dataloaders...")
    train_loader, val_loader, feature_cols = create_datasets_and_loaders(df_processed)
    n_features = len(feature_cols)
    print(f"[OK] Features: {n_features}")
    print(f"[OK] Train batches: {len(train_loader)}")
    print(f"[OK] Val batches: {len(val_loader)}")

    # Step 2: Create model
    print("\nCreating Temporal Fusion Transformer...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = create_tft_model(n_features, feature_cols, device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"[OK] Model created: {total_params:,} parameters")
    print(f"[OK] Device: {device}")

    # Step 3: Train model
    print(f"\n{'='*60}")
    print(f"TRAINING MODEL ({epochs} epochs)")
    print(f"{'='*60}")
    trained_model = train_model(model, train_loader, val_loader, epochs, device, metadata)

    # Step 4: Save model
    print(f"{'='*60}")
    print("SAVING MODEL")
    print(f"{'='*60}")
    save_model_and_history(trained_model, outdir, trained_model.training_history)

    # Step 5: Generate plots (optional)
    try:
        from visualization.training_plots import plot_training_curves
        if hasattr(trained_model, 'training_history'):
            plot_training_curves(trained_model, outdir)
    except Exception as e:
        print(f"[WARNING] Could not generate training plots: {e}")

    print(f"{'='*60}")
    print("TRAINING PIPELINE COMPLETED")
    print(f"{'='*60}\n")

    return trained_model, val_loader, device
