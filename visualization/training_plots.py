"""
Training Visualization Module
==============================

This module contains functions for visualizing model training metrics and predictions.

Functions:
    plot_training_curves: Plot training and validation loss curves with comprehensive metrics
    plot_predictions: Plot model predictions vs actual values with uncertainty bands
    plot_multistep_predictions: Plot multistep forecast examples showing full prediction horizons
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from pathlib import Path

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: PyTorch not available, some visualization features disabled")

from config import CONFIG


def plot_training_curves(model, outdir):
    """Plot training and validation curves with comprehensive metrics"""

    # Ensure output directory exists
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    if not hasattr(model, 'training_history'):
        print("No training history found in model")
        return

    history = model.training_history
    epochs = range(1, len(history['train_losses']) + 1)

    # Create subplots for different metrics
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

    # Plot 1: Loss curves
    ax1.plot(epochs, history['train_losses'], 'b-', label='Training Loss', linewidth=2)
    ax1.plot(epochs, history['val_losses'], 'r-', label='Validation Loss', linewidth=2)
    ax1.set_title(f'{getattr(model, "model_type", "Model")} - Pinball Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Pinball Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Add annotations for best validation loss
    best_epoch = np.argmin(history['val_losses']) + 1
    best_val_loss = min(history['val_losses'])
    ax1.annotate(f'Best: {best_val_loss:.4f}\nEpoch: {best_epoch}',
                xy=(best_epoch, best_val_loss),
                xytext=(best_epoch + len(epochs)*0.1, best_val_loss + max(history['val_losses'])*0.1),
                arrowprops=dict(arrowstyle='->', color='red', alpha=0.7),
                bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))

    # Plot additional metrics if available
    if 'val_metrics' in history and history['val_metrics']:
        # Plot 2: MAE
        val_mae = [m['mae'] for m in history['val_metrics']]
        train_mae = [m['mae'] for m in history['train_metrics']] if 'train_metrics' in history else []
        if train_mae:
            ax2.plot(epochs, train_mae, 'b-', label='Training MAE', linewidth=2)
        ax2.plot(epochs, val_mae, 'r-', label='Validation MAE', linewidth=2)
        ax2.set_title('Mean Absolute Error (MAE)')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('MAE (MVA)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Plot 3: RMSE
        val_rmse = [m['rmse'] for m in history['val_metrics']]
        train_rmse = [m['rmse'] for m in history['train_metrics']] if 'train_metrics' in history else []
        if train_rmse:
            ax3.plot(epochs, train_rmse, 'b-', label='Training RMSE', linewidth=2)
        ax3.plot(epochs, val_rmse, 'r-', label='Validation RMSE', linewidth=2)
        ax3.set_title('Root Mean Squared Error (RMSE)')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('RMSE (MVA)')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # Plot 4: MAPE
        val_mape = [m['mape'] for m in history['val_metrics']]
        train_mape = [m['mape'] for m in history['train_metrics']] if 'train_metrics' in history else []
        if train_mape:
            ax4.plot(epochs, train_mape, 'b-', label='Training MAPE', linewidth=2)
        ax4.plot(epochs, val_mape, 'r-', label='Validation MAPE', linewidth=2)
        ax4.set_title('Mean Absolute Percentage Error (MAPE)')
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('MAPE (%)')
        ax4.legend()
        ax4.grid(True, alpha=0.3)

    else:

        # Hide unused subplots if no metrics available
        ax2.axis('off')
        ax3.axis('off')
        ax4.axis('off')

    plt.tight_layout()
    plt.savefig(outdir / 'training_curves.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Training curves saved to: {outdir / 'training_curves.png'}")
    print(f"Final training loss: {history['train_losses'][-1]:.4f}")
    print(f"Final validation loss: {history['val_losses'][-1]:.4f}")

    # Create second figure for probabilistic metrics
    if 'val_metrics' in history and history['val_metrics']:
        fig2, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

        # Plot 1: Coverage (80% interval)
        val_coverage_80 = [m['coverage_80'] for m in history['val_metrics']]
        ax1.plot(epochs, val_coverage_80, 'g-', label='80% Coverage', linewidth=2)
        ax1.axhline(y=80, color='r', linestyle='--', label='Target (80%)', alpha=0.7)
        ax1.fill_between(epochs, 75, 85, color='green', alpha=0.1, label='±5% tolerance')
        ax1.set_title('Prediction Interval Coverage (80%)')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Coverage (%)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim([50, 100])

        # Plot 2: Sharpness (Interval Width)
        val_sharpness = [m['sharpness'] for m in history['val_metrics']]
        ax2.plot(epochs, val_sharpness, 'm-', label='Sharpness', linewidth=2)
        ax2.set_title('Prediction Interval Sharpness')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Interval Width (MVA)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Plot 3: CRPS (Continuous Ranked Probability Score)
        val_crps = [m['crps'] for m in history['val_metrics']]
        ax3.plot(epochs, val_crps, 'c-', label='CRPS', linewidth=2)
        ax3.set_title('Continuous Ranked Probability Score (CRPS)')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('CRPS')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # Plot 4: Per-Quantile Coverage
        val_cov_q10 = [m['coverage_q10'] for m in history['val_metrics']]
        val_cov_q50 = [m['coverage_q50'] for m in history['val_metrics']]
        val_cov_q90 = [m['coverage_q90'] for m in history['val_metrics']]
        ax4.plot(epochs, val_cov_q10, 'b-', label='Q10 Coverage', linewidth=2)
        ax4.plot(epochs, val_cov_q50, 'g-', label='Q50 Coverage', linewidth=2)
        ax4.plot(epochs, val_cov_q90, 'r-', label='Q90 Coverage', linewidth=2)
        ax4.axhline(y=10, color='b', linestyle='--', alpha=0.5)
        ax4.axhline(y=50, color='g', linestyle='--', alpha=0.5)
        ax4.axhline(y=90, color='r', linestyle='--', alpha=0.5)
        ax4.set_title('Per-Quantile Coverage')
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('Coverage (%)')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        ax4.set_ylim([0, 100])

        plt.tight_layout()
        plt.savefig(outdir / 'probabilistic_metrics.png', dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Probabilistic metrics curves saved to: {outdir / 'probabilistic_metrics.png'}")

    print(f"Best validation loss: {best_val_loss:.4f} at epoch {best_epoch}")
    return history


def plot_predictions(model, val_loader, device, outdir, quantiles=[0.1, 0.5, 0.9], metadata=None):
    """
    Plot model predictions vs actual values with uncertainty bands
    Creates comprehensive visualization of:
    - Actual vs Predicted scatter plot
    - Time series with prediction intervals
    - Residual analysis
    - Quantile calibration plot
    """

    if not TORCH_AVAILABLE:
        print("Warning: PyTorch not available, skipping prediction plots")
        return

    # Ensure output directory exists
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Extract normalization stats for denormalization
    target_mean = 0.0
    target_std = 1.0
    if metadata is not None and 'normalization' in metadata:
        target_mean = metadata['normalization'].get('target_mean', 0.0)
        target_std = metadata['normalization'].get('target_std', 1.0)

    print("Generating prediction plots...")

    model.eval()
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for features, hist_target, targets in val_loader:
            features, hist_target, targets = features.to(device), hist_target.to(device), targets.to(device)
            predictions = model(features, hist_target)
            all_predictions.append(predictions.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

    # Concatenate all batches
    all_predictions = np.concatenate(all_predictions, axis=0)  # [N, horizon, num_quantiles]
    all_targets = np.concatenate(all_targets, axis=0)  # [N, horizon]

    # IMPORTANT: Use only one-step-ahead predictions to avoid overlapping timesteps
    # This gives us N independent predictions (one per sequence)
    # Using all horizons would create artifacts since consecutive sequences overlap

    pred_q10 = all_predictions[:, 0, 0]  # [N] - first timestep of each sequence
    pred_q50 = all_predictions[:, 0, 1]  # [N]
    pred_q90 = all_predictions[:, 0, 2]  # [N]
    actual = all_targets[:, 0]  # [N]

    # ============================================================
    # CRITICAL: DENORMALIZE predictions and targets back to original scale (MVA)
    # ============================================================

    pred_q10 = pred_q10 * target_std + target_mean
    pred_q50 = pred_q50 * target_std + target_mean
    pred_q90 = pred_q90 * target_std + target_mean
    actual = actual * target_std + target_mean

    # For time series plot with uncertainty, we can show a sliding window
    # Use every sequence's full horizon but limit to first 500 for visualization

    n_sequences_to_plot = min(500, len(all_predictions))
    plot_indices = np.linspace(0, len(all_predictions)-1, n_sequences_to_plot, dtype=int)

    # Create figure with 6 subplots (3x2)
    fig = plt.figure(figsize=(18, 15))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

    # ============ Plot 1: Actual vs Predicted (Scatter) ============
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.scatter(actual, pred_q50, alpha=0.3, s=10, c='blue', label='Predictions')

    # Perfect prediction line
    min_val = min(actual.min(), pred_q50.min())
    max_val = max(actual.max(), pred_q50.max())
    ax1.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')

    # Calculate R²
    ss_res = np.sum((actual - pred_q50) ** 2)
    ss_tot = np.sum((actual - np.mean(actual)) ** 2)
    r2 = 1 - (ss_res / ss_tot)

    ax1.set_xlabel('Actual Load (MVA)', fontsize=12)
    ax1.set_ylabel('Predicted Load (MVA)', fontsize=12)
    ax1.set_title(f'Actual vs Predicted (R² = {r2:.4f})', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # ============ Plot 2: Time Series with Uncertainty Bands ============
    ax2 = fig.add_subplot(gs[0, 1])

    # Show one-step-ahead predictions over time (500 sequences)
    n_samples = min(500, len(actual))
    timesteps = np.arange(n_samples)
    ax2.plot(timesteps, actual[:n_samples], 'k-', linewidth=1.5, label='Actual', alpha=0.7)
    ax2.plot(timesteps, pred_q50[:n_samples], 'b-', linewidth=1.5, label='Predicted (q50, 1-step ahead)')
    ax2.fill_between(timesteps, pred_q10[:n_samples], pred_q90[:n_samples], alpha=0.3, color='blue', label='80% Prediction Interval')
    ax2.set_xlabel('Sequence Number', fontsize=12)
    ax2.set_ylabel('Load (MVA)', fontsize=12)
    ax2.set_title('One-Step-Ahead Predictions with Uncertainty', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    '''ax2.text(0.02, 0.98, 'Note: Each point is an independent 1-step forecast',
             transform=ax2.transAxes, fontsize=9, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))'''

    # ============ Plot 3: Residual Plot ============
    ax3 = fig.add_subplot(gs[1, 0])
    residuals = actual - pred_q50
    ax3.scatter(pred_q50, residuals, alpha=0.3, s=10, c='green')
    ax3.axhline(y=0, color='r', linestyle='--', linewidth=2)
    ax3.axhline(y=residuals.std(), color='orange', linestyle=':', linewidth=1, label=f'±1 std ({residuals.std():.2f})')
    ax3.axhline(y=-residuals.std(), color='orange', linestyle=':', linewidth=1)
    ax3.set_xlabel('Predicted Load (MVA)', fontsize=12)
    ax3.set_ylabel('Residual (Actual - Predicted)', fontsize=12)
    ax3.set_title(f'Residual Plot (Mean = {residuals.mean():.4f})', fontsize=14, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # ============ Plot 4: Residual Histogram ============
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.hist(residuals, bins=50, density=True, alpha=0.7, color='purple', edgecolor='black')

    # Overlay normal distribution
    mu, sigma = residuals.mean(), residuals.std()
    x = np.linspace(residuals.min(), residuals.max(), 100)
    ax4.plot(x, stats.norm.pdf(x, mu, sigma), 'r-', linewidth=2, label=f'Normal(μ={mu:.2f}, σ={sigma:.2f})')
    ax4.set_xlabel('Residual (MVA)', fontsize=12)
    ax4.set_ylabel('Density', fontsize=12)
    ax4.set_title('Residual Distribution', fontsize=14, fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # ============ Plot 5: Prediction Interval Coverage ============
    ax5 = fig.add_subplot(gs[2, 0])

    # Check coverage at different thresholds
    coverage_levels = []
    percentiles = np.linspace(0.1, 0.9, 9)
    for p in percentiles:
        # For simplicity, approximate using quantile predictions
        if p <= 0.5:
            lower = pred_q10 + (pred_q50 - pred_q10) * (p / 0.5)
            upper = pred_q90 - (pred_q90 - pred_q50) * (p / 0.5)
        else:
            lower = pred_q10
            upper = pred_q50 + (pred_q90 - pred_q50) * ((p - 0.5) / 0.4)

        coverage = np.mean((actual >= lower) & (actual <= upper)) * 100
        coverage_levels.append(coverage)

    expected_coverage = percentiles * 100

    ax5.plot(expected_coverage, coverage_levels, 'bo-', linewidth=2, markersize=8, label='Actual Coverage')
    ax5.plot(expected_coverage, expected_coverage, 'r--', linewidth=2, label='Perfect Calibration')
    ax5.fill_between(expected_coverage, expected_coverage - 5, expected_coverage + 5, alpha=0.2, color='red', label='±5% tolerance')
    ax5.set_xlabel('Expected Coverage (%)', fontsize=12)
    ax5.set_ylabel('Actual Coverage (%)', fontsize=12)
    ax5.set_title('Calibration Plot', fontsize=14, fontweight='bold')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    ax5.set_xlim([0, 100])
    ax5.set_ylim([0, 100])

    # ============ Plot 6: Error Metrics Summary ============
    ax6 = fig.add_subplot(gs[2, 1])
    ax6.axis('off')

    # Calculate metrics
    mae = np.mean(np.abs(residuals))
    rmse = np.sqrt(np.mean(residuals ** 2))
    mape = np.mean(np.abs(residuals[actual != 0] / actual[actual != 0])) * 100
    coverage_80 = np.mean((actual >= pred_q10) & (actual <= pred_q90)) * 100
    sharpness = np.mean(pred_q90 - pred_q10)

    # Create text summary
    summary_text = f"""

    VALIDATION SET METRICS
    ═══════════════════════════════════
    Point Forecast Metrics:
    ─────────────────────────────────
    MAE:           {mae:.4f} MVA
    RMSE:          {rmse:.4f} MVA
    MAPE:          {mape:.2f}%
    R² Score:      {r2:.4f}
    Mean Error:    {mu:.4f} MVA
    Std Error:     {sigma:.4f} MVA
    Probabilistic Metrics:
    ─────────────────────────────────
    Coverage 80%:  {coverage_80:.2f}% (target: 80%)
    Sharpness:     {sharpness:.4f} MVA
    Dataset Statistics:
    ─────────────────────────────────
    Samples:       {len(actual):,}
    Mean Load:     {actual.mean():.2f} MVA
    Load Range:    [{actual.min():.2f}, {actual.max():.2f}] MVA
    """

    ax6.text(0.1, 0.5, summary_text, fontsize=11, family='monospace',
             verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.suptitle('Model Prediction Analysis - Validation Set', fontsize=16, fontweight='bold', y=0.995)
    plt.savefig(outdir / 'prediction_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Prediction analysis saved to: {outdir / 'prediction_analysis.png'}")


def plot_multistep_predictions(model, val_loader, device, outdir, quantiles=[0.1, 0.5, 0.9], metadata=None, num_examples=5):
    """
    Plot multistep forecast examples showing full prediction horizons
    Creates visualization with multiple subplots, each showing:
    - Full forecast horizon (all 12 timesteps)
    - Actual values vs Median (P50) predictions
    - 80% prediction interval (P10-P90) shaded area
    - Vertical reference lines at key horizons
    Args:
        model: Trained model
        val_loader: Validation data loader
        device: Torch device
        outdir: Output directory
        quantiles: Quantile levels (default: [0.1, 0.5, 0.9])
        metadata: Metadata dict with normalization stats
        num_examples: Number of example sequences to plot (default: 5)
    """

    if not TORCH_AVAILABLE:
        print("Warning: PyTorch not available, skipping multistep prediction plots")
        return

    # Ensure output directory exists
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Extract normalization stats for denormalization
    target_mean = 0.0
    target_std = 1.0
    if metadata is not None and 'normalization' in metadata:
        target_mean = metadata['normalization'].get('target_mean', 0.0)
        target_std = metadata['normalization'].get('target_std', 1.0)

    print("Generating multistep prediction plots...")

    model.eval()
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for features, hist_target, targets in val_loader:
            features, hist_target, targets = features.to(device), hist_target.to(device), targets.to(device)
            predictions = model(features, hist_target)
            all_predictions.append(predictions.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

    # Concatenate all batches
    all_predictions = np.concatenate(all_predictions, axis=0)  # [N, horizon, num_quantiles]
    all_targets = np.concatenate(all_targets, axis=0)  # [N, horizon]

    # Denormalize predictions and targets
    all_predictions = all_predictions * target_std + target_mean
    all_targets = all_targets * target_std + target_mean

    # Select random sequences to plot
    n_sequences = all_predictions.shape[0]
    horizon = all_predictions.shape[1]

    # Select evenly spaced examples across the validation set
    indices = np.linspace(0, n_sequences - 1, num_examples, dtype=int)

    # Create figure with subplots
    fig, axes = plt.subplots(num_examples, 1, figsize=(12, 3 * num_examples))
    if num_examples == 1:
        axes = [axes]

    for i, idx in enumerate(indices):
        ax = axes[i]

        # Extract data for this sequence
        actual = all_targets[idx, :]  # [horizon]
        pred_q10 = all_predictions[idx, :, 0]  # [horizon]
        pred_q50 = all_predictions[idx, :, 1]  # [horizon]
        pred_q90 = all_predictions[idx, :, 2]  # [horizon]

        # X-axis: forecast horizon (0 to horizon-1)
        x = np.arange(horizon)

        # Plot 80% prediction interval (P10-P90)
        ax.fill_between(x, pred_q10, pred_q90, color='blue', alpha=0.25, label='80% PI (P10-P90)')

        # Plot actual values
        ax.plot(x, actual, 'o-', color='black', linewidth=2, markersize=6, label='Actual', zorder=3)

        # Plot median predictions
        ax.plot(x, pred_q50, 's-', color='blue', linewidth=2, markersize=5, label='Median (P50)', zorder=2)

        # Add vertical reference lines at key horizons
        # At 1/4 horizon (3 steps = 30 min)
        ax.axvline(x=horizon // 4, color='orange', linestyle='--', linewidth=1.5, alpha=0.7)

        # At 1/2 horizon (6 steps = 60 min)
        ax.axvline(x=horizon // 2, color='red', linestyle='--', linewidth=1.5, alpha=0.7)

        # Formatting
        ax.set_xlabel('Forecast Horizon (10-min intervals)', fontsize=11)
        ax.set_ylabel('S_TOTAL (kVA)', fontsize=11)
        ax.set_title(f'4-Hour Forecast (Next {horizon} timesteps = {horizon * 10} min)', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', fontsize=9)

        # Set x-axis limits
        ax.set_xlim(-0.5, horizon - 0.5)

    plt.tight_layout()
    plt.savefig(outdir / 'predictions_multistep.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Multistep prediction plot saved to: {outdir / 'predictions_multistep.png'}")


def plot_multistep_evaluation(point_metrics, prob_metrics, calib_data, outdir):
    """
    Generate comprehensive multi-step evaluation plots.

    Creates 4 plots:
        1. multistep_point_metrics.png: MAE/RMSE/MAPE/R² vs horizon
        2. multistep_prob_metrics.png: CRPS/PICP/Sharpness/Pinball vs horizon
        3. reliability_diagram.png: q10/q50/q90 calibration curves
        4. pit_histogram.png: PIT uniformity check

    Args:
        point_metrics: dict from compute_multistep_point_metrics()
        prob_metrics: dict from compute_multistep_probabilistic_metrics()
        calib_data: dict from compute_calibration_data()
        outdir: Output directory path
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Plot 1: Point Forecast Metrics vs Horizon
    _plot_point_metrics_by_horizon(point_metrics, outdir)

    # Plot 2: Probabilistic Metrics vs Horizon
    _plot_probabilistic_metrics_by_horizon(prob_metrics, outdir)

    # Plot 3: Reliability Diagrams
    _plot_reliability_diagrams(calib_data, outdir)

    # Plot 4: PIT Histogram
    _plot_pit_histogram(calib_data, outdir)

    print(f"Multi-step evaluation plots saved to: {outdir}/")


def _plot_point_metrics_by_horizon(point_metrics, outdir):
    """
    Plot point forecast metrics (MAE, RMSE, MAPE, R²) vs horizon.

    4-panel figure showing degradation across forecast lead time.
    """
    if not point_metrics.get('per_horizon'):
        print("  Warning: No per-horizon metrics to plot")
        return

    horizons = sorted(point_metrics['per_horizon'].keys())
    mae_vals = [point_metrics['per_horizon'][h]['mae'] for h in horizons]
    rmse_vals = [point_metrics['per_horizon'][h]['rmse'] for h in horizons]
    mape_vals = [point_metrics['per_horizon'][h]['mape'] for h in horizons]
    r2_vals = [point_metrics['per_horizon'][h]['r2'] for h in horizons]

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

    # Panel 1: MAE vs Horizon
    ax1.plot(horizons, mae_vals, 'o-', color='#2E86AB', linewidth=2, markersize=6)
    ax1.set_title('Mean Absolute Error (MAE) vs Forecast Horizon', fontsize=13, fontweight='bold')
    ax1.set_xlabel('Forecast Horizon (10-min steps)', fontsize=11)
    ax1.set_ylabel('MAE (MVA)', fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(left=0)

    # Panel 2: RMSE vs Horizon
    ax2.plot(horizons, rmse_vals, 'o-', color='#A23B72', linewidth=2, markersize=6)
    ax2.set_title('Root Mean Squared Error (RMSE) vs Forecast Horizon', fontsize=13, fontweight='bold')
    ax2.set_xlabel('Forecast Horizon (10-min steps)', fontsize=11)
    ax2.set_ylabel('RMSE (MVA)', fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(left=0)

    # Panel 3: MAPE vs Horizon
    ax3.plot(horizons, mape_vals, 'o-', color='#F18F01', linewidth=2, markersize=6)
    ax3.set_title('Mean Absolute Percentage Error (MAPE) vs Forecast Horizon', fontsize=13, fontweight='bold')
    ax3.set_xlabel('Forecast Horizon (10-min steps)', fontsize=11)
    ax3.set_ylabel('MAPE (%)', fontsize=11)
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(left=0)

    # Panel 4: R² vs Horizon
    ax4.plot(horizons, r2_vals, 'o-', color='#6A994E', linewidth=2, markersize=6)
    ax4.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax4.set_title('R² Score vs Forecast Horizon', fontsize=13, fontweight='bold')
    ax4.set_xlabel('Forecast Horizon (10-min steps)', fontsize=11)
    ax4.set_ylabel('R² Score', fontsize=11)
    ax4.grid(True, alpha=0.3)
    ax4.set_xlim(left=0)
    ax4.set_ylim(0, 1.05)

    plt.tight_layout()
    plt.savefig(outdir / 'multistep_point_metrics.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  - multistep_point_metrics.png")


def _plot_probabilistic_metrics_by_horizon(prob_metrics, outdir):
    """
    Plot probabilistic metrics (CRPS, PICP, Sharpness, Pinball) vs horizon.

    4-panel figure showing uncertainty quantification quality.
    """
    if not prob_metrics.get('per_horizon'):
        print("  Warning: No per-horizon probabilistic metrics to plot")
        return

    horizons = sorted(prob_metrics['per_horizon'].keys())
    crps_vals = [prob_metrics['per_horizon'][h]['crps'] for h in horizons]
    picp_vals = [prob_metrics['per_horizon'][h]['picp_80'] for h in horizons]
    sharpness_vals = [prob_metrics['per_horizon'][h]['sharpness'] for h in horizons]
    pinball_total_vals = [prob_metrics['per_horizon'][h]['pinball_total'] for h in horizons]

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

    # Panel 1: CRPS vs Horizon
    ax1.plot(horizons, crps_vals, 'o-', color='#2E86AB', linewidth=2, markersize=6)
    ax1.set_title('Continuous Ranked Probability Score (CRPS) vs Horizon', fontsize=13, fontweight='bold')
    ax1.set_xlabel('Forecast Horizon (10-min steps)', fontsize=11)
    ax1.set_ylabel('CRPS (MVA)', fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(left=0)

    # Panel 2: PICP vs Horizon
    ax2.plot(horizons, picp_vals, 'o-', color='#A23B72', linewidth=2, markersize=6)
    ax2.axhline(y=80, color='gray', linestyle='--', linewidth=1, alpha=0.7, label='Nominal 80%')
    ax2.fill_between(horizons, 75, 85, color='green', alpha=0.1, label='Well-calibrated range')
    ax2.set_title('Prediction Interval Coverage Probability (PICP) vs Horizon', fontsize=13, fontweight='bold')
    ax2.set_xlabel('Forecast Horizon (10-min steps)', fontsize=11)
    ax2.set_ylabel('PICP (%)', fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='best', fontsize=9)
    ax2.set_xlim(left=0)
    ax2.set_ylim(60, 100)

    # Panel 3: Sharpness vs Horizon
    ax3.plot(horizons, sharpness_vals, 'o-', color='#F18F01', linewidth=2, markersize=6)
    ax3.set_title('Prediction Interval Sharpness (Width) vs Horizon', fontsize=13, fontweight='bold')
    ax3.set_xlabel('Forecast Horizon (10-min steps)', fontsize=11)
    ax3.set_ylabel('Mean Interval Width (MVA)', fontsize=11)
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(left=0)

    # Panel 4: Total Pinball Loss vs Horizon
    ax4.plot(horizons, pinball_total_vals, 'o-', color='#6A994E', linewidth=2, markersize=6)
    ax4.set_title('Total Pinball Loss vs Horizon', fontsize=13, fontweight='bold')
    ax4.set_xlabel('Forecast Horizon (10-min steps)', fontsize=11)
    ax4.set_ylabel('Total Pinball Loss', fontsize=11)
    ax4.grid(True, alpha=0.3)
    ax4.set_xlim(left=0)

    plt.tight_layout()
    plt.savefig(outdir / 'multistep_prob_metrics.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  - multistep_prob_metrics.png")


def _plot_reliability_diagrams(calib_data, outdir):
    """
    Plot reliability diagrams for q10, q50, q90 quantiles.

    3-panel figure showing calibration quality (expected vs observed coverage).
    """
    if 'reliability_diagram' not in calib_data or not calib_data['reliability_diagram']:
        print("  Warning: No reliability diagram data to plot")
        return

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
    axes = [ax1, ax2, ax3]
    quantile_keys = ['q10', 'q50', 'q90']
    titles = ['10th Percentile (q10)', '50th Percentile (q50)', '90th Percentile (q90)']
    colors = ['#2E86AB', '#F18F01', '#A23B72']

    for ax, q_key, title, color in zip(axes, quantile_keys, titles, colors):
        if q_key not in calib_data['reliability_diagram']:
            continue

        data = calib_data['reliability_diagram'][q_key]
        expected = data['expected']
        observed = data['observed']

        # Scatter plot: Expected vs Observed coverage
        ax.scatter(expected, observed, s=80, alpha=0.7, color=color, edgecolors='black', linewidth=1)

        # Perfect calibration line (diagonal)
        ax.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Perfect Calibration', alpha=0.5)

        # ±5% calibration bands
        ax.fill_between([0, 1], [0, 1], [0.05, 1.05], color='green', alpha=0.1, label='±5% band')
        ax.fill_between([0, 1], [-0.05, 0.95], [0, 1], color='green', alpha=0.1)

        ax.set_title(f'Reliability Diagram: {title}', fontsize=13, fontweight='bold')
        ax.set_xlabel('Expected Coverage', fontsize=11)
        ax.set_ylabel('Observed Coverage', fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', fontsize=9)
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.set_aspect('equal')

    plt.tight_layout()
    plt.savefig(outdir / 'reliability_diagram.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  - reliability_diagram.png")


def _plot_pit_histogram(calib_data, outdir):
    """
    Plot PIT (Probability Integral Transform) histogram.

    Well-calibrated forecasts should produce uniform PIT values on [0, 1].
    """
    if 'pit_histogram' not in calib_data or not calib_data['pit_histogram']:
        print("  Warning: No PIT histogram data to plot")
        return

    pit_data = calib_data['pit_histogram']
    hist = pit_data.get('hist', [])
    bins = pit_data.get('bins', [])
    chi_square_stat = pit_data.get('chi_square_stat', np.nan)
    p_value = pit_data.get('p_value', np.nan)

    if len(hist) == 0 or len(bins) == 0:
        print("  Warning: Empty PIT histogram data")
        return

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    # Plot histogram
    bin_centers = (bins[:-1] + bins[1:]) / 2
    bar_width = bins[1] - bins[0]
    ax.bar(bin_centers, hist, width=bar_width * 0.9, color='#2E86AB', alpha=0.7,
           edgecolor='black', linewidth=1, label='PIT Histogram')

    # Uniform reference line
    uniform_height = len(pit_data.get('values', [])) / len(hist)
    ax.axhline(y=uniform_height, color='red', linestyle='--', linewidth=2,
               label=f'Uniform Reference (count={uniform_height:.0f})')

    # Annotations
    ax.set_title('Probability Integral Transform (PIT) Histogram', fontsize=14, fontweight='bold')
    ax.set_xlabel('PIT Value', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.grid(True, alpha=0.3, axis='y')
    ax.legend(loc='upper right', fontsize=10)

    # Add statistical test results
    textstr = f'Chi-Square Test:\n  χ² = {chi_square_stat:.2f}\n  p-value = {p_value:.4f}'
    if p_value > 0.05:
        textstr += '\n  ✓ Well-calibrated'
        text_color = 'green'
    else:
        textstr += '\n  ✗ Miscalibrated'
        text_color = 'red'

    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white',
            alpha=0.8, edgecolor=text_color, linewidth=2))

    ax.set_xlim(0, 1)

    plt.tight_layout()
    plt.savefig(outdir / 'pit_histogram.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  - pit_histogram.png")


# ============================================================================
# ADVANCED CALIBRATION VISUALIZATIONS (Phase 3)
# ============================================================================

def plot_per_horizon_calibration(calibration_data, outdir):
    """
    Plot reliability diagrams for multiple forecast horizons.

    Shows calibration curves for each quantile at different forecast horizons
    to reveal how calibration degrades (or improves) with forecast lead time.

    Args:
        calibration_data: Dict from compute_reliability_per_horizon()
        outdir: Output directory path
    """
    if not calibration_data:
        print("  Warning: No per-horizon calibration data to plot")
        return

    outdir = Path(outdir)

    # Extract horizon keys
    horizon_keys = sorted([k for k in calibration_data.keys() if k.startswith('horizon_')],
                          key=lambda x: int(x.split('_')[1]))

    if len(horizon_keys) == 0:
        print("  Warning: No horizon data found")
        return

    # Number of horizons to plot
    num_horizons = len(horizon_keys)

    # Create subplots (2x2 grid for 4 horizons)
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()

    colors = {'q10': '#E63946', 'q50': '#457B9D', 'q90': '#2A9D8F'}

    for idx, horizon_key in enumerate(horizon_keys[:4]):  # Plot up to 4 horizons
        ax = axes[idx]
        horizon_num = int(horizon_key.split('_')[1])
        horizon_data = calibration_data[horizon_key]

        # Perfect calibration line
        ax.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Perfect Calibration', alpha=0.7)

        # Plot each quantile's reliability curve
        quantiles_data = horizon_data.get('quantiles', {})

        for q_name in ['q10', 'q50', 'q90']:
            if q_name not in quantiles_data:
                continue

            q_data = quantiles_data[q_name]
            nominal = q_data['nominal_coverage']
            observed = q_data['observed_coverage']

            # Plot as a single point on the reliability diagram
            ax.scatter([nominal], [observed], s=200, color=colors[q_name],
                      label=f'{q_name} (obs={observed:.3f})', marker='o',
                      edgecolors='black', linewidth=2, zorder=5)

            # Add error bar showing deviation from perfect calibration
            ax.plot([nominal, nominal], [nominal, observed], color=colors[q_name],
                   linewidth=2, alpha=0.5, linestyle=':')

        # Styling
        ax.set_title(f'Horizon {horizon_num} (t+{horizon_num * 10} min)',
                    fontsize=12, fontweight='bold')
        ax.set_xlabel('Nominal Coverage', fontsize=10)
        ax.set_ylabel('Observed Coverage', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper left', fontsize=8)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect('equal')

        # Add calibration error annotation
        mean_error = horizon_data.get('mean_calibration_error', np.nan)
        ax.text(0.98, 0.02, f'Mean Cal. Error: {mean_error:.4f}',
               transform=ax.transAxes, fontsize=9,
               verticalalignment='bottom', horizontalalignment='right',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.suptitle('Per-Horizon Reliability Diagrams', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(outdir / 'per_horizon_calibration.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  - per_horizon_calibration.png")


def plot_pit_by_horizon(pit_data, outdir):
    """
    Plot PIT histograms for multiple forecast horizons.

    Shows how probabilistic forecast calibration varies across forecast horizons.
    Includes chi-square test results for uniformity testing.

    Args:
        pit_data: Dict from compute_pit_per_horizon()
        outdir: Output directory path
    """
    if not pit_data:
        print("  Warning: No per-horizon PIT data to plot")
        return

    outdir = Path(outdir)

    # Extract horizon keys
    horizon_keys = sorted([k for k in pit_data.keys() if k.startswith('horizon_')],
                          key=lambda x: int(x.split('_')[1]))

    if len(horizon_keys) == 0:
        print("  Warning: No horizon data found")
        return

    # Create subplots (2x2 grid)
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()

    for idx, horizon_key in enumerate(horizon_keys[:4]):  # Up to 4 horizons
        ax = axes[idx]
        horizon_num = int(horizon_key.split('_')[1])
        horizon_data = pit_data[horizon_key]

        pit_values = np.array(horizon_data.get('pit_values', []))
        chi2_stat = horizon_data.get('chi2_statistic', np.nan)
        chi2_pval = horizon_data.get('chi2_pvalue', np.nan)
        ks_stat = horizon_data.get('ks_statistic', np.nan)
        ks_pval = horizon_data.get('ks_pvalue', np.nan)

        if len(pit_values) == 0:
            continue

        # Plot histogram
        num_bins = 10
        counts, bins, patches = ax.hist(pit_values, bins=num_bins, range=(0, 1),
                                        color='#457B9D', alpha=0.7, edgecolor='black',
                                        linewidth=1.5)

        # Uniform reference line
        uniform_height = len(pit_values) / num_bins
        ax.axhline(y=uniform_height, color='red', linestyle='--', linewidth=2,
                  label=f'Uniform Ref ({uniform_height:.0f})')

        # Styling
        ax.set_title(f'Horizon {horizon_num} (t+{horizon_num * 10} min)',
                    fontsize=12, fontweight='bold')
        ax.set_xlabel('PIT Value', fontsize=10)
        ax.set_ylabel('Frequency', fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')
        ax.legend(loc='upper right', fontsize=8)
        ax.set_xlim(0, 1)

        # Add test statistics
        is_calibrated = chi2_pval > 0.05
        status_text = '✓ Well-calibrated' if is_calibrated else '✗ Miscalibrated'
        text_color = 'green' if is_calibrated else 'red'

        textstr = (f'Chi-Square: χ²={chi2_stat:.2f}, p={chi2_pval:.4f}\n'
                  f'KS Test: D={ks_stat:.4f}, p={ks_pval:.4f}\n'
                  f'{status_text}')

        ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=8,
               verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.9,
                        edgecolor=text_color, linewidth=2))

    plt.suptitle('PIT Histograms by Forecast Horizon', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(outdir / 'pit_by_horizon.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  - pit_by_horizon.png")


def plot_calibration_heatmap(heatmap_data, outdir):
    """
    Plot 2D heatmap of calibration errors across horizons and quantiles.

    Provides a comprehensive overview of where forecast calibration breaks down.

    Args:
        heatmap_data: Dict from compute_calibration_heatmap_data()
        outdir: Output directory path
    """
    if not heatmap_data:
        print("  Warning: No heatmap data to plot")
        return

    outdir = Path(outdir)

    calibration_errors = heatmap_data.get('calibration_errors')
    observed_coverages = heatmap_data.get('observed_coverages')
    horizons = heatmap_data.get('horizons', [])
    quantiles = heatmap_data.get('quantiles', [])

    if calibration_errors is None or len(horizons) == 0 or len(quantiles) == 0:
        print("  Warning: Incomplete heatmap data")
        return

    # Create figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Subplot 1: Calibration Error Heatmap
    im1 = ax1.imshow(calibration_errors, cmap='YlOrRd', aspect='auto', vmin=0, vmax=0.1)
    ax1.set_title('Calibration Error by Horizon × Quantile', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Quantile Level', fontsize=12)
    ax1.set_ylabel('Forecast Horizon', fontsize=12)

    # Set ticks
    ax1.set_xticks(range(len(quantiles)))
    ax1.set_xticklabels([f'q{int(q*100)}' for q in quantiles])
    ax1.set_yticks(range(len(horizons)))
    ax1.set_yticklabels([f'H{h}' for h in horizons])

    # Add colorbar
    cbar1 = plt.colorbar(im1, ax=ax1)
    cbar1.set_label('Calibration Error', fontsize=11)

    # Annotate cells with values
    for i in range(len(horizons)):
        for j in range(len(quantiles)):
            text = ax1.text(j, i, f'{calibration_errors[i, j]:.3f}',
                          ha='center', va='center', color='black', fontsize=9)

    # Subplot 2: Observed Coverage Heatmap
    im2 = ax2.imshow(observed_coverages, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
    ax2.set_title('Observed Coverage by Horizon × Quantile', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Quantile Level', fontsize=12)
    ax2.set_ylabel('Forecast Horizon', fontsize=12)

    # Set ticks
    ax2.set_xticks(range(len(quantiles)))
    ax2.set_xticklabels([f'q{int(q*100)}' for q in quantiles])
    ax2.set_yticks(range(len(horizons)))
    ax2.set_yticklabels([f'H{h}' for h in horizons])

    # Add colorbar
    cbar2 = plt.colorbar(im2, ax=ax2)
    cbar2.set_label('Observed Coverage', fontsize=11)

    # Annotate cells with values and add reference lines for nominal coverage
    for i in range(len(horizons)):
        for j in range(len(quantiles)):
            # Color text based on deviation from nominal
            deviation = abs(observed_coverages[i, j] - quantiles[j])
            text_color = 'black' if deviation < 0.05 else 'red'

            text = ax2.text(j, i, f'{observed_coverages[i, j]:.3f}',
                          ha='center', va='center', color=text_color, fontsize=9,
                          fontweight='bold' if deviation >= 0.05 else 'normal')

    plt.tight_layout()
    plt.savefig(outdir / 'calibration_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  - calibration_heatmap.png")


def plot_conditional_calibration(conditional_data, outdir):
    """
    Plot calibration stratified by conditions (load regime, peak/offpeak, day type).

    Shows how forecast calibration varies across different operating conditions.

    Args:
        conditional_data: Dict from compute_conditional_calibration()
        outdir: Output directory path
    """
    if not conditional_data or len(conditional_data) == 0:
        print("  Warning: No conditional calibration data to plot")
        return

    outdir = Path(outdir)

    # Create figure with 3 subplots (one for each stratification)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    stratifications = [
        ('by_load_regime', 'Load Regime', ['low', 'medium', 'high']),
        ('by_peak_period', 'Peak Period', ['offpeak', 'peak']),
        ('by_day_type', 'Day Type', ['weekend', 'weekday'])
    ]

    colors_map = {
        'low': '#2A9D8F', 'medium': '#E9C46A', 'high': '#E63946',
        'offpeak': '#457B9D', 'peak': '#E76F51',
        'weekend': '#8338EC', 'weekday': '#06A77D'
    }

    for ax_idx, (strat_key, strat_title, categories) in enumerate(stratifications):
        ax = axes[ax_idx]

        if strat_key not in conditional_data:
            ax.text(0.5, 0.5, f'No data for\n{strat_title}',
                   ha='center', va='center', transform=ax.transAxes)
            continue

        strat_data = conditional_data[strat_key]

        # Plot calibration error across horizons for each category
        for category in categories:
            if category not in strat_data:
                continue

            category_data = strat_data[category]

            # Extract horizons and calibration errors
            horizon_keys = sorted([k for k in category_data.keys() if k.startswith('horizon_')],
                                 key=lambda x: int(x.split('_')[1]))

            horizons = [int(k.split('_')[1]) for k in horizon_keys]
            cal_errors = [category_data[k]['mean_calibration_error'] for k in horizon_keys]

            if len(horizons) > 0:
                ax.plot(horizons, cal_errors, marker='o', linewidth=2, markersize=8,
                       label=category.capitalize(), color=colors_map.get(category, 'gray'))

        # Styling
        ax.set_title(f'Calibration by {strat_title}', fontsize=12, fontweight='bold')
        ax.set_xlabel('Forecast Horizon', fontsize=10)
        ax.set_ylabel('Mean Calibration Error', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', fontsize=9)
        ax.set_ylim(bottom=0)

        # Add reference line at 0.05 (5% error threshold)
        ax.axhline(y=0.05, color='red', linestyle='--', linewidth=1, alpha=0.5,
                  label='5% threshold')

    plt.suptitle('Conditional Calibration Analysis', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(outdir / 'conditional_calibration.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  - conditional_calibration.png")


def plot_interval_scores(interval_data, outdir):
    """
    Plot interval scores (Winkler score and related metrics) across horizons.

    Shows how prediction interval quality evolves with forecast lead time.

    Args:
        interval_data: Dict from compute_interval_scores()
        outdir: Output directory path
    """
    if not interval_data or len(interval_data) == 0:
        print("  Warning: No interval score data to plot")
        return

    outdir = Path(outdir)

    # Extract data
    horizon_keys = sorted([k for k in interval_data.keys() if k.startswith('horizon_')],
                          key=lambda x: int(x.split('_')[1]))

    if len(horizon_keys) == 0:
        print("  Warning: No horizon data found")
        return

    horizons = [int(k.split('_')[1]) for k in horizon_keys]
    winkler_scores = [interval_data[k]['winkler_score'] for k in horizon_keys]
    interval_scores = [interval_data[k]['interval_score'] for k in horizon_keys]
    sharpness = [interval_data[k]['sharpness'] for k in horizon_keys]
    coverage = [interval_data[k]['coverage'] for k in horizon_keys]
    nominal_coverage = interval_data[horizon_keys[0]]['nominal_coverage']

    # Create 2x2 subplot
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Winkler Score
    ax1.plot(horizons, winkler_scores, marker='o', linewidth=2, markersize=8,
            color='#E63946', label='Winkler Score')
    ax1.set_title('Winkler Score by Horizon', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Forecast Horizon', fontsize=10)
    ax1.set_ylabel('Winkler Score (lower is better)', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Plot 2: Interval Score (same as Winkler in this implementation)
    ax2.plot(horizons, interval_scores, marker='s', linewidth=2, markersize=8,
            color='#457B9D', label='Interval Score')
    ax2.set_title('Interval Score by Horizon', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Forecast Horizon', fontsize=10)
    ax2.set_ylabel('Interval Score (lower is better)', fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    # Plot 3: Sharpness (interval width)
    ax3.plot(horizons, sharpness, marker='D', linewidth=2, markersize=8,
            color='#2A9D8F', label='Sharpness (Interval Width)')
    ax3.set_title('Prediction Interval Sharpness', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Forecast Horizon', fontsize=10)
    ax3.set_ylabel('Mean Interval Width (MVA)', fontsize=10)
    ax3.grid(True, alpha=0.3)
    ax3.legend()

    # Plot 4: Coverage
    ax4.plot(horizons, coverage, marker='^', linewidth=2, markersize=8,
            color='#F4A261', label='Observed Coverage')
    ax4.axhline(y=nominal_coverage, color='red', linestyle='--', linewidth=2,
               label=f'Nominal Coverage ({nominal_coverage:.0%})')
    ax4.fill_between(horizons, nominal_coverage - 0.05, nominal_coverage + 0.05,
                     color='green', alpha=0.2, label='±5% tolerance')
    ax4.set_title('Prediction Interval Coverage', fontsize=12, fontweight='bold')
    ax4.set_xlabel('Forecast Horizon', fontsize=10)
    ax4.set_ylabel('Coverage Probability', fontsize=10)
    ax4.grid(True, alpha=0.3)
    ax4.legend()
    ax4.set_ylim(0, 1)

    plt.suptitle('Interval Score Analysis', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(outdir / 'interval_scores.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  - interval_scores.png")


def plot_advanced_calibration_summary(calibration_results, outdir):
    """
    Generate all advanced calibration plots from comprehensive analysis results.

    This is the main entry point for Phase 3 advanced calibration visualizations.

    Args:
        calibration_results: Dict from run_advanced_calibration_analysis()
        outdir: Output directory path
    """
    print("\nGenerating advanced calibration visualizations...")

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Plot per-horizon calibration
    if 'reliability_per_horizon' in calibration_results:
        plot_per_horizon_calibration(calibration_results['reliability_per_horizon'], outdir)

    # Plot per-horizon PIT
    if 'pit_per_horizon' in calibration_results:
        plot_pit_by_horizon(calibration_results['pit_per_horizon'], outdir)

    # Plot calibration heatmap
    if 'heatmap_data' in calibration_results:
        plot_calibration_heatmap(calibration_results['heatmap_data'], outdir)

    # Plot conditional calibration
    if 'conditional_calibration' in calibration_results:
        plot_conditional_calibration(calibration_results['conditional_calibration'], outdir)

    # Plot interval scores
    if 'interval_scores' in calibration_results:
        plot_interval_scores(calibration_results['interval_scores'], outdir)

    print("  [OK] Advanced calibration visualizations complete")
