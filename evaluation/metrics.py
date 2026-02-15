"""
Evaluation Metrics Module

Computes comprehensive evaluation metrics for probabilistic forecasting:
- Point forecast metrics (MAE, RMSE, MAPE, R²)
- Probabilistic metrics (Coverage, Sharpness, CRPS)
- Calibration metrics (Miscalibration, Per-quantile coverage)
"""

import numpy as np

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


def pinball_loss(predictions, targets, quantiles, smoothing=0.1, quantile_weights=None):
    """
    Compute pinball loss (quantile loss) for probabilistic forecasting

    Args:
        predictions: Model predictions [batch, horizon, num_quantiles]
        targets: True values [batch, horizon]
        quantiles: List of quantiles (e.g., [0.1, 0.5, 0.9])
        smoothing: Label smoothing factor (default: 0.1)
        quantile_weights: Optional weights per quantile (default: uniform)

    Returns:
        torch.Tensor: Average pinball loss across all quantiles
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is required for pinball_loss computation")

    # Ensure targets has the right shape
    if targets.dim() == 1:
        targets = targets.unsqueeze(0)
    if targets.dim() == 3:
        targets = targets.squeeze(-1)

    # Apply label smoothing (inject noise for regularization)
    if smoothing > 0 and targets.requires_grad:
        noise = torch.randn_like(targets) * smoothing
        targets_smooth = targets + noise
    else:
        targets_smooth = targets

    batch_size, horizon, num_quantiles = predictions.shape

    # Expand targets to match predictions shape
    targets_expanded = targets_smooth.unsqueeze(-1).expand(-1, -1, num_quantiles)

    # Compute errors
    errors = targets_expanded - predictions

    # Clamp errors to prevent extreme values during training
    errors = torch.clamp(errors, -100, 100)

    # Compute pinball loss for each quantile
    quantile_tensor = torch.tensor(quantiles, device=predictions.device, dtype=predictions.dtype)
    quantile_tensor = quantile_tensor.view(1, 1, num_quantiles)

    # Pinball loss formula: max(q * error, (q - 1) * error)
    loss_per_quantile = torch.where(
        errors >= 0,
        quantile_tensor * errors,
        (quantile_tensor - 1) * errors
    )

    # Apply optional quantile weights (e.g., emphasize median or tails)
    if quantile_weights is not None:
        weights = torch.tensor(quantile_weights, device=predictions.device, dtype=predictions.dtype)
        weights = weights.view(1, 1, num_quantiles)
        loss_per_quantile = loss_per_quantile * weights

    # Average across batch, horizon, and quantiles
    # Simple average performs best (advanced weighting degraded performance)
    loss = loss_per_quantile.mean()

    return loss


def compute_evaluation_metrics(predictions, targets, quantiles=[0.1, 0.5, 0.9]):
    """
    Compute comprehensive evaluation metrics for probabilistic forecasting
    Args:
        predictions: Model predictions [batch, horizon, num_quantiles]
        targets: True values [batch, horizon]
        quantiles: List of quantiles used (e.g., [0.1, 0.5, 0.9])
    Returns:
        dict: Dictionary containing all evaluation metrics including:
            - Point forecast metrics (MAE, RMSE, MAPE, R²)
            - Probabilistic metrics (Coverage, Sharpness, Pinball Loss)
            - Calibration metrics (Miscalibration, Per-quantile coverage)
    """

    # Ensure targets has the right shape
    if targets.dim() == 1:
        targets = targets.unsqueeze(0)
    if targets.dim() == 3:
        targets = targets.squeeze(-1)

    # Use median (q50) predictions for point forecasts
    median_idx = quantiles.index(0.5) if 0.5 in quantiles else 1

    # IMPORTANT: Use only one-step-ahead predictions to avoid overlapping timesteps
    # This prevents artifacts from averaging over duplicate actual values
    point_predictions = predictions[:, 0, median_idx]  # [batch] - first timestep only

    # Convert to numpy for easier computation
    pred_np = point_predictions.detach().cpu().numpy()  # [batch] - no flatten needed
    true_np = targets[:, 0].detach().cpu().numpy()  # [batch] - first timestep only

    # Get all quantile predictions (one-step-ahead only)
    all_quantiles_np = predictions[:, 0, :].detach().cpu().numpy()  # [batch, num_quantiles]

    # Remove any NaN or infinite values
    valid_mask = ~(np.isnan(pred_np) | np.isnan(true_np) |
                   np.isinf(pred_np) | np.isinf(true_np))

    pred_np = pred_np[valid_mask]
    true_np = true_np[valid_mask]
    all_quantiles_flat = all_quantiles_np[valid_mask]  # [N, num_quantiles]

    if len(pred_np) == 0:
        return {
            'mae': float('inf'), 'mse': float('inf'), 'rmse': float('inf'),
            'mape': float('inf'), 'quantile_loss': float('inf'),
            'coverage_80': 0.0, 'coverage_50': 0.0, 'sharpness': float('inf')
        }

    # ==================== Point Forecast Metrics ====================
    errors = pred_np - true_np
    abs_errors = np.abs(errors)
    squared_errors = errors ** 2

    # Mean Absolute Error
    mae = np.mean(abs_errors)

    # Mean Squared Error
    mse = np.mean(squared_errors)

    # Root Mean Squared Error
    rmse = np.sqrt(mse)

    # Mean Absolute Percentage Error (avoid division by zero)
    non_zero_mask = true_np != 0

    if np.any(non_zero_mask):
        mape = np.mean(np.abs(errors[non_zero_mask] / true_np[non_zero_mask])) * 100
    else:
        mape = float('inf')

    # Pinball loss (quantile loss - overall)
    q_loss = pinball_loss(predictions, targets, quantiles).item()

    # Mean Error (bias)
    mean_error = np.mean(errors)

    # R-squared
    ss_res = np.sum(squared_errors)
    ss_tot = np.sum((true_np - np.mean(true_np)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else float('-inf')

    # Normalized metrics
    mean_true = np.mean(true_np)
    normalized_mae = mae / mean_true if mean_true != 0 else float('inf')
    normalized_rmse = rmse / mean_true if mean_true != 0 else float('inf')

    # ==================== Probabilistic Metrics ====================

    # 1. Coverage Metrics (Prediction Interval Coverage)
    # Check if true values fall within prediction intervals
    # Identify lower and upper quantiles for intervals
    lower_idx = 0  # q10
    upper_idx = -1  # q90
    lower_bounds = all_quantiles_flat[:, lower_idx]
    upper_bounds = all_quantiles_flat[:, upper_idx]

    # 80% Prediction Interval Coverage (q10 to q90)
    within_80_interval = (true_np >= lower_bounds) & (true_np <= upper_bounds)
    coverage_80 = np.mean(within_80_interval) * 100  # Should be ~80%

    # If we have middle quantiles, compute 50% interval coverage
    if len(quantiles) >= 3:

        # Approximate 50% interval as q25-q75 using q10 and q90
        # This is an approximation; for exact coverage, we'd need q25 and q75
        q_range = quantiles[-1] - quantiles[0]  # e.g., 0.9 - 0.1 = 0.8
        interval_width = upper_bounds - lower_bounds

        # Approximate narrower interval
        narrower_margin = interval_width * 0.3  # 30% of the 80% interval
        approx_lower_50 = lower_bounds + narrower_margin
        approx_upper_50 = upper_bounds - narrower_margin
        within_50_interval = (true_np >= approx_lower_50) & (true_np <= approx_upper_50)
        coverage_50 = np.mean(within_50_interval) * 100  # Should be ~50%

    else:
        coverage_50 = 0.0

    # 2. Sharpness (Average width of prediction intervals)
    # Narrower intervals are better (sharper), but must maintain calibration
    sharpness = np.mean(upper_bounds - lower_bounds)
    normalized_sharpness = sharpness / mean_true if mean_true != 0 else float('inf')

    # 3. Quantile Score (Pinball Loss per quantile)
    # This measures the quality of each quantile prediction
    quantile_scores = {}
    for i, q in enumerate(quantiles):
        q_pred = all_quantiles_flat[:, i]
        q_errors = true_np - q_pred

        # Pinball loss formula
        pinball = np.where(q_errors >= 0, q * q_errors, (q - 1) * q_errors)
        quantile_scores[f'pinball_q{int(q*100)}'] = float(np.mean(pinball))

    # 4. Miscalibration (deviation from expected coverage)
    # Measures how well-calibrated the probabilistic predictions are
    expected_coverage_80 = 80.0  # Expected coverage for 80% interval
    miscalibration_80 = abs(coverage_80 - expected_coverage_80)

    # 5. Continuous Ranked Probability Score (CRPS)
    # Measures the quality of the entire predictive distribution
    # CRPS for quantile forecasts using piecewise linear approximation
    crps_values = []
    for i in range(len(true_np)):
        y_true = true_np[i]
        quantile_preds = all_quantiles_flat[i, :]

        # Approximate CRPS using quantile predictions
        # CRPS = integral of (F(x) - 1{y <= x})^2 dx
        # For quantile forecasts, we approximate this
        crps_sum = 0
        for j in range(len(quantiles) - 1):
            q1, q2 = quantiles[j], quantiles[j+1]
            p1, p2 = quantile_preds[j], quantile_preds[j+1]

            # Contribution from this segment
            if y_true <= p1:
                crps_sum += (q1)**2 * (p2 - p1)
            elif y_true >= p2:
                crps_sum += (1 - q2)**2 * (p2 - p1)
            else:
                crps_sum += q1**2 * (y_true - p1) + (1 - q2)**2 * (p2 - y_true)
        crps_values.append(crps_sum)
    crps = np.mean(crps_values)

    # 6. Quantile Reliability (per-quantile coverage check)
    # Check if each quantile has the expected proportion of observations below it
    quantile_reliability = {}
    for i, q in enumerate(quantiles):
        q_pred = all_quantiles_flat[:, i]
        below_quantile = np.mean(true_np <= q_pred) * 100
        quantile_reliability[f'coverage_q{int(q*100)}'] = float(below_quantile)

    # ==================== Compile All Metrics ====================
    metrics = {
        # Point forecast metrics
        'mae': float(mae),
        'mse': float(mse),
        'rmse': float(rmse),
        'mape': float(mape),
        'mean_error': float(mean_error),
        'r2_score': float(r2),
        'normalized_mae': float(normalized_mae),
        'normalized_rmse': float(normalized_rmse),
        # Probabilistic metrics
        'quantile_loss': float(q_loss),
        'coverage_80': float(coverage_80),
        'coverage_50': float(coverage_50),
        'miscalibration_80': float(miscalibration_80),
        'sharpness': float(sharpness),
        'normalized_sharpness': float(normalized_sharpness),
        'crps': float(crps),
        # Sample count
        'n_samples': len(pred_np)
    }

    # Add per-quantile pinball scores
    metrics.update(quantile_scores)

    # Add per-quantile reliability
    metrics.update(quantile_reliability)

    return metrics
