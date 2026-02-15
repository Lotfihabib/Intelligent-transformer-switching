"""
Loss Functions Module

Implements pinball loss (quantile loss) for probabilistic forecasting with
optional weighted quantiles for improved median accuracy.
"""

import torch


def pinball_loss(predictions, targets, quantiles, smoothing=0.1, quantile_weights=None):
    """
    Weighted Pinball Loss (Quantile Loss) for probabilistic forecasting.

    The pinball loss is an asymmetric loss function used for quantile regression.
    For a given quantile q ∈ (0, 1), the loss is defined as:
        L_q(y, ŷ) = max(q(y - ŷ), (q - 1)(y - ŷ))

    Where:
        - y = actual target value
        - ŷ = predicted value for quantile q
        - q = quantile level (e.g., 0.1, 0.5, 0.9)

    Properties:
        - Asymmetric: Penalizes over-predictions and under-predictions differently
        - q < 0.5: Penalizes under-predictions more (conservative lower bound)
        - q > 0.5: Penalizes over-predictions more (conservative upper bound)
        - q = 0.5: Reduces to Mean Absolute Error (symmetric)

    v2.0 Enhancement: Supports weighted quantiles to improve median accuracy.
    Example: [1.0, 1.5, 1.0] gives median 1.5× importance.

    Args:
        predictions: Tensor [batch, horizon, num_quantiles] - Model predictions
        targets: Tensor [batch, horizon] - Ground truth values
        quantiles: List of quantile levels (e.g., [0.1, 0.5, 0.9])
        smoothing: Label smoothing factor for regularization (default: 0.1)
        quantile_weights: List of weights for each quantile (e.g., [1.0, 1.5, 1.0])
                         Higher weight = more importance. If None, uses uniform weights.

    Returns:
        Weighted average pinball loss across all quantiles
    """

    losses = []

    # Ensure targets has the right shape - should be [batch, horizon]
    if targets.dim() == 1:
        targets = targets.unsqueeze(0)

    if targets.dim() == 3:
        targets = targets.squeeze(-1)

    # Check for NaN or infinite values
    if torch.isnan(predictions).any() or torch.isinf(predictions).any():
        print("WARNING: NaN or Inf detected in predictions!")
        return torch.tensor(float('inf'), device=predictions.device)

    if torch.isnan(targets).any() or torch.isinf(targets).any():
        print("WARNING: NaN or Inf detected in targets!")
        return torch.tensor(float('inf'), device=targets.device)

    # Add noise for regularization (label smoothing / data augmentation)
    if smoothing > 0:
        noise = torch.randn_like(targets) * smoothing * targets.std()
        targets = targets + noise

    # Compute pinball loss for each quantile (standard formulation)
    for i, q in enumerate(quantiles):
        error = targets - predictions[:, :, i]  # Prediction error: y - ŷ

        # Clamp error to prevent extreme values that could destabilize training
        error = torch.clamp(error, min=-100, max=100)

        # Pinball loss formula: max(q * error, (q - 1) * error)
        loss = torch.maximum(q * error, (q - 1) * error)

        # Compute mean loss for this quantile
        loss_mean = loss.mean()

        # Check for NaN in individual quantile loss
        if torch.isnan(loss_mean) or torch.isinf(loss_mean):
            print(f"WARNING: NaN/Inf in quantile {q} pinball loss!")
            return torch.tensor(float('inf'), device=predictions.device)

        losses.append(loss_mean)

    # Weighted average loss across all quantiles
    if quantile_weights is None:
        # Default: uniform weights (standard averaging)
        total_loss = sum(losses) / len(quantiles)
    else:
        # Weighted average: sum(weight_i * loss_i) / sum(weight_i)
        if len(quantile_weights) != len(quantiles):
            raise ValueError(f"quantile_weights length ({len(quantile_weights)}) must match quantiles length ({len(quantiles)})")

        weighted_sum = sum(w * loss for w, loss in zip(quantile_weights, losses))
        weight_sum = sum(quantile_weights)
        total_loss = weighted_sum / weight_sum

    # Final NaN check
    if torch.isnan(total_loss) or torch.isinf(total_loss):
        print("WARNING: NaN/Inf in final pinball loss!")
        return torch.tensor(float('inf'), device=predictions.device)

    return total_loss
