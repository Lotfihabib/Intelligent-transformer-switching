"""
Forecasting Module

This module handles probabilistic load forecasting and trajectory sampling for the
stochastic MPC system. It converts quantile predictions into scenario samples.

Key Functions:
- sample_trajectories_from_quantiles: Sample load trajectories from quantile forecasts
- generate_forecast_samples: DEPRECATED - synthetic forecasts no longer supported

Sampling Method:
1. Extract quantiles (q10, q50, q90) from model predictions
2. Estimate standard deviation: σ = (q90 - q10) / 2.56
3. Sample M trajectories from N(q50, σ)
4. Enforce bounds and non-negativity constraints
"""

import numpy as np

def sample_trajectories_from_quantiles(quantile_predictions, M):
    """
    Sample M trajectories from quantile predictions
    Args:
        quantile_predictions: Array [horizon, num_quantiles] - Quantile predictions
        M: Number of trajectories to sample
        quantiles: List of quantile levels (e.g., [0.1, 0.5, 0.9])
    Returns:
        trajectories: Array [M, horizon] - Sampled load trajectories
    """
    #horizon, num_quantiles = quantile_predictions.shape

    # Extract quantiles
    q10 = quantile_predictions[:, 0]  # Lower bound (10th percentile)
    q50 = quantile_predictions[:, 1]  # Median (50th percentile)
    q90 = quantile_predictions[:, 2]  # Upper bound (90th percentile)

    # Estimate standard deviation from quantiles
    # For normal distribution: q90 - q10 ≈ 2.56 * sigma
    sigma = (q90 - q10) / 2.56
    sigma = np.maximum(sigma, 0.1)  # Minimum uncertainty

    # Sample M trajectories
    trajectories = []
    for _ in range(M):
        # Sample from normal distribution for each timestep
        trajectory = np.random.normal(q50, sigma)

        # Clamp to ensure within quantile bounds (with some tolerance)
        trajectory = np.clip(trajectory, q10, q90)

        # Ensure non-negative loads
        trajectory = np.maximum(trajectory, 0)
        trajectories.append(trajectory)

    trajectories = np.array(trajectories)  # [M, horizon]
    return trajectories

