"""
Advanced Calibration Analysis for Probabilistic Forecasts

This module provides detailed calibration analysis including:
- Per-horizon PIT (Probability Integral Transform) analysis
- Conditional calibration by load regime, time-of-day, day-type
- Interval scores (Winkler score, interval score)
- Calibration error metrics and heatmaps
- Chi-square goodness-of-fit tests for PIT uniformity
"""

import numpy as np
import torch
from scipy import stats
from typing import Dict, List, Optional, Tuple
import pandas as pd


def compute_pit_per_horizon(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    quantiles: List[float] = [0.1, 0.5, 0.9],
    horizons: Optional[List[int]] = None
) -> Dict[str, Dict]:
    """
    Compute Probability Integral Transform (PIT) for each forecast horizon.

    PIT is used to assess calibration: if forecasts are well-calibrated,
    PIT values should be uniformly distributed on [0,1].

    Args:
        predictions: [N, H, Q] tensor of quantile predictions
        targets: [N, H] tensor of actual values
        quantiles: List of quantile levels
        horizons: List of horizons to analyze (default: all)

    Returns:
        Dictionary with per-horizon PIT data and chi-square tests
    """
    N, H, Q = predictions.shape

    if horizons is None:
        horizons = list(range(1, H + 1))

    pit_results = {}

    for h in horizons:
        h_idx = h - 1

        # Get predictions and targets for this horizon
        preds_h = predictions[:, h_idx, :]  # [N, Q]
        targets_h = targets[:, h_idx]       # [N]

        # Compute PIT values
        pit_values = []

        for i in range(N):
            pred_quantiles = preds_h[i].cpu().numpy()  # [Q]
            actual = targets_h[i].item()

            # Find which quantile interval the actual value falls into
            # PIT = interpolated quantile level where actual value would be
            if actual <= pred_quantiles[0]:
                # Below lowest quantile
                pit = quantiles[0] * (actual / pred_quantiles[0]) if pred_quantiles[0] > 0 else 0
            elif actual >= pred_quantiles[-1]:
                # Above highest quantile
                excess_ratio = (actual - pred_quantiles[-1]) / (pred_quantiles[-1] - pred_quantiles[-2]) if pred_quantiles[-1] > pred_quantiles[-2] else 0
                pit = quantiles[-1] + (1 - quantiles[-1]) * min(excess_ratio, 1.0)
            else:
                # Interpolate between quantiles
                for j in range(len(quantiles) - 1):
                    if pred_quantiles[j] <= actual <= pred_quantiles[j + 1]:
                        if pred_quantiles[j + 1] > pred_quantiles[j]:
                            ratio = (actual - pred_quantiles[j]) / (pred_quantiles[j + 1] - pred_quantiles[j])
                            pit = quantiles[j] + ratio * (quantiles[j + 1] - quantiles[j])
                        else:
                            pit = quantiles[j]
                        break

            pit_values.append(np.clip(pit, 0, 1))

        pit_values = np.array(pit_values)

        # Chi-square goodness-of-fit test for uniformity
        num_bins = 10
        observed_freq, _ = np.histogram(pit_values, bins=num_bins, range=(0, 1))
        expected_freq = len(pit_values) / num_bins
        chi2_stat = np.sum((observed_freq - expected_freq) ** 2 / expected_freq)
        chi2_pvalue = 1 - stats.chi2.cdf(chi2_stat, df=num_bins - 1)

        # Kolmogorov-Smirnov test for uniformity
        ks_stat, ks_pvalue = stats.kstest(pit_values, 'uniform')

        pit_results[f'horizon_{h}'] = {
            'pit_values': pit_values.tolist(),
            'mean': float(np.mean(pit_values)),
            'std': float(np.std(pit_values)),
            'chi2_statistic': float(chi2_stat),
            'chi2_pvalue': float(chi2_pvalue),
            'ks_statistic': float(ks_stat),
            'ks_pvalue': float(ks_pvalue),
            'is_uniform': chi2_pvalue > 0.05,  # Null hypothesis: uniform distribution
        }

    return pit_results


def compute_reliability_per_horizon(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    quantiles: List[float] = [0.1, 0.5, 0.9],
    horizons: Optional[List[int]] = None,
    num_bins: int = 10
) -> Dict[str, Dict]:
    """
    Compute reliability (calibration) metrics for each horizon.

    For each quantile, computes:
    - Observed coverage vs nominal coverage
    - Calibration error (absolute difference)
    - Reliability diagram data (binned)

    Args:
        predictions: [N, H, Q] tensor
        targets: [N, H] tensor
        quantiles: List of quantile levels
        horizons: List of horizons to analyze
        num_bins: Number of bins for reliability diagram

    Returns:
        Dictionary with per-horizon reliability metrics
    """
    N, H, Q = predictions.shape

    if horizons is None:
        horizons = list(range(1, H + 1))

    reliability_results = {}

    for h in horizons:
        h_idx = h - 1

        preds_h = predictions[:, h_idx, :].cpu().numpy()  # [N, Q]
        targets_h = targets[:, h_idx].cpu().numpy()       # [N]

        quantile_metrics = {}

        for q_idx, q_level in enumerate(quantiles):
            pred_q = preds_h[:, q_idx]

            # Observed coverage: fraction of actuals below predicted quantile
            observed_coverage = np.mean(targets_h <= pred_q)

            # Calibration error
            calibration_error = abs(observed_coverage - q_level)

            # Reliability diagram: bin predictions and compute observed freq
            bin_edges = np.percentile(pred_q, np.linspace(0, 100, num_bins + 1))
            bin_indices = np.digitize(pred_q, bin_edges[1:-1])

            bin_mean_pred = []
            bin_observed_freq = []
            bin_counts = []

            for bin_idx in range(num_bins):
                mask = bin_indices == bin_idx
                if np.sum(mask) > 0:
                    bin_mean_pred.append(np.mean(pred_q[mask]))
                    bin_observed_freq.append(np.mean(targets_h[mask] <= pred_q[mask]))
                    bin_counts.append(np.sum(mask))
                else:
                    bin_mean_pred.append(np.nan)
                    bin_observed_freq.append(np.nan)
                    bin_counts.append(0)

            quantile_metrics[f'q{int(q_level*100)}'] = {
                'nominal_coverage': q_level,
                'observed_coverage': float(observed_coverage),
                'calibration_error': float(calibration_error),
                'reliability_diagram': {
                    'bin_mean_pred': bin_mean_pred,
                    'bin_observed_freq': bin_observed_freq,
                    'bin_counts': bin_counts
                }
            }

        # Aggregate calibration error across quantiles
        total_calibration_error = np.mean([
            quantile_metrics[f'q{int(q*100)}']['calibration_error']
            for q in quantiles
        ])

        reliability_results[f'horizon_{h}'] = {
            'quantiles': quantile_metrics,
            'mean_calibration_error': float(total_calibration_error)
        }

    return reliability_results


def compute_interval_scores(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    quantiles: List[float] = [0.1, 0.5, 0.9],
    horizons: Optional[List[int]] = None,
    alpha: float = 0.2  # For 80% prediction interval (q10 to q90)
) -> Dict[str, Dict]:
    """
    Compute interval scores (Winkler score and Interval Score) per horizon.

    Interval scores are proper scoring rules for evaluating prediction intervals.
    Lower scores indicate better calibrated and sharper intervals.

    Args:
        predictions: [N, H, Q] tensor
        targets: [N, H] tensor
        quantiles: List of quantile levels
        horizons: Horizons to analyze
        alpha: Miscoverage level (e.g., 0.2 for 80% interval)

    Returns:
        Dictionary with interval scores per horizon
    """
    N, H, Q = predictions.shape

    if horizons is None:
        horizons = list(range(1, H + 1))

    # Identify lower and upper quantiles for interval
    # For 80% interval with q10, q50, q90: use q10 and q90
    lower_idx = 0
    upper_idx = -1
    lower_quantile = quantiles[lower_idx]
    upper_quantile = quantiles[upper_idx]

    interval_results = {}

    for h in horizons:
        h_idx = h - 1

        lower_pred = predictions[:, h_idx, lower_idx].cpu().numpy()  # [N]
        upper_pred = predictions[:, h_idx, upper_idx].cpu().numpy()  # [N]
        targets_h = targets[:, h_idx].cpu().numpy()                  # [N]

        # Winkler Score: penalizes wide intervals and miscoverage
        # WS = (upper - lower) + (2/alpha) * (lower - y) if y < lower
        #                      + (2/alpha) * (y - upper) if y > upper
        interval_width = upper_pred - lower_pred

        penalties = np.zeros(N)
        below_mask = targets_h < lower_pred
        above_mask = targets_h > upper_pred

        penalties[below_mask] = (2 / alpha) * (lower_pred[below_mask] - targets_h[below_mask])
        penalties[above_mask] = (2 / alpha) * (targets_h[above_mask] - upper_pred[above_mask])

        winkler_scores = interval_width + penalties

        # Interval Score (generalized version)
        # IS = (u - l) + (2/alpha) * (l - y) * I(y < l) + (2/alpha) * (y - u) * I(y > u)
        # This is identical to Winkler score in this formulation
        interval_scores = winkler_scores.copy()

        # Sharpness (interval width)
        sharpness = np.mean(interval_width)

        # Coverage (empirical)
        coverage = np.mean((targets_h >= lower_pred) & (targets_h <= upper_pred))

        interval_results[f'horizon_{h}'] = {
            'winkler_score': float(np.mean(winkler_scores)),
            'interval_score': float(np.mean(interval_scores)),
            'sharpness': float(sharpness),
            'coverage': float(coverage),
            'nominal_coverage': 1 - alpha,
            'coverage_error': float(abs(coverage - (1 - alpha))),
            'lower_quantile': lower_quantile,
            'upper_quantile': upper_quantile
        }

    return interval_results


def compute_conditional_calibration(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    metadata_df: pd.DataFrame,
    quantiles: List[float] = [0.1, 0.5, 0.9],
    horizons: List[int] = [1, 6, 12, 24],
    load_regime_breakpoints: List[float] = [38.6, 66.9]
) -> Dict[str, Dict]:
    """
    Compute calibration metrics conditional on various factors.

    Stratifies by:
    - Load regime (low/medium/high based on breakpoints)
    - Peak vs off-peak hours
    - Weekday vs weekend

    Args:
        predictions: [N, H, Q] tensor
        targets: [N, H] tensor
        metadata_df: DataFrame with 'hour', 'day_of_week', 'load_mva' columns
        quantiles: List of quantile levels
        horizons: Horizons to analyze
        load_regime_breakpoints: Thresholds for low/med/high load

    Returns:
        Dictionary with conditional calibration metrics
    """
    N, H, Q = predictions.shape

    # Ensure metadata has required columns
    required_cols = ['hour', 'day_of_week']
    for col in required_cols:
        if col not in metadata_df.columns:
            print(f"[WARNING] Missing column '{col}' in metadata, skipping conditional calibration")
            return {}

    # Get first-step actual load for regime classification
    targets_first_step = targets[:, 0].cpu().numpy()

    # Classify load regime
    load_regime = np.zeros(N, dtype=int)
    load_regime[targets_first_step >= load_regime_breakpoints[1]] = 2  # High
    load_regime[(targets_first_step >= load_regime_breakpoints[0]) &
                (targets_first_step < load_regime_breakpoints[1])] = 1  # Medium
    # load_regime[targets_first_step < load_regime_breakpoints[0]] = 0  # Low (default)

    regime_names = ['low', 'medium', 'high']

    # Classify peak vs off-peak (peak: 8am-8pm)
    hours = metadata_df['hour'].values[:N]
    is_peak = (hours >= 8) & (hours < 20)
    peak_names = ['offpeak', 'peak']

    # Classify weekday vs weekend
    day_of_week = metadata_df['day_of_week'].values[:N]
    is_weekday = day_of_week < 5
    day_type_names = ['weekend', 'weekday']

    conditional_results = {}

    # Stratify by load regime
    conditional_results['by_load_regime'] = {}
    for regime_idx, regime_name in enumerate(regime_names):
        mask = load_regime == regime_idx
        if np.sum(mask) < 10:  # Skip if too few samples
            continue

        regime_results = _compute_calibration_for_subset(
            predictions[mask],
            targets[mask],
            quantiles,
            horizons
        )
        conditional_results['by_load_regime'][regime_name] = regime_results

    # Stratify by peak/off-peak
    conditional_results['by_peak_period'] = {}
    for peak_idx, peak_name in enumerate(peak_names):
        mask = is_peak if peak_idx == 1 else ~is_peak
        if np.sum(mask) < 10:
            continue

        peak_results = _compute_calibration_for_subset(
            predictions[mask],
            targets[mask],
            quantiles,
            horizons
        )
        conditional_results['by_peak_period'][peak_name] = peak_results

    # Stratify by day type
    conditional_results['by_day_type'] = {}
    for day_idx, day_name in enumerate(day_type_names):
        mask = is_weekday if day_idx == 1 else ~is_weekday
        if np.sum(mask) < 10:
            continue

        day_results = _compute_calibration_for_subset(
            predictions[mask],
            targets[mask],
            quantiles,
            horizons
        )
        conditional_results['by_day_type'][day_name] = day_results

    return conditional_results


def _compute_calibration_for_subset(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    quantiles: List[float],
    horizons: List[int]
) -> Dict:
    """Helper function to compute calibration metrics for a data subset."""
    N, H, Q = predictions.shape

    results = {}

    for h in horizons:
        if h > H:
            continue
        h_idx = h - 1

        preds_h = predictions[:, h_idx, :].cpu().numpy()
        targets_h = targets[:, h_idx].cpu().numpy()

        quantile_metrics = {}
        for q_idx, q_level in enumerate(quantiles):
            pred_q = preds_h[:, q_idx]
            observed_coverage = np.mean(targets_h <= pred_q)
            calibration_error = abs(observed_coverage - q_level)

            quantile_metrics[f'q{int(q_level*100)}'] = {
                'observed_coverage': float(observed_coverage),
                'calibration_error': float(calibration_error)
            }

        mean_calib_error = np.mean([
            quantile_metrics[f'q{int(q*100)}']['calibration_error']
            for q in quantiles
        ])

        results[f'horizon_{h}'] = {
            'quantiles': quantile_metrics,
            'mean_calibration_error': float(mean_calib_error),
            'num_samples': int(N)
        }

    return results


def compute_calibration_heatmap_data(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    quantiles: List[float] = [0.1, 0.5, 0.9],
    horizons: Optional[List[int]] = None
) -> Dict[str, np.ndarray]:
    """
    Compute data for calibration error heatmap (horizons × quantiles).

    Args:
        predictions: [N, H, Q] tensor
        targets: [N, H] tensor
        quantiles: List of quantile levels
        horizons: Horizons to include (default: all)

    Returns:
        Dictionary with heatmap data arrays
    """
    N, H, Q = predictions.shape

    if horizons is None:
        horizons = list(range(1, H + 1))

    num_horizons = len(horizons)
    num_quantiles = len(quantiles)

    # Initialize arrays
    calibration_errors = np.zeros((num_horizons, num_quantiles))
    observed_coverages = np.zeros((num_horizons, num_quantiles))

    for h_idx, h in enumerate(horizons):
        h_array_idx = h - 1

        preds_h = predictions[:, h_array_idx, :].cpu().numpy()
        targets_h = targets[:, h_array_idx].cpu().numpy()

        for q_idx, q_level in enumerate(quantiles):
            pred_q = preds_h[:, q_idx]
            observed_coverage = np.mean(targets_h <= pred_q)
            calibration_error = abs(observed_coverage - q_level)

            calibration_errors[h_idx, q_idx] = calibration_error
            observed_coverages[h_idx, q_idx] = observed_coverage

    return {
        'calibration_errors': calibration_errors,
        'observed_coverages': observed_coverages,
        'horizons': horizons,
        'quantiles': quantiles
    }


def run_advanced_calibration_analysis(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    metadata_df: pd.DataFrame,
    quantiles: List[float] = [0.1, 0.5, 0.9],
    horizons: List[int] = [1, 6, 12, 24],
    load_regime_breakpoints: List[float] = [38.6, 66.9]
) -> Dict[str, Dict]:
    """
    Run comprehensive advanced calibration analysis.

    This is the main entry point that computes all calibration metrics.

    Args:
        predictions: [N, H, Q] tensor of quantile predictions
        targets: [N, H] tensor of actual values
        metadata_df: DataFrame with temporal and contextual features
        quantiles: List of quantile levels
        horizons: Key horizons for detailed analysis
        load_regime_breakpoints: Thresholds for load regime classification

    Returns:
        Dictionary containing all calibration analysis results
    """
    print("\nRunning advanced calibration analysis...")

    results = {}

    # 1. Per-horizon PIT analysis
    print("  [1/5] Computing per-horizon PIT analysis...")
    results['pit_per_horizon'] = compute_pit_per_horizon(
        predictions, targets, quantiles, horizons
    )

    # 2. Per-horizon reliability metrics
    print("  [2/5] Computing per-horizon reliability metrics...")
    results['reliability_per_horizon'] = compute_reliability_per_horizon(
        predictions, targets, quantiles, horizons
    )

    # 3. Interval scores
    print("  [3/5] Computing interval scores...")
    results['interval_scores'] = compute_interval_scores(
        predictions, targets, quantiles, horizons
    )

    # 4. Conditional calibration
    print("  [4/5] Computing conditional calibration...")
    results['conditional_calibration'] = compute_conditional_calibration(
        predictions, targets, metadata_df, quantiles, horizons, load_regime_breakpoints
    )

    # 5. Calibration heatmap data
    print("  [5/5] Computing calibration heatmap data...")
    results['heatmap_data'] = compute_calibration_heatmap_data(
        predictions, targets, quantiles, horizons
    )

    print("  [OK] Advanced calibration analysis complete")

    return results
