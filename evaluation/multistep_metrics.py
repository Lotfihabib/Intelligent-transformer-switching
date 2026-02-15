"""
Multi-Step Evaluation Metrics Module

Computes comprehensive per-horizon evaluation metrics for probabilistic forecasting:
- Point forecast metrics (MAE, RMSE, MAPE, R²) for each horizon step
- Probabilistic metrics (CRPS, PICP, Sharpness, Pinball Loss) per horizon
- Calibration analysis (Reliability diagrams, PIT histograms)
- Stratified analysis (by time-of-day, day-type, load regime)

All functions use vectorized operations for efficiency.
"""

import numpy as np
import pandas as pd
from scipy import stats

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


def compute_multistep_point_metrics(predictions, targets, horizons=None, quantiles=[0.1, 0.5, 0.9]):
    """
    Compute point forecast metrics (MAE, RMSE, MAPE, R²) separately for each forecast horizon.

    Args:
        predictions: torch.Tensor [batch, horizon, num_quantiles] (denormalized, MVA)
        targets: torch.Tensor [batch, horizon] (denormalized, MVA)
        horizons: List of horizons to evaluate, or None for all (1 to horizon)
        quantiles: List of quantiles to identify median (default: q50 at index 1)

    Returns:
        dict: {
            'per_horizon': {
                1: {'mae': float, 'rmse': float, 'mape': float, 'r2': float},
                2: {...},
                ...
            },
            'aggregate': {'mae': float, 'rmse': float, 'mape': float, 'r2': float}
        }
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is required for metric computation")

    # Ensure correct shapes
    if predictions.dim() == 2:
        predictions = predictions.unsqueeze(-1)
    if targets.dim() == 1:
        targets = targets.unsqueeze(0)

    batch_size, horizon_len, num_quantiles = predictions.shape

    # Extract median (q50) for point forecasts
    median_idx = len(quantiles) // 2  # Assumes q50 is middle quantile
    point_preds = predictions[:, :, median_idx]  # [batch, horizon]

    # Determine which horizons to evaluate
    if horizons is None:
        horizons = list(range(1, horizon_len + 1))

    results = {'per_horizon': {}, 'aggregate': {}}

    # Vectorized computation for all horizons at once
    errors = torch.abs(point_preds - targets)  # [batch, horizon]
    squared_errors = (point_preds - targets) ** 2  # [batch, horizon]

    # Compute metrics per horizon
    for h_idx, h in enumerate(horizons):
        if h < 1 or h > horizon_len:
            continue

        h_idx_actual = h - 1  # Convert to 0-indexed

        # Extract predictions and targets for this horizon
        pred_h = point_preds[:, h_idx_actual]  # [batch]
        target_h = targets[:, h_idx_actual]  # [batch]

        # Filter out NaN/Inf
        valid_mask = ~(torch.isnan(pred_h) | torch.isinf(pred_h) |
                       torch.isnan(target_h) | torch.isinf(target_h))
        pred_h = pred_h[valid_mask]
        target_h = target_h[valid_mask]

        if len(pred_h) == 0:
            continue

        # MAE
        mae = torch.mean(torch.abs(pred_h - target_h)).item()

        # RMSE
        mse = torch.mean((pred_h - target_h) ** 2).item()
        rmse = np.sqrt(mse)

        # MAPE (handle zero targets)
        target_nonzero = target_h[target_h != 0]
        pred_nonzero = pred_h[target_h != 0]
        if len(target_nonzero) > 0:
            mape = torch.mean(torch.abs((target_nonzero - pred_nonzero) / target_nonzero) * 100).item()
        else:
            mape = np.nan

        # R² score
        ss_res = torch.sum((target_h - pred_h) ** 2).item()
        ss_tot = torch.sum((target_h - torch.mean(target_h)) ** 2).item()
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else np.nan

        results['per_horizon'][h] = {
            'mae': mae,
            'rmse': rmse,
            'mape': mape,
            'r2': r2
        }

    # Aggregate metrics (average across all evaluated horizons)
    if results['per_horizon']:
        results['aggregate'] = {
            'mae': np.mean([m['mae'] for m in results['per_horizon'].values()]),
            'rmse': np.mean([m['rmse'] for m in results['per_horizon'].values()]),
            'mape': np.nanmean([m['mape'] for m in results['per_horizon'].values()]),
            'r2': np.nanmean([m['r2'] for m in results['per_horizon'].values()])
        }

    return results


def compute_multistep_probabilistic_metrics(predictions, targets, horizons=None, quantiles=[0.1, 0.5, 0.9]):
    """
    Compute probabilistic metrics (CRPS, PICP, Sharpness, Pinball Loss) per horizon.

    Args:
        predictions: torch.Tensor [batch, horizon, num_quantiles] (denormalized, MVA)
        targets: torch.Tensor [batch, horizon] (denormalized, MVA)
        horizons: List of horizons to evaluate, or None for all
        quantiles: List of quantile levels (default: [0.1, 0.5, 0.9])

    Returns:
        dict: {
            'per_horizon': {
                1: {
                    'crps': float,
                    'picp_80': float,  # Prediction Interval Coverage Probability
                    'sharpness': float,  # Mean interval width
                    'pinball_q10': float,
                    'pinball_q50': float,
                    'pinball_q90': float,
                    'pinball_total': float
                },
                ...
            },
            'aggregate': {...}
        }
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is required for metric computation")

    # Ensure correct shapes
    if predictions.dim() == 2:
        predictions = predictions.unsqueeze(-1)
    if targets.dim() == 1:
        targets = targets.unsqueeze(0)

    batch_size, horizon_len, num_quantiles = predictions.shape

    # Determine which horizons to evaluate
    if horizons is None:
        horizons = list(range(1, horizon_len + 1))

    results = {'per_horizon': {}, 'aggregate': {}}

    # Compute metrics per horizon
    for h in horizons:
        if h < 1 or h > horizon_len:
            continue

        h_idx = h - 1  # Convert to 0-indexed

        # Extract predictions and targets for this horizon
        preds_h = predictions[:, h_idx, :]  # [batch, num_quantiles]
        targets_h = targets[:, h_idx]  # [batch]

        # Filter out NaN/Inf
        valid_mask = ~(torch.isnan(targets_h) | torch.isinf(targets_h))
        for q_idx in range(num_quantiles):
            valid_mask &= ~(torch.isnan(preds_h[:, q_idx]) | torch.isinf(preds_h[:, q_idx]))

        preds_h = preds_h[valid_mask]
        targets_h = targets_h[valid_mask]

        if len(preds_h) == 0:
            continue

        # Extract quantiles
        q10 = preds_h[:, 0]  # [batch]
        q50 = preds_h[:, 1]  # [batch]
        q90 = preds_h[:, 2]  # [batch]

        # PICP (Prediction Interval Coverage Probability) - 80% interval
        within_interval = (targets_h >= q10) & (targets_h <= q90)
        picp_80 = torch.mean(within_interval.float()).item() * 100  # Percentage

        # Sharpness (Mean Interval Width)
        sharpness = torch.mean(q90 - q10).item()

        # Pinball Loss per quantile
        pinball_losses = []
        for q_idx, q_level in enumerate(quantiles):
            pred_q = preds_h[:, q_idx]
            error = targets_h - pred_q
            loss = torch.where(error >= 0,
                              q_level * error,
                              (q_level - 1) * error)
            pinball_losses.append(torch.mean(loss).item())

        # CRPS (Continuous Ranked Probability Score)
        # Piecewise linear approximation from quantiles
        crps = _compute_crps_from_quantiles(preds_h, targets_h, quantiles)

        results['per_horizon'][h] = {
            'crps': crps,
            'picp_80': picp_80,
            'sharpness': sharpness,
            'pinball_q10': pinball_losses[0],
            'pinball_q50': pinball_losses[1],
            'pinball_q90': pinball_losses[2],
            'pinball_total': sum(pinball_losses)
        }

    # Aggregate metrics
    if results['per_horizon']:
        results['aggregate'] = {
            'crps': np.mean([m['crps'] for m in results['per_horizon'].values()]),
            'picp_80': np.mean([m['picp_80'] for m in results['per_horizon'].values()]),
            'sharpness': np.mean([m['sharpness'] for m in results['per_horizon'].values()]),
            'pinball_q10': np.mean([m['pinball_q10'] for m in results['per_horizon'].values()]),
            'pinball_q50': np.mean([m['pinball_q50'] for m in results['per_horizon'].values()]),
            'pinball_q90': np.mean([m['pinball_q90'] for m in results['per_horizon'].values()]),
            'pinball_total': np.mean([m['pinball_total'] for m in results['per_horizon'].values()])
        }

    return results


def _compute_crps_from_quantiles(preds_quantiles, targets, quantiles):
    """
    Compute CRPS using piecewise linear approximation from quantile predictions.

    Args:
        preds_quantiles: torch.Tensor [batch, num_quantiles]
        targets: torch.Tensor [batch]
        quantiles: List of quantile levels

    Returns:
        float: CRPS value
    """
    # Convert to numpy for easier manipulation
    preds_np = preds_quantiles.cpu().numpy()
    targets_np = targets.cpu().numpy()

    crps_values = []

    for i in range(len(targets_np)):
        target = targets_np[i]
        quantile_preds = preds_np[i]

        # Compute CDF at target value using linear interpolation
        # CRPS = ∫ (F(x) - 1{y <= x})² dx

        # Simplification: Approximate using quantile scores
        # CRPS ≈ 2 * mean(pinball_loss_across_quantiles)
        crps_sample = 0
        for q_idx, q_level in enumerate(quantiles):
            pred = quantile_preds[q_idx]
            error = target - pred
            if error >= 0:
                crps_sample += 2 * q_level * error
            else:
                crps_sample += 2 * (q_level - 1) * abs(error)

        crps_sample /= len(quantiles)
        crps_values.append(crps_sample)

    return np.mean(crps_values)


def compute_calibration_data(predictions, targets, quantiles=[0.1, 0.5, 0.9], num_bins=10):
    """
    Generate reliability diagram and PIT histogram data for calibration assessment.

    Args:
        predictions: torch.Tensor [batch, horizon, num_quantiles] (denormalized)
        targets: torch.Tensor [batch, horizon] (denormalized)
        quantiles: List of quantile levels
        num_bins: Number of bins for PIT histogram

    Returns:
        dict: {
            'reliability_diagram': {
                'q10': {'expected': array, 'observed': array, 'counts': array},
                'q50': {...},
                'q90': {...}
            },
            'pit_histogram': {
                'values': array,  # PIT values (should be uniform [0,1])
                'hist': array,  # Histogram counts
                'bins': array,  # Bin edges
                'chi_square_stat': float,
                'p_value': float
            }
        }
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is required for calibration computation")

    # Flatten across batch and horizon dimensions
    if predictions.dim() == 2:
        predictions = predictions.unsqueeze(-1)
    if targets.dim() == 1:
        targets = targets.unsqueeze(0)

    batch_size, horizon_len, num_quantiles = predictions.shape

    # Flatten
    preds_flat = predictions.reshape(-1, num_quantiles)  # [batch*horizon, num_quantiles]
    targets_flat = targets.reshape(-1)  # [batch*horizon]

    # Filter out NaN/Inf
    valid_mask = ~(torch.isnan(targets_flat) | torch.isinf(targets_flat))
    for q_idx in range(num_quantiles):
        valid_mask &= ~(torch.isnan(preds_flat[:, q_idx]) | torch.isinf(preds_flat[:, q_idx]))

    preds_flat = preds_flat[valid_mask]
    targets_flat = targets_flat[valid_mask]

    if len(preds_flat) == 0:
        return {'reliability_diagram': {}, 'pit_histogram': {}}

    # Reliability Diagram: Observed vs Expected coverage per quantile
    reliability_data = {}

    for q_idx, q_level in enumerate(quantiles):
        pred_q = preds_flat[:, q_idx]

        # Compute observed coverage (proportion of targets below prediction)
        below_pred = (targets_flat <= pred_q).float()
        observed_coverage = torch.mean(below_pred).item()

        # Bin predictions for calibration curve
        # Sort by prediction value and create bins
        sorted_indices = torch.argsort(pred_q)
        bin_size = len(pred_q) // num_bins

        expected_coverages = []
        observed_coverages = []
        counts = []

        for b in range(num_bins):
            start_idx = b * bin_size
            end_idx = (b + 1) * bin_size if b < num_bins - 1 else len(pred_q)

            bin_indices = sorted_indices[start_idx:end_idx]
            bin_targets = targets_flat[bin_indices]
            bin_preds = pred_q[bin_indices]

            # Expected coverage for this bin: quantile level
            expected_cov = q_level

            # Observed coverage for this bin
            observed_cov = torch.mean((bin_targets <= bin_preds).float()).item()

            expected_coverages.append(expected_cov)
            observed_coverages.append(observed_cov)
            counts.append(len(bin_indices))

        reliability_data[f'q{int(q_level*100)}'] = {
            'expected': np.array(expected_coverages),
            'observed': np.array(observed_coverages),
            'counts': np.array(counts)
        }

    # PIT (Probability Integral Transform) Histogram
    # For each observation, compute its CDF value through the predictive distribution
    pit_values = _compute_pit_values(preds_flat, targets_flat, quantiles)

    # Create histogram
    hist, bins = np.histogram(pit_values, bins=num_bins, range=(0, 1))

    # Chi-square test for uniformity
    expected_count = len(pit_values) / num_bins
    chi_square_stat = np.sum((hist - expected_count) ** 2 / expected_count)
    p_value = 1 - stats.chi2.cdf(chi_square_stat, num_bins - 1)

    pit_data = {
        'values': pit_values,
        'hist': hist,
        'bins': bins,
        'chi_square_stat': chi_square_stat,
        'p_value': p_value
    }

    return {
        'reliability_diagram': reliability_data,
        'pit_histogram': pit_data
    }


def _compute_pit_values(preds_quantiles, targets, quantiles):
    """
    Compute Probability Integral Transform values.

    For each observation, compute where it falls in the predictive CDF.
    If forecasts are well-calibrated, PIT values should be uniform on [0, 1].

    Args:
        preds_quantiles: torch.Tensor [N, num_quantiles]
        targets: torch.Tensor [N]
        quantiles: List of quantile levels

    Returns:
        np.array: PIT values in [0, 1]
    """
    preds_np = preds_quantiles.cpu().numpy()
    targets_np = targets.cpu().numpy()

    pit_values = []

    for i in range(len(targets_np)):
        target = targets_np[i]
        quantile_preds = preds_np[i]

        # Find where target falls in quantile predictions
        # Use linear interpolation between quantiles

        if target <= quantile_preds[0]:
            # Below lowest quantile
            pit = quantiles[0] * (target / quantile_preds[0]) if quantile_preds[0] > 0 else 0
        elif target >= quantile_preds[-1]:
            # Above highest quantile
            pit = quantiles[-1] + (1 - quantiles[-1]) * ((target - quantile_preds[-1]) /
                                                          (quantile_preds[-1] + 1e-6))
            pit = min(pit, 1.0)
        else:
            # Interpolate between quantiles
            for j in range(len(quantiles) - 1):
                if quantile_preds[j] <= target <= quantile_preds[j + 1]:
                    # Linear interpolation
                    if quantile_preds[j + 1] - quantile_preds[j] > 1e-6:
                        alpha = (target - quantile_preds[j]) / (quantile_preds[j + 1] - quantile_preds[j])
                        pit = quantiles[j] + alpha * (quantiles[j + 1] - quantiles[j])
                    else:
                        pit = (quantiles[j] + quantiles[j + 1]) / 2
                    break

        pit_values.append(np.clip(pit, 0, 1))

    return np.array(pit_values)


def compute_stratified_multistep_metrics(predictions, targets, metadata_df, horizons=[1, 6, 12, 24],
                                         quantiles=[0.1, 0.5, 0.9]):
    """
    Compute metrics stratified by conditions (time-of-day, day-type, load regime).

    Args:
        predictions: torch.Tensor [batch, horizon, num_quantiles] (denormalized)
        targets: torch.Tensor [batch, horizon] (denormalized)
        metadata_df: pandas.DataFrame with index-aligned columns:
            - 'hour' (0-23)
            - 'is_weekend' (bool)
            - 'load_regime' (categorical: 'low', 'medium', 'high')
        horizons: List of specific horizons to evaluate (for efficiency)
        quantiles: List of quantile levels

    Returns:
        dict: {
            'by_hour': {0: {1: metrics, 6: metrics, ...}, 1: {...}, ...},
            'by_day_type': {'weekday': {1: metrics, ...}, 'weekend': {...}},
            'by_load_regime': {'low': {...}, 'medium': {...}, 'high': {...}},
            'peak_vs_offpeak': {'peak': {...}, 'offpeak': {...}}
        }
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is required for stratified metrics")

    # Ensure correct shapes
    if predictions.dim() == 2:
        predictions = predictions.unsqueeze(-1)
    if targets.dim() == 1:
        targets = targets.unsqueeze(0)

    batch_size, horizon_len, num_quantiles = predictions.shape

    # Verify metadata alignment
    if len(metadata_df) != batch_size:
        raise ValueError(f"Metadata length ({len(metadata_df)}) must match batch size ({batch_size})")

    results = {
        'by_hour': {},
        'by_day_type': {},
        'by_load_regime': {},
        'peak_vs_offpeak': {}
    }

    # Stratify by hour
    if 'hour' in metadata_df.columns:
        for hour in range(24):
            hour_mask = metadata_df['hour'] == hour
            if hour_mask.sum() > 0:
                hour_preds = predictions[hour_mask.values]
                hour_targets = targets[hour_mask.values]

                hour_metrics = {}
                for h in horizons:
                    if h > horizon_len:
                        continue
                    h_idx = h - 1
                    point_metric = compute_multistep_point_metrics(
                        hour_preds[:, h_idx:h_idx+1, :],
                        hour_targets[:, h_idx:h_idx+1],
                        horizons=[1], quantiles=quantiles
                    )
                    hour_metrics[h] = point_metric['per_horizon'][1] if 1 in point_metric['per_horizon'] else {}

                results['by_hour'][hour] = hour_metrics

    # Stratify by day type (weekday vs weekend)
    if 'is_weekend' in metadata_df.columns:
        for day_type, is_weekend_val in [('weekday', False), ('weekend', True)]:
            day_mask = metadata_df['is_weekend'] == is_weekend_val
            if day_mask.sum() > 0:
                day_preds = predictions[day_mask.values]
                day_targets = targets[day_mask.values]

                day_metrics = {}
                for h in horizons:
                    if h > horizon_len:
                        continue
                    h_idx = h - 1
                    point_metric = compute_multistep_point_metrics(
                        day_preds[:, h_idx:h_idx+1, :],
                        day_targets[:, h_idx:h_idx+1],
                        horizons=[1], quantiles=quantiles
                    )
                    day_metrics[h] = point_metric['per_horizon'][1] if 1 in point_metric['per_horizon'] else {}

                results['by_day_type'][day_type] = day_metrics

    # Stratify by load regime
    if 'load_regime' in metadata_df.columns:
        for regime in ['low', 'medium', 'high']:
            regime_mask = metadata_df['load_regime'] == regime
            if regime_mask.sum() > 0:
                regime_preds = predictions[regime_mask.values]
                regime_targets = targets[regime_mask.values]

                regime_metrics = {}
                for h in horizons:
                    if h > horizon_len:
                        continue
                    h_idx = h - 1
                    point_metric = compute_multistep_point_metrics(
                        regime_preds[:, h_idx:h_idx+1, :],
                        regime_targets[:, h_idx:h_idx+1],
                        horizons=[1], quantiles=quantiles
                    )
                    regime_metrics[h] = point_metric['per_horizon'][1] if 1 in point_metric['per_horizon'] else {}

                results['by_load_regime'][regime] = regime_metrics

    # Peak vs off-peak (peak: 10-14, 18-22)
    if 'hour' in metadata_df.columns:
        peak_hours = [10, 11, 12, 13, 14, 18, 19, 20, 21, 22]
        peak_mask = metadata_df['hour'].isin(peak_hours)

        for period, mask in [('peak', peak_mask), ('offpeak', ~peak_mask)]:
            if mask.sum() > 0:
                period_preds = predictions[mask.values]
                period_targets = targets[mask.values]

                period_metrics = {}
                for h in horizons:
                    if h > horizon_len:
                        continue
                    h_idx = h - 1
                    point_metric = compute_multistep_point_metrics(
                        period_preds[:, h_idx:h_idx+1, :],
                        period_targets[:, h_idx:h_idx+1],
                        horizons=[1], quantiles=quantiles
                    )
                    period_metrics[h] = point_metric['per_horizon'][1] if 1 in point_metric['per_horizon'] else {}

                results['peak_vs_offpeak'][period] = period_metrics

    return results
