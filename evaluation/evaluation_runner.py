"""
Evaluation Runner Module

Orchestrates comprehensive multi-step evaluation workflow:
- Batch prediction generation
- Denormalization
- Multi-step metric computation
- Calibration analysis
- Stratified analysis
- Result saving (CSV/JSON)
- Visualization generation
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


def run_comprehensive_evaluation(model, data_loader, device, metadata, outdir, horizons='all', advanced_calibration=False):
    """
    Main entry point for post-training comprehensive evaluation.

    Workflow:
        1. Batch-generate predictions on full dataset
        2. Denormalize using metadata
        3. Compute per-horizon metrics
        4. Compute calibration analysis
        5. Compute stratified metrics
        6. (Optional) Advanced calibration analysis
        7. Save results (CSV/JSON)
        8. Generate visualizations

    Args:
        model: Trained TFT model
        data_loader: PyTorch DataLoader (test or validation set)
        device: torch device ('cuda' or 'cpu')
        metadata: Preprocessing metadata dict with normalization stats
        outdir: Output directory path (str or Path)
        horizons: 'all' for all horizons, or list of specific horizons to evaluate
        advanced_calibration: bool, if True run advanced calibration analysis (Phase 3)

    Returns:
        dict: {
            'point': point_metrics,
            'probabilistic': prob_metrics,
            'calibration': calib_data,
            'stratified': stratified_metrics,
            'advanced_calibration': advanced_calib_results (if enabled)
        }
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is required for evaluation")

    from .multistep_metrics import (
        compute_multistep_point_metrics,
        compute_multistep_probabilistic_metrics,
        compute_calibration_data,
        compute_stratified_multistep_metrics
    )

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print("\n[STEP 1/7] Collecting predictions...")
    all_preds, all_targets, timestamps = collect_predictions(model, data_loader, device)
    print(f"  Collected {len(all_preds)} samples with horizon={all_preds.shape[1]}, quantiles={all_preds.shape[2]}")

    print("\n[STEP 2/7] Denormalizing predictions and targets...")
    preds_denorm = denormalize_predictions(all_preds, metadata)
    targets_denorm = denormalize_targets(all_targets, metadata)
    print(f"  Denormalized to MVA range: [{targets_denorm.min():.2f}, {targets_denorm.max():.2f}]")

    # Determine horizons to evaluate
    horizon_len = all_preds.shape[1]
    if horizons == 'all':
        eval_horizons = None  # Evaluate all
    elif isinstance(horizons, list):
        eval_horizons = horizons
    else:
        eval_horizons = None

    print("\n[STEP 3/7] Computing per-horizon point metrics...")
    point_metrics = compute_multistep_point_metrics(preds_denorm, targets_denorm, horizons=eval_horizons)
    print(f"  Computed metrics for {len(point_metrics['per_horizon'])} horizons")
    print(f"  Aggregate MAE: {point_metrics['aggregate']['mae']:.3f} MVA")
    print(f"  Aggregate RMSE: {point_metrics['aggregate']['rmse']:.3f} MVA")

    print("\n[STEP 4/7] Computing per-horizon probabilistic metrics...")
    prob_metrics = compute_multistep_probabilistic_metrics(preds_denorm, targets_denorm, horizons=eval_horizons)
    print(f"  Aggregate CRPS: {prob_metrics['aggregate']['crps']:.3f}")
    print(f"  Aggregate PICP (80%): {prob_metrics['aggregate']['picp_80']:.1f}%")

    print("\n[STEP 5/7] Computing calibration data...")
    calib_data = compute_calibration_data(preds_denorm, targets_denorm)
    if 'pit_histogram' in calib_data and 'p_value' in calib_data['pit_histogram']:
        print(f"  PIT uniformity test p-value: {calib_data['pit_histogram']['p_value']:.4f}")

    print("\n[STEP 6/7] Computing stratified metrics...")
    metadata_df = build_metadata_df(timestamps, targets_denorm, metadata)
    key_horizons = [1, 6, 12, 24] if horizon_len >= 24 else [1, horizon_len // 2, horizon_len]
    stratified_metrics = compute_stratified_multistep_metrics(preds_denorm, targets_denorm, metadata_df, horizons=key_horizons)
    print(f"  Stratified by: {list(stratified_metrics.keys())}")

    # Advanced calibration analysis (Phase 3)
    advanced_calib_results = None
    if advanced_calibration:
        print("\n[STEP 6.5/7] Running advanced calibration analysis...")
        try:
            from .calibration_analysis import run_advanced_calibration_analysis
            from config import CONFIG

            # Get configuration parameters
            calib_horizons = CONFIG.get('calibration_horizons', [1, 6, 12, 24])
            load_regime_bp = CONFIG.get('load_regime_breakpoints', [38.6, 66.9])

            advanced_calib_results = run_advanced_calibration_analysis(
                predictions=preds_denorm,
                targets=targets_denorm,
                metadata_df=metadata_df,
                quantiles=[0.1, 0.5, 0.9],
                horizons=calib_horizons,
                load_regime_breakpoints=load_regime_bp
            )
            print(f"  [OK] Advanced calibration analysis complete")

        except Exception as e:
            print(f"  [WARNING] Advanced calibration analysis failed: {e}")
            import traceback
            traceback.print_exc()

    print("\n[STEP 7/7] Saving results...")
    save_evaluation_results(point_metrics, prob_metrics, calib_data, stratified_metrics, outdir, advanced_calib_results)
    print(f"  Saved to {outdir}/")

    # Generate visualizations
    print("\nGenerating visualizations...")
    try:
        from visualization.training_plots import plot_multistep_evaluation
        plot_multistep_evaluation(point_metrics, prob_metrics, calib_data, outdir)
        print(f"  Generated evaluation plots in {outdir}/")
    except ImportError as e:
        print(f"  Warning: Could not generate plots: {e}")

    # Generate advanced calibration visualizations
    if advanced_calibration and advanced_calib_results is not None:
        try:
            from visualization.training_plots import plot_advanced_calibration_summary
            plot_advanced_calibration_summary(advanced_calib_results, outdir)
        except Exception as e:
            print(f"  Warning: Could not generate advanced calibration plots: {e}")
            import traceback
            traceback.print_exc()

    results = {
        'point': point_metrics,
        'probabilistic': prob_metrics,
        'calibration': calib_data,
        'stratified': stratified_metrics
    }

    if advanced_calib_results is not None:
        results['advanced_calibration'] = advanced_calib_results

    return results


def collect_predictions(model, data_loader, device):
    """
    Batch-generate predictions and collect results.

    Args:
        model: Trained TFT model
        data_loader: PyTorch DataLoader
        device: torch device

    Returns:
        tuple: (predictions, targets, timestamps)
            predictions: torch.Tensor [N_samples, horizon, num_quantiles]
            targets: torch.Tensor [N_samples, horizon]
            timestamps: list of timestamps (if available) or None
    """
    model.eval()
    preds_list = []
    targets_list = []
    timestamps_list = []

    with torch.no_grad():
        for batch_data in data_loader:
            # Unpack batch (format from SimpleTimeSeriesDataset)
            if len(batch_data) == 3:
                features, hist_target, future_target = batch_data
            else:
                raise ValueError(f"Expected 3 items from data_loader, got {len(batch_data)}")

            features = features.to(device)
            hist_target = hist_target.to(device)

            # Generate predictions
            predictions = model(features, hist_target)  # [batch, horizon, num_quantiles]

            preds_list.append(predictions.cpu())
            targets_list.append(future_target)

            # Extract timestamps if available (requires custom dataset)
            # For now, timestamps will be None
            # In future, can extend SimpleTimeSeriesDataset to include timestamps

    # Concatenate all batches
    all_preds = torch.cat(preds_list, dim=0)
    all_targets = torch.cat(targets_list, dim=0)

    return all_preds, all_targets, None


def denormalize_predictions(predictions, metadata):
    """
    Denormalize predictions from normalized (z-score) to original scale (MVA).

    Args:
        predictions: torch.Tensor [N, horizon, num_quantiles] (normalized)
        metadata: dict with 'normalization' key containing 'target_mean' and 'target_std'

    Returns:
        torch.Tensor [N, horizon, num_quantiles] (denormalized, MVA)
    """
    if 'normalization' not in metadata:
        print("  Warning: No normalization metadata found, assuming already denormalized")
        return predictions

    target_mean = metadata['normalization'].get('target_mean', 0)
    target_std = metadata['normalization'].get('target_std', 1)

    # Denormalize: value = (normalized * std) + mean
    predictions_denorm = predictions * target_std + target_mean

    return predictions_denorm


def denormalize_targets(targets, metadata):
    """
    Denormalize targets from normalized (z-score) to original scale (MVA).

    Args:
        targets: torch.Tensor [N, horizon] (normalized)
        metadata: dict with 'normalization' key

    Returns:
        torch.Tensor [N, horizon] (denormalized, MVA)
    """
    if 'normalization' not in metadata:
        print("  Warning: No normalization metadata found, assuming already denormalized")
        return targets

    target_mean = metadata['normalization'].get('target_mean', 0)
    target_std = metadata['normalization'].get('target_std', 1)

    # Denormalize: value = (normalized * std) + mean
    targets_denorm = targets * target_std + target_mean

    return targets_denorm


def build_metadata_df(timestamps, targets_denorm, metadata):
    """
    Build metadata DataFrame with temporal and load features.

    Args:
        timestamps: list of pd.Timestamp or None
        targets_denorm: torch.Tensor [N, horizon] (denormalized, MVA)
        metadata: Preprocessing metadata dict

    Returns:
        pd.DataFrame with columns: hour, is_weekend, load_regime
    """
    from config import CONFIG

    n_samples = targets_denorm.shape[0]

    # Initialize metadata dict
    meta_dict = {}

    # If timestamps available, extract temporal features
    if timestamps is not None and len(timestamps) > 0:
        # Convert to pandas timestamps if not already
        if not isinstance(timestamps[0], pd.Timestamp):
            timestamps = pd.to_datetime(timestamps)

        meta_dict['hour'] = [ts.hour for ts in timestamps]
        meta_dict['day_of_week'] = [ts.dayofweek for ts in timestamps]
        meta_dict['is_weekend'] = [ts.dayofweek >= 5 for ts in timestamps]
    else:
        # Generate placeholder temporal features
        # Use cycling pattern to approximate hour of day
        meta_dict['hour'] = [i % 24 for i in range(n_samples)]
        meta_dict['day_of_week'] = [i % 7 for i in range(n_samples)]
        meta_dict['is_weekend'] = [(i % 7) >= 5 for i in range(n_samples)]

    # Compute load regime based on first-step targets
    targets_first_step = targets_denorm[:, 0].cpu().numpy()  # [N]

    # Get breakpoints from config
    breakpoints = CONFIG.get('load_regime_breakpoints', [38.6, 66.9])
    bp_low = breakpoints[0]
    bp_high = breakpoints[1]

    load_regimes = []
    for load in targets_first_step:
        if load < bp_low:
            regime = 'low'
        elif load < bp_high:
            regime = 'medium'
        else:
            regime = 'high'
        load_regimes.append(regime)

    meta_dict['load_regime'] = load_regimes

    return pd.DataFrame(meta_dict)


def save_evaluation_results(point_metrics, prob_metrics, calib_data, stratified_metrics, outdir, advanced_calib_results=None):
    """
    Save evaluation results to CSV and JSON files.

    Args:
        point_metrics: dict from compute_multistep_point_metrics()
        prob_metrics: dict from compute_multistep_probabilistic_metrics()
        calib_data: dict from compute_calibration_data()
        stratified_metrics: dict from compute_stratified_multistep_metrics()
        outdir: Path object for output directory
        advanced_calib_results: dict from run_advanced_calibration_analysis() (optional)
    """
    outdir = Path(outdir)

    # 1. Per-horizon metrics CSV
    horizon_rows = []
    for h in sorted(point_metrics['per_horizon'].keys()):
        row = {'horizon': h}
        row.update(point_metrics['per_horizon'][h])
        if h in prob_metrics['per_horizon']:
            row.update(prob_metrics['per_horizon'][h])
        horizon_rows.append(row)

    df_horizons = pd.DataFrame(horizon_rows)
    df_horizons.to_csv(outdir / 'multistep_metrics_per_horizon.csv', index=False, float_format='%.4f')
    print(f"    - multistep_metrics_per_horizon.csv ({len(df_horizons)} horizons)")

    # 2. Calibration data JSON
    # Convert numpy arrays to lists for JSON serialization
    calib_data_serializable = _make_json_serializable(calib_data)
    with open(outdir / 'calibration_data.json', 'w') as f:
        json.dump(calib_data_serializable, f, indent=2)
    print(f"    - calibration_data.json")

    # 3. Stratified metrics JSON
    stratified_serializable = _make_json_serializable(stratified_metrics)
    with open(outdir / 'stratified_metrics.json', 'w') as f:
        json.dump(stratified_serializable, f, indent=2)
    print(f"    - stratified_metrics.json")

    # 4. Evaluation summary JSON
    summary = {
        'aggregate_point_metrics': point_metrics['aggregate'],
        'aggregate_probabilistic_metrics': prob_metrics['aggregate']
    }

    # Add advanced calibration summary if available
    if advanced_calib_results is not None:
        # Extract key metrics from advanced calibration
        summary['advanced_calibration_summary'] = {
            'mean_calibration_errors_by_horizon': {},
            'pit_uniformity_tests': {},
            'interval_score_summary': {}
        }

        # Extract per-horizon calibration errors
        if 'reliability_per_horizon' in advanced_calib_results:
            for h_key, h_data in advanced_calib_results['reliability_per_horizon'].items():
                summary['advanced_calibration_summary']['mean_calibration_errors_by_horizon'][h_key] = \
                    h_data.get('mean_calibration_error', None)

        # Extract PIT test results
        if 'pit_per_horizon' in advanced_calib_results:
            for h_key, h_data in advanced_calib_results['pit_per_horizon'].items():
                summary['advanced_calibration_summary']['pit_uniformity_tests'][h_key] = {
                    'chi2_pvalue': h_data.get('chi2_pvalue', None),
                    'ks_pvalue': h_data.get('ks_pvalue', None),
                    'is_uniform': h_data.get('is_uniform', None)
                }

        # Extract interval scores
        if 'interval_scores' in advanced_calib_results:
            for h_key, h_data in advanced_calib_results['interval_scores'].items():
                summary['advanced_calibration_summary']['interval_score_summary'][h_key] = {
                    'winkler_score': h_data.get('winkler_score', None),
                    'coverage': h_data.get('coverage', None),
                    'sharpness': h_data.get('sharpness', None)
                }

    # Convert summary to JSON-serializable format
    summary_serializable = _make_json_serializable(summary)
    with open(outdir / 'evaluation_summary.json', 'w') as f:
        json.dump(summary_serializable, f, indent=2)
    print(f"    - evaluation_summary.json")

    # 5. Advanced calibration results JSON (if available)
    if advanced_calib_results is not None:
        advanced_calib_serializable = _make_json_serializable(advanced_calib_results)
        with open(outdir / 'advanced_calibration_results.json', 'w') as f:
            json.dump(advanced_calib_serializable, f, indent=2)
        print(f"    - advanced_calibration_results.json")


def _make_json_serializable(obj):
    """
    Recursively convert numpy arrays and torch tensors to lists for JSON serialization.

    Args:
        obj: Any object (dict, list, array, etc.)

    Returns:
        JSON-serializable version of obj
    """
    if isinstance(obj, dict):
        return {key: _make_json_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [_make_json_serializable(item) for item in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    elif TORCH_AVAILABLE and isinstance(obj, torch.Tensor):
        return obj.cpu().numpy().tolist()
    elif isinstance(obj, (int, float, str, bool, type(None))):
        return obj
    else:
        # Fallback: convert to string
        return str(obj)
