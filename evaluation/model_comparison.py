"""
Model Comparison Module

Compares multiple forecasting models on the same test set:
- Collects predictions from all models
- Computes unified metrics
- Generates comparison tables and visualizations

Handles both quantile-based (TFT) and point forecast (Persistence, LSTM) models.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


def compare_models(models_dict, data_loader, device, metadata, outdir):
    """
    Compare multiple models on the same test set.

    Args:
        models_dict: dict mapping model names to model instances
            Example: {'TFT': tft_model, 'Persistence': pers_model, 'LSTM': lstm_model}
        data_loader: PyTorch DataLoader (test set)
        device: torch device ('cuda' or 'cpu')
        metadata: Preprocessing metadata dict with normalization stats
        outdir: Output directory path

    Returns:
        dict: Comparison results with structure:
            {
                'model_name': {
                    'point': point_metrics_dict,
                    'probabilistic': prob_metrics_dict (if quantiles available)
                }
            }
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is required for model comparison")

    from .evaluation_runner import collect_predictions, denormalize_predictions, denormalize_targets
    from .multistep_metrics import compute_multistep_point_metrics, compute_multistep_probabilistic_metrics

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*70)
    print("MODEL COMPARISON")
    print("="*70)

    results = {}

    for name, model in models_dict.items():
        print(f"\n[Evaluating {name}]")

        # Collect predictions
        preds, targets, _ = collect_predictions(model, data_loader, device)
        print(f"  Collected {len(preds)} predictions")

        # Handle point forecasts (models without quantiles)
        if preds.shape[-1] == 1:
            print(f"  Point forecast model (no quantiles)")
            # Convert to pseudo-quantiles for consistent interface
            # Repeat the point forecast as q10, q50, q90 (all equal)
            preds = preds.repeat(1, 1, 3)
        else:
            print(f"  Probabilistic forecast model ({preds.shape[-1]} quantiles)")

        # Denormalize
        preds_denorm = denormalize_predictions(preds, metadata)
        targets_denorm = denormalize_targets(targets, metadata)

        # Compute point metrics
        print(f"  Computing point forecast metrics...")
        point_metrics = compute_multistep_point_metrics(
            preds_denorm, targets_denorm,
            horizons=None,  # Compute all horizons
            quantiles=[0.1, 0.5, 0.9]
        )

        # Compute probabilistic metrics (will be same for all quantiles if point forecast)
        print(f"  Computing probabilistic metrics...")
        prob_metrics = compute_multistep_probabilistic_metrics(
            preds_denorm, targets_denorm,
            horizons=None,
            quantiles=[0.1, 0.5, 0.9]
        )

        results[name] = {
            'point': point_metrics,
            'probabilistic': prob_metrics,
            'predictions': preds_denorm,  # Store for later analysis
            'targets': targets_denorm
        }

        # Print summary
        print(f"  Aggregate MAE: {point_metrics['aggregate']['mae']:.3f} MVA")
        print(f"  Aggregate RMSE: {point_metrics['aggregate']['rmse']:.3f} MVA")
        print(f"  Aggregate R²: {point_metrics['aggregate']['r2']:.3f}")

    print("\n[Saving comparison results...]")

    # Save comparison tables
    save_comparison_tables(results, outdir)

    # Generate comparison plots
    plot_comparison_charts(results, outdir)

    print(f"[OK] Model comparison complete. Results saved to {outdir}/")

    return results


def save_comparison_tables(results, outdir):
    """
    Save comparison results to CSV tables.

    Creates two tables:
        1. model_comparison_aggregate.csv - Aggregate metrics across all horizons
        2. model_comparison_horizons.csv - Per-horizon breakdown

    Args:
        results: dict from compare_models()
        outdir: Output directory path
    """
    outdir = Path(outdir)

    # Table 1: Aggregate metrics
    aggregate_rows = []
    for model_name, model_results in results.items():
        row = {'Model': model_name}
        row.update(model_results['point']['aggregate'])
        row.update(model_results['probabilistic']['aggregate'])
        aggregate_rows.append(row)

    df_aggregate = pd.DataFrame(aggregate_rows)

    # Reorder columns for readability
    col_order = ['Model', 'mae', 'rmse', 'mape', 'r2', 'crps', 'picp_80', 'sharpness']
    existing_cols = [c for c in col_order if c in df_aggregate.columns]
    df_aggregate = df_aggregate[existing_cols]

    aggregate_path = outdir / 'model_comparison_aggregate.csv'
    df_aggregate.to_csv(aggregate_path, index=False, float_format='%.4f')
    print(f"  - model_comparison_aggregate.csv")

    # Table 2: Per-horizon metrics
    horizon_rows = []
    for model_name, model_results in results.items():
        for h, h_metrics in model_results['point']['per_horizon'].items():
            row = {'Model': model_name, 'Horizon': h}
            row.update(h_metrics)
            if h in model_results['probabilistic']['per_horizon']:
                row.update(model_results['probabilistic']['per_horizon'][h])
            horizon_rows.append(row)

    df_horizons = pd.DataFrame(horizon_rows)
    horizons_path = outdir / 'model_comparison_horizons.csv'
    df_horizons.to_csv(horizons_path, index=False, float_format='%.4f')
    print(f"  - model_comparison_horizons.csv")

    # Print summary table to console
    print("\n" + "="*70)
    print("MODEL COMPARISON SUMMARY")
    print("="*70)
    print(df_aggregate.to_string(index=False))
    print("="*70)


def plot_comparison_charts(results, outdir):
    """
    Generate comparison visualizations.

    Creates two plots:
        1. model_comparison_aggregate.png - Bar charts of aggregate metrics
        2. model_comparison_horizons.png - Line plots showing error growth across horizons

    Args:
        results: dict from compare_models()
        outdir: Output directory path
    """
    outdir = Path(outdir)

    # Plot 1: Aggregate metrics bar chart
    _plot_aggregate_comparison(results, outdir)

    # Plot 2: Per-horizon error curves
    _plot_horizon_comparison(results, outdir)


def _plot_aggregate_comparison(results, outdir):
    """
    Bar chart comparing aggregate metrics across models.

    4-panel figure: MAE, RMSE, MAPE, R²
    """
    model_names = list(results.keys())
    n_models = len(model_names)

    # Extract aggregate metrics
    mae_vals = [results[m]['point']['aggregate']['mae'] for m in model_names]
    rmse_vals = [results[m]['point']['aggregate']['rmse'] for m in model_names]
    mape_vals = [results[m]['point']['aggregate']['mape'] for m in model_names]
    r2_vals = [results[m]['point']['aggregate']['r2'] for m in model_names]

    # Create figure
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))

    # Color palette
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#6A994E'][:n_models]

    # Panel 1: MAE
    bars1 = ax1.bar(range(n_models), mae_vals, color=colors, edgecolor='black', linewidth=1.5)
    ax1.set_title('Mean Absolute Error (MAE)', fontsize=13, fontweight='bold')
    ax1.set_ylabel('MAE (MVA)', fontsize=11)
    ax1.set_xticks(range(n_models))
    ax1.set_xticklabels(model_names, fontsize=10)
    ax1.grid(axis='y', alpha=0.3)
    # Annotate bars
    for i, (bar, val) in enumerate(zip(bars1, mae_vals)):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(mae_vals)*0.02,
                f'{val:.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    # Panel 2: RMSE
    bars2 = ax2.bar(range(n_models), rmse_vals, color=colors, edgecolor='black', linewidth=1.5)
    ax2.set_title('Root Mean Squared Error (RMSE)', fontsize=13, fontweight='bold')
    ax2.set_ylabel('RMSE (MVA)', fontsize=11)
    ax2.set_xticks(range(n_models))
    ax2.set_xticklabels(model_names, fontsize=10)
    ax2.grid(axis='y', alpha=0.3)
    for i, (bar, val) in enumerate(zip(bars2, rmse_vals)):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(rmse_vals)*0.02,
                f'{val:.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    # Panel 3: MAPE
    bars3 = ax3.bar(range(n_models), mape_vals, color=colors, edgecolor='black', linewidth=1.5)
    ax3.set_title('Mean Absolute Percentage Error (MAPE)', fontsize=13, fontweight='bold')
    ax3.set_ylabel('MAPE (%)', fontsize=11)
    ax3.set_xticks(range(n_models))
    ax3.set_xticklabels(model_names, fontsize=10)
    ax3.grid(axis='y', alpha=0.3)
    for i, (bar, val) in enumerate(zip(bars3, mape_vals)):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(mape_vals)*0.02,
                f'{val:.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    # Panel 4: R²
    bars4 = ax4.bar(range(n_models), r2_vals, color=colors, edgecolor='black', linewidth=1.5)
    ax4.set_title('R² Score', fontsize=13, fontweight='bold')
    ax4.set_ylabel('R²', fontsize=11)
    ax4.set_xticks(range(n_models))
    ax4.set_xticklabels(model_names, fontsize=10)
    ax4.grid(axis='y', alpha=0.3)
    ax4.set_ylim(0, 1.05)
    for i, (bar, val) in enumerate(zip(bars4, r2_vals)):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.tight_layout()
    plt.savefig(outdir / 'model_comparison_aggregate.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  - model_comparison_aggregate.png")


def _plot_horizon_comparison(results, outdir):
    """
    Line plots showing error growth across forecast horizons.

    4-panel figure: MAE, RMSE, MAPE, R² vs horizon for all models
    """
    model_names = list(results.keys())
    n_models = len(model_names)

    # Get horizons from first model
    first_model = model_names[0]
    horizons = sorted(results[first_model]['point']['per_horizon'].keys())

    # Color and style for each model
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#6A994E'][:n_models]
    markers = ['o', 's', '^', 'D'][:n_models]
    linestyles = ['-', '--', '-.', ':'][:n_models]

    # Create figure
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

    # Panel 1: MAE vs Horizon
    for i, model_name in enumerate(model_names):
        mae_vals = [results[model_name]['point']['per_horizon'][h]['mae'] for h in horizons]
        ax1.plot(horizons, mae_vals, marker=markers[i], linestyle=linestyles[i],
                color=colors[i], linewidth=2, markersize=6, label=model_name)

    ax1.set_title('MAE vs Forecast Horizon', fontsize=13, fontweight='bold')
    ax1.set_xlabel('Forecast Horizon (10-min steps)', fontsize=11)
    ax1.set_ylabel('MAE (MVA)', fontsize=11)
    ax1.legend(loc='best', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(left=0)

    # Panel 2: RMSE vs Horizon
    for i, model_name in enumerate(model_names):
        rmse_vals = [results[model_name]['point']['per_horizon'][h]['rmse'] for h in horizons]
        ax2.plot(horizons, rmse_vals, marker=markers[i], linestyle=linestyles[i],
                color=colors[i], linewidth=2, markersize=6, label=model_name)

    ax2.set_title('RMSE vs Forecast Horizon', fontsize=13, fontweight='bold')
    ax2.set_xlabel('Forecast Horizon (10-min steps)', fontsize=11)
    ax2.set_ylabel('RMSE (MVA)', fontsize=11)
    ax2.legend(loc='best', fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(left=0)

    # Panel 3: MAPE vs Horizon
    for i, model_name in enumerate(model_names):
        mape_vals = [results[model_name]['point']['per_horizon'][h]['mape'] for h in horizons]
        ax3.plot(horizons, mape_vals, marker=markers[i], linestyle=linestyles[i],
                color=colors[i], linewidth=2, markersize=6, label=model_name)

    ax3.set_title('MAPE vs Forecast Horizon', fontsize=13, fontweight='bold')
    ax3.set_xlabel('Forecast Horizon (10-min steps)', fontsize=11)
    ax3.set_ylabel('MAPE (%)', fontsize=11)
    ax3.legend(loc='best', fontsize=10)
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(left=0)

    # Panel 4: R² vs Horizon
    for i, model_name in enumerate(model_names):
        r2_vals = [results[model_name]['point']['per_horizon'][h]['r2'] for h in horizons]
        ax4.plot(horizons, r2_vals, marker=markers[i], linestyle=linestyles[i],
                color=colors[i], linewidth=2, markersize=6, label=model_name)

    ax4.set_title('R² Score vs Forecast Horizon', fontsize=13, fontweight='bold')
    ax4.set_xlabel('Forecast Horizon (10-min steps)', fontsize=11)
    ax4.set_ylabel('R²', fontsize=11)
    ax4.legend(loc='best', fontsize=10)
    ax4.grid(True, alpha=0.3)
    ax4.set_xlim(left=0)
    ax4.set_ylim(0, 1.05)

    plt.tight_layout()
    plt.savefig(outdir / 'model_comparison_horizons.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  - model_comparison_horizons.png")


def print_improvement_summary(results, baseline_name='Persistence'):
    """
    Print summary of improvements over baseline.

    Args:
        results: dict from compare_models()
        baseline_name: Name of baseline model for comparison (default: 'Persistence')
    """
    if baseline_name not in results:
        print(f"[WARNING] Baseline model '{baseline_name}' not found in results")
        return

    baseline = results[baseline_name]['point']['aggregate']

    print("\n" + "="*70)
    print(f"IMPROVEMENT OVER {baseline_name.upper()}")
    print("="*70)

    for model_name, model_results in results.items():
        if model_name == baseline_name:
            continue

        model_metrics = model_results['point']['aggregate']

        # Compute improvements
        mae_improvement = (baseline['mae'] - model_metrics['mae']) / baseline['mae'] * 100
        rmse_improvement = (baseline['rmse'] - model_metrics['rmse']) / baseline['rmse'] * 100
        mape_improvement = (baseline['mape'] - model_metrics['mape']) / baseline['mape'] * 100

        print(f"\n{model_name}:")
        print(f"  MAE:  {mae_improvement:+.1f}% ({model_metrics['mae']:.2f} vs {baseline['mae']:.2f})")
        print(f"  RMSE: {rmse_improvement:+.1f}% ({model_metrics['rmse']:.2f} vs {baseline['rmse']:.2f})")
        print(f"  MAPE: {mape_improvement:+.1f}% ({model_metrics['mape']:.2f} vs {baseline['mape']:.2f})")
        print(f"  R²:   {model_metrics['r2']:.3f} vs {baseline['r2']:.3f}")

    print("="*70)
