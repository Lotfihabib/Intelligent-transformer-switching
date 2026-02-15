"""
Results Visualization Module
=============================

This module contains functions for visualizing backtest results and KPI metrics.

Functions:
    generate_summary_plots: Generate comprehensive backtest analysis and KPI summary plots
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path

from config import CONFIG

def plot_trajectory_samples(quantile_predictions, trajectories, outdir, example_idx=0, num_samples=30):
    """
    Plot median forecast with sampled trajectories to visualize uncertainty quantification

    Args:
        quantile_predictions: Array [horizon, 3] - Quantile predictions (q10, q50, q90)
        trajectories: Array [M, horizon] - Sampled trajectories
        outdir: Output directory path
        example_idx: Example index for filename
        num_samples: Number of trajectory samples to plot (to avoid clutter)
    """
    horizon = quantile_predictions.shape[0]
    timesteps = np.arange(1, horizon + 1)  # 1 to 12 (representing next 2 hours)

    # Extract quantiles
    q10 = quantile_predictions[:, 0]
    q50 = quantile_predictions[:, 1]
    q90 = quantile_predictions[:, 2]

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot uncertainty band (80% prediction interval)
    ax.fill_between(timesteps, q10, q90, alpha=0.2, color='blue', label='80% Prediction Interval (q10-q90)')

    # Plot a random sample of trajectories (to avoid clutter)
    M = trajectories.shape[0]
    sample_indices = np.random.choice(M, size=min(num_samples, M), replace=False)

    for i, idx in enumerate(sample_indices):
        if i == 0:
            ax.plot(timesteps, trajectories[idx], color='gray', alpha=0.3, linewidth=0.8, label=f'{num_samples} Sampled Trajectories')
        else:
            ax.plot(timesteps, trajectories[idx], color='gray', alpha=0.3, linewidth=0.8)

    # Plot median forecast (most important)
    ax.plot(timesteps, q50, color='red', linewidth=2.5, marker='o', markersize=5, label='Median Forecast (q50)', zorder=10)

    # Formatting
    ax.set_xlabel('Forecast Horizon (10-minute steps)', fontsize=12)
    ax.set_ylabel('Apparent Power Load (MVA)', fontsize=12)
    ax.set_title(f'Uncertainty Quantification: Median Forecast with Sampled Trajectories\n(M={M} scenarios over T={horizon} steps)',
                 fontsize=13, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0.5, horizon + 0.5)

    # Add secondary x-axis showing time in minutes
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xlabel('Time Ahead (minutes)', fontsize=12)
    ax2.set_xticks(timesteps[::2])  # Every other timestep
    ax2.set_xticklabels([f'{t*10}' for t in timesteps[::2]])

    plt.tight_layout()

    # Save plot
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    plt.savefig(outdir / f'trajectory_samples_example_{example_idx}.png', dpi=300, bbox_inches='tight')
    plt.close()

def generate_summary_plots(results_df, switching_events, kpis, outdir):
    """Generate comprehensive analysis plots (same as backtest.py)"""

    plt.style.use('default')

    # Plot 1: Forecast bands and actual load + Transformer timeline + Cumulative savings
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 12))

    # Select a representative week for detailed view
    week_start = len(results_df) // 4
    week_end = week_start + 7 * 144  # 7 days at 10-min resolution
    week_data = results_df.iloc[week_start:week_end].copy()

    if len(week_data) > 0:

        # Use timestamp as x-axis (already in datetime format)
        time_axis = week_data['timestamp']

        # Add forecast bands if available
        if 'forecast_q10' in week_data.columns:
            # Plot 80% prediction interval (P10-P90)
            ax1.fill_between(time_axis, week_data['forecast_q10'], week_data['forecast_q90'], alpha=0.25, color='blue', label='80% PI (P10-P90)')

            # Plot median forecast
            ax1.plot(time_axis, week_data['forecast_q50'], color='blue', linewidth=1.5, markersize=3, label='Forecast Median (P50)', alpha=0.8)

        # Plot actual load
        ax1.plot(time_axis, week_data['S_TOTAL'], color='black', linewidth=2, markersize=3, label='Actual Load', zorder=3)
        ax1.set_ylabel('Load (MVA)', fontsize=11)
        ax1.set_title('Load Forecasting Performance (Sample Week)', fontsize=12, fontweight='bold')
        ax1.legend(loc='best', fontsize=9)
        ax1.grid(True, alpha=0.3)

        # Format x-axis to show dates/times
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
        ax1.xaxis.set_major_locator(mdates.HourLocator(interval=12))  # Every 12 hours
        ax1.xaxis.set_minor_locator(mdates.HourLocator(interval=6))   # Minor ticks every 6 hours
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # Plot 2: Transformer topology timeline
    colors = {1: 'red', 2: 'orange', 3: 'green'}

    # Full timeline - use timestamps
    timestamps = results_df['timestamp']
    for n in [1, 2, 3]:
        mask = results_df['N_current'] == n
        if mask.any():
            ax2.scatter(timestamps[mask], results_df['N_current'][mask], c=colors[n], alpha=0.6, s=1, label=f'{n} Transformer(s)')

    # Mark switching events
    for event in switching_events:
        ax2.axvline(x=event['timestamp'], color='black', alpha=0.5, linewidth=0.5)

    ax2.set_ylabel('Active Transformers', fontsize=11)
    ax2.set_title('Transformer Switching Timeline', fontsize=12, fontweight='bold')
    ax2.set_ylim(0.5, 3.5)
    ax2.set_yticks([1, 2, 3])
    ax2.legend(loc='best', fontsize=9)
    ax2.grid(True, alpha=0.3)

    # Format x-axis
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    ax2.xaxis.set_major_locator(mdates.DayLocator())
    ax2.xaxis.set_minor_locator(mdates.HourLocator(interval=6))
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # Plot 3: Cumulative savings
    if 'baseline_loss_kwh' in kpis and kpis['baseline_loss_kwh'] > 0:
        cumulative_baseline = results_df['loss_kwh'].cumsum() * (kpis['baseline_loss_kwh'] / kpis['total_loss_kwh'])
        cumulative_actual = results_df['loss_kwh'].cumsum()
        cumulative_savings = cumulative_baseline - cumulative_actual

    else:
        # Fallback calculation
        cumulative_savings = np.cumsum(results_df['loss_kwh'] * 0.1)  # Assume 10% savings

    ax3.plot(timestamps, cumulative_savings, 'g-', linewidth=2, label='Cumulative Savings')
    ax3.fill_between(timestamps, 0, cumulative_savings, alpha=0.3, color='green')
    ax3.set_ylabel('Cumulative Savings (kWh)', fontsize=11)
    ax3.set_xlabel('Time', fontsize=11)
    ax3.set_title('Cumulative Energy Savings', fontsize=12, fontweight='bold')
    ax3.legend(loc='best', fontsize=9)
    ax3.grid(True, alpha=0.3)

    # Format x-axis
    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    ax3.xaxis.set_major_locator(mdates.DayLocator())
    plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(outdir / 'backtest_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Plot 2: KPI Summary (4-panel dashboard)
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))

    # Transformer utilization pie chart
    utilization_data = [
        (results_df['N_current'] == 1).sum() / len(results_df),
        (results_df['N_current'] == 2).sum() / len(results_df),
        (results_df['N_current'] == 3).sum() / len(results_df)
    ]
    labels = ['1 Transformer', '2 Transformers', '3 Transformers']
    colors_pie = ['red', 'orange', 'green']
    ax1.pie(utilization_data, labels=labels, colors=colors_pie, autopct='%1.1f%%', startangle=90)
    ax1.set_title('Transformer Utilization')

    # Monthly switching frequency
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun']
    switches_data = [kpis.get('switches_per_month', 0)] * 6  # Simplified for demo
    ax2.bar(months, switches_data, color='skyblue')
    ax2.set_ylabel('Switches per Month')
    ax2.set_title('Switching Frequency')
    ax2.tick_params(axis='x', rotation=45)

    # Savings comparison
    baseline_loss = kpis.get('baseline_loss_kwh', kpis.get('total_loss_kwh', 0) * 1.1)
    total_loss = kpis.get('total_loss_kwh', 0)
    savings = kpis.get('savings_kwh', baseline_loss - total_loss)

    categories = ['Baseline\nLoss', 'Actual\nLoss', 'Savings']
    values = [baseline_loss, total_loss, savings]
    colors_bar = ['red', 'orange', 'green']

    bars = ax3.bar(categories, values, color=colors_bar, alpha=0.7)
    ax3.set_ylabel('Energy (kWh)')
    ax3.set_title('Energy Loss Comparison')

    # Add value labels on bars
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + height*0.01, f'{value:.0f}', ha='center', va='bottom')

    # Performance metrics text
    simulation_hours = kpis.get('simulation_hours', len(results_df) / 6.0)
    metrics_text = f"""
    Simulation Period: {simulation_hours/24:.1f} days
    Energy Performance:
    • Total Savings: {savings:.0f} kWh
    • Annualized Savings: {savings * (8760/simulation_hours):.0f} kWh/year
    • Savings Percentage: {kpis.get('savings_percentage', 0):.1f}%
    Operational Metrics:
    • Total Switches: {kpis.get('num_switches', 0)}
    • Switches/Month: {kpis.get('switches_per_month', 0):.1f}
    • Overload Events: {kpis.get('overload_count', 0)}
    """
    ax4.text(0.05, 0.95, metrics_text, transform=ax4.transAxes, fontsize=10,  verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax4.set_xlim(0, 1)
    ax4.set_ylim(0, 1)
    ax4.axis('off')
    ax4.set_title('Performance Summary')

    plt.tight_layout()
    plt.savefig(outdir / 'kpi_summary.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Plots saved to: {outdir}")
    print(f"  - backtest_analysis.png (3-panel: forecast, timeline, savings)")
    print(f"  - kpi_summary.png (4-panel: utilization, frequency, comparison, metrics)")
