"""
Analysis Module

Analyzes model prediction errors to identify patterns and weaknesses:
- Errors by time of day (hour patterns)
- Errors by load level (low/medium/high load)
- Errors by load volatility (rate of change)
- Errors by day of week (weekday vs weekend patterns)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def analyze_median_prediction_errors(logs_path, outdir):
    """
    Analyze median (q50) prediction errors from backtest logs
    Investigates when and why median predictions are inaccurate by analyzing:
    - Errors by time of day (hour patterns)
    - Errors by load level (low/medium/high load)
    - Errors by load volatility (rate of change)
    - Errors by day of week (weekday vs weekend patterns)
    Creates comprehensive visualization showing error patterns to identify
    missing features or periods where the model struggles.
    Args:
        logs_path: Path to logs.csv file from backtest
        outdir: Output directory for analysis plots
    """

    # Ensure output directory exists
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Load logs
    print(f"Loading backtest logs from: {logs_path}")

    df = pd.read_csv(logs_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.set_index('timestamp', inplace=True)

    # Calculate median prediction error
    df['median_error'] = df['S_TOTAL'] - df['forecast_q50']
    df['median_abs_error'] = np.abs(df['median_error'])
    df['median_pct_error'] = (df['median_error'] / df['S_TOTAL']) * 100

    # Extract time features
    df['hour'] = df.index.hour
    df['day_of_week'] = df.index.dayofweek
    df['is_weekend'] = df['day_of_week'].isin([5, 6])

    # Calculate load volatility (rate of change)
    df['load_change'] = df['S_TOTAL'].diff().abs()
    df['load_volatility'] = df['load_change'].rolling(window=6, min_periods=1).mean()  # 1-hour window

    # Categorize load levels
    load_bins = [0, df['S_TOTAL'].quantile(0.33), df['S_TOTAL'].quantile(0.67), df['S_TOTAL'].max()]
    df['load_category'] = pd.cut(df['S_TOTAL'], bins=load_bins, labels=['Low', 'Medium', 'High'])

    # Categorize volatility
    vol_bins = [0, df['load_volatility'].quantile(0.33), df['load_volatility'].quantile(0.67), df['load_volatility'].max()]
    df['volatility_category'] = pd.cut(df['load_volatility'], bins=vol_bins, labels=['Low', 'Medium', 'High'])

    # Print summary statistics
    print("\n" + "="*60)
    print("MEDIAN PREDICTION ERROR ANALYSIS")
    print("="*60)
    print(f"Total Samples: {len(df):,}")
    print(f"Mean Absolute Error: {df['median_abs_error'].mean():.4f} MVA")
    print(f"Median Absolute Error: {df['median_abs_error'].median():.4f} MVA")
    print(f"95th Percentile Error: {df['median_abs_error'].quantile(0.95):.4f} MVA")
    print(f"Max Absolute Error: {df['median_abs_error'].max():.4f} MVA")
    print(f"Mean Percentage Error: {df['median_pct_error'].mean():.2f}%")

    # Create comprehensive visualization
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3)

    # 1. Error time series
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(df.index, df['median_error'], alpha=0.6, linewidth=0.5, color='darkred', label='Median Error')
    ax1.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax1.fill_between(df.index, 0, df['median_error'], where=(df['median_error']>0), alpha=0.3, color='red', label='Over-prediction')
    ax1.fill_between(df.index, 0, df['median_error'], where=(df['median_error']<0), alpha=0.3, color='blue', label='Under-prediction')
    ax1.set_xlabel('Time', fontsize=11)
    ax1.set_ylabel('Median Error (MVA)', fontsize=11)
    ax1.set_title('Median Prediction Error Time Series', fontsize=13, fontweight='bold')
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)

    # 2. Error by hour of day
    ax2 = fig.add_subplot(gs[1, 0])
    hourly_errors = df.groupby('hour')['median_abs_error'].agg(['mean', 'std'])
    ax2.bar(hourly_errors.index, hourly_errors['mean'], yerr=hourly_errors['std'], alpha=0.7, color='steelblue', capsize=3)
    ax2.set_xlabel('Hour of Day', fontsize=11)
    ax2.set_ylabel('Mean Absolute Error (MVA)', fontsize=11)
    ax2.set_title('Errors by Hour of Day', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_xticks(range(0, 24, 4))

    # Find worst hours
    worst_hours = hourly_errors['mean'].nlargest(3)

    print(f"\nWorst Hours (highest MAE):")

    for hour, mae in worst_hours.items():
        print(f"  Hour {hour:02d}:00 - MAE: {mae:.4f} MVA")

    # 3. Error by load level
    ax3 = fig.add_subplot(gs[1, 1])
    load_errors = df.groupby('load_category', observed=True)['median_abs_error'].agg(['mean', 'std', 'count'])
    colors_load = ['lightgreen', 'orange', 'darkred']
    ax3.bar(range(len(load_errors)), load_errors['mean'], yerr=load_errors['std'], alpha=0.7, color=colors_load, capsize=3)
    ax3.set_xticks(range(len(load_errors)))
    ax3.set_xticklabels(load_errors.index)
    ax3.set_xlabel('Load Level', fontsize=11)
    ax3.set_ylabel('Mean Absolute Error (MVA)', fontsize=11)
    ax3.set_title('Errors by Load Level', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')

    print(f"\nErrors by Load Level:")

    for cat, mae in load_errors['mean'].items():
        count = load_errors.loc[cat, 'count']
        print(f"  {cat} Load: MAE={mae:.4f} MVA (n={count:,})")

    # 4. Error by volatility
    ax4 = fig.add_subplot(gs[1, 2])
    vol_errors = df.dropna(subset=['volatility_category']).groupby('volatility_category', observed=True)['median_abs_error'].agg(['mean', 'std', 'count'])
    colors_vol = ['lightblue', 'yellow', 'red']
    ax4.bar(range(len(vol_errors)), vol_errors['mean'], yerr=vol_errors['std'], alpha=0.7, color=colors_vol, capsize=3)
    ax4.set_xticks(range(len(vol_errors)))
    ax4.set_xticklabels(vol_errors.index)
    ax4.set_xlabel('Load Volatility', fontsize=11)
    ax4.set_ylabel('Mean Absolute Error (MVA)', fontsize=11)
    ax4.set_title('Errors by Volatility', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='y')

    print(f"\nErrors by Volatility:")
    for cat, mae in vol_errors['mean'].items():
        count = vol_errors.loc[cat, 'count']
        print(f"  {cat} Volatility: MAE={mae:.4f} MVA (n={count:,})")

    # 5. Error by day of week
    ax5 = fig.add_subplot(gs[2, 0])
    dow_errors = df.groupby('day_of_week')['median_abs_error'].mean()
    dow_labels = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    colors_dow = ['steelblue']*5 + ['orange', 'orange']
    ax5.bar(range(7), dow_errors.values, alpha=0.7, color=colors_dow)
    ax5.set_xticks(range(7))
    ax5.set_xticklabels(dow_labels)
    ax5.set_xlabel('Day of Week', fontsize=11)
    ax5.set_ylabel('Mean Absolute Error (MVA)', fontsize=11)
    ax5.set_title('Errors by Day of Week', fontsize=12, fontweight='bold')
    ax5.grid(True, alpha=0.3, axis='y')

    # 6. Error distribution
    ax6 = fig.add_subplot(gs[2, 1])
    ax6.hist(df['median_error'], bins=50, alpha=0.7, color='purple', edgecolor='black')
    ax6.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Perfect Prediction')
    ax6.axvline(x=df['median_error'].mean(), color='green', linestyle='--', linewidth=2, label=f'Mean Error: {df["median_error"].mean():.3f}')
    ax6.set_xlabel('Median Error (MVA)', fontsize=11)
    ax6.set_ylabel('Frequency', fontsize=11)
    ax6.set_title('Error Distribution', fontsize=12, fontweight='bold')
    ax6.legend(loc='best')
    ax6.grid(True, alpha=0.3, axis='y')

    # 7. Scatter: Actual vs Predicted
    ax7 = fig.add_subplot(gs[2, 2])
    sample_indices = np.random.choice(len(df), size=min(2000, len(df)), replace=False)
    df_sample = df.iloc[sample_indices]
    scatter = ax7.scatter(df_sample['S_TOTAL'], df_sample['forecast_q50'], c=df_sample['median_abs_error'], cmap='RdYlGn_r', alpha=0.5, s=10)
    ax7.plot([df['S_TOTAL'].min(), df['S_TOTAL'].max()], [df['S_TOTAL'].min(), df['S_TOTAL'].max()], 'k--', linewidth=2, label='Perfect Prediction')
    ax7.set_xlabel('Actual Load (MVA)', fontsize=11)
    ax7.set_ylabel('Predicted Load (MVA)', fontsize=11)
    ax7.set_title('Actual vs Predicted (colored by error)', fontsize=12, fontweight='bold')
    ax7.legend(loc='best')
    ax7.grid(True, alpha=0.3)
    cbar = plt.colorbar(scatter, ax=ax7)
    cbar.set_label('Absolute Error (MVA)', fontsize=9)

    plt.suptitle('Median Prediction Error Analysis', fontsize=16, fontweight='bold', y=0.995)

    # Save plot
    plt.savefig(outdir / 'median_error_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\nMedian error analysis plot saved to: {outdir / 'median_error_analysis.png'}")

    # Generate insights
    print("\n" + "="*60)
    print("KEY INSIGHTS")
    print("="*60)

    # Check for bias
    mean_error = df['median_error'].mean()
    if abs(mean_error) > 0.5:
        print(f"[!] BIAS DETECTED: Mean error = {mean_error:.3f} MVA")
        if mean_error > 0:
            print("  -> Model tends to UNDER-predict (actual > predicted)")
        else:
            print("  -> Model tends to OVER-predict (actual < predicted)")
    else:
        print(f"[OK] Low bias: Mean error = {mean_error:.3f} MVA")

    # Check volatility impact
    vol_ratio = vol_errors.loc['High', 'mean'] / vol_errors.loc['Low', 'mean'] if 'High' in vol_errors.index and 'Low' in vol_errors.index else 1.0
    if vol_ratio > 1.5:
        print(f"[!] HIGH VOLATILITY IMPACT: Errors {vol_ratio:.1f}x higher during volatile periods")
        print("  -> Consider adding more volatility-related features (rolling std, rate of change)")

    # Check load level impact
    load_ratio = load_errors.loc['High', 'mean'] / load_errors.loc['Low', 'mean'] if 'High' in load_errors.index and 'Low' in load_errors.index else 1.0
    if load_ratio > 1.3:
        print(f"[!] HIGH LOAD IMPACT: Errors {load_ratio:.1f}x higher at high loads")
        print("  -> Model struggles with peak load prediction")

    # Check time of day patterns
    hour_variance = hourly_errors['mean'].std()
    if hour_variance > hourly_errors['mean'].mean() * 0.3:
        print(f"[!] STRONG TIME-OF-DAY PATTERNS: Hour variance = {hour_variance:.3f}")
        print("  -> Consider enhancing temporal features (hour embeddings, interaction terms)")

    print("="*60 + "\n")

    return df
