"""
Data Analysis Visualization Module
====================================

This module contains functions for exploratory data analysis (EDA) visualizations
of power system load data.

Functions:
    plot_load_duration_curve: Load Duration Curve (LDC) analysis
    plot_temporal_heatmap: Day-of-week vs Hour-of-day heatmap
    plot_distribution_analysis: Distribution with KDE and Gaussian fit
    plot_seasonal_decomposition: STL decomposition (Trend, Seasonal, Residual)
    plot_power_factor_analysis: Reactive power vs Power Factor scatter
    generate_all_analysis_plots: Generate all 5 analysis plots
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats
from scipy.signal import find_peaks
from statsmodels.tsa.seasonal import STL
import warnings

warnings.filterwarnings('ignore')


def plot_load_duration_curve(df, outdir, column='S_TOTAL', bq12=38.6, bq23=66.9):
    """
    Figure 1: Augmented Load Duration Curve (LDC) with Efficiency Zones

    A line plot showing the sorted load data in descending order with colored zones
    indicating optimal transformer configurations based on exact switching breakpoints.

    Args:
        df: DataFrame with load data
        outdir: Output directory path
        column: Column name for load data (default: 'S_TOTAL')
        bq12: Breakpoint between 1-TR and 2-TR (MVA) (default: 38.6)
        bq23: Breakpoint between 2-TR and 3-TR (MVA) (default: 66.9)
    """
    # Extract load data and sort in descending order
    load_data = df[column].dropna().values
    sorted_load = np.sort(load_data)[::-1]

    # Calculate time percentage (0-100%)
    n = len(sorted_load)
    time_percentage = np.linspace(0, 100, n)

    # Calculate zone boundaries based on exact breakpoints
    # Red zone: S > bq23 (3-TR required)
    # Yellow zone: bq12 < S <= bq23 (2-TR optimal)
    # Green zone: S <= bq12 (1-TR optimal)

    # Find the percentage points where load crosses breakpoints
    idx_bq23 = np.searchsorted(-sorted_load, -bq23)  # First index where load <= bq23
    idx_bq12 = np.searchsorted(-sorted_load, -bq12)  # First index where load <= bq12

    pct_bq23 = (idx_bq23 / n) * 100 if idx_bq23 < n else 0
    pct_bq12 = (idx_bq12 / n) * 100 if idx_bq12 < n else 0

    # Calculate actual zone percentages
    red_zone_pct = pct_bq23
    yellow_zone_pct = pct_bq12 - pct_bq23
    green_zone_pct = 100 - pct_bq12

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 7))

    # Plot load duration curve
    ax.plot(time_percentage, sorted_load, color='black', linewidth=2.5,
           label='Load Duration Curve', zorder=5)

    # Add Colored Efficiency Zones with exact breakpoints
    if red_zone_pct > 0:
        ax.axvspan(0, pct_bq23, color='red', alpha=0.15,
                  label=f'3-TR Required (S > {bq23} MVA)', zorder=1)

    ax.axvspan(pct_bq23, pct_bq12, color='orange', alpha=0.15,
              label=f'2-TR Optimal ({bq12} < S ≤ {bq23} MVA)', zorder=1)

    if pct_bq12 < 100:
        ax.axvspan(pct_bq12, 100, color='green', alpha=0.15,
                  label=f'1-TR Optimal (S ≤ {bq12} MVA)', zorder=1)

    # Add horizontal lines at breakpoints
    ax.axhline(bq23, color='red', linestyle='--', alpha=0.6, linewidth=2,
              label=f'bq₂₃ = {bq23} MVA', zorder=4)
    ax.axhline(bq12, color='green', linestyle='--', alpha=0.6, linewidth=2,
              label=f'bq₁₂ = {bq12} MVA', zorder=4)

    # Add vertical lines at zone boundaries
    if red_zone_pct > 0:
        ax.axvline(pct_bq23, color='red', linestyle=':', alpha=0.5, linewidth=1.5)
    ax.axvline(pct_bq12, color='green', linestyle=':', alpha=0.5, linewidth=1.5)

    # Calculate statistics
    mean_load = np.mean(sorted_load)
    median_load = np.median(sorted_load)
    max_load = np.max(sorted_load)
    min_load = np.min(sorted_load)

    # Add statistics box
    '''stats_text = f'Load Statistics:\n'
    stats_text += f'Max: {max_load:.2f} MVA\n'
    stats_text += f'Mean: {mean_load:.2f} MVA\n'
    stats_text += f'Median: {median_load:.2f} MVA\n'
    stats_text += f'Min: {min_load:.2f} MVA\n'
    stats_text += f'\nBreakpoints:\n'
    #stats_text += f'bq23: {bq23} MVA\n'
    #stats_text += f'bq12: {bq12} MVA\n'
    #stats_text += f'\nZone Distribution:\n'
    #stats_text += f'High (3-TR): {red_zone_pct:.1f}%\n'
    #stats_text += f'Med (2-TR): {yellow_zone_pct:.1f}%\n'
    #stats_text += f'Low (1-TR): {green_zone_pct:.1f}%

    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
           verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9),
           fontsize=10, family='monospace')'''

    # Formatting
    ax.set_xlabel('Time Duration (%)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Apparent Power (MVA)', fontsize=13, fontweight='bold')
    ax.set_title('Augmented Load Duration Curve with Efficiency Regimes\n(Based on Exact Switching Breakpoints)',
                fontsize=14, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='upper right', fontsize=10)
    ax.set_xlim(0, 100)
    ax.set_ylim(min_load * 0.95, max_load * 1.05)

    plt.tight_layout()

    # Save plot
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    plt.savefig(outdir / 'load_duration_curve.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"[OK] Load Duration Curve saved to {outdir / 'load_duration_curve.png'}")


def plot_temporal_heatmap(df, outdir, column='S_TOTAL'):
    """
    Figure 2: Temporal Heatmap (Day-of-Week vs Hour-of-Day)

    A 2D heatmap showing mean load for each combination of day-of-week and hour-of-day.

    Args:
        df: DataFrame with datetime index and load data
        outdir: Output directory path
        column: Column name for load data (default: 'S_TOTAL')
    """
    # Ensure datetime index
    if not isinstance(df.index, pd.DatetimeIndex):
        if 'timestamp' in df.columns:
            df = df.set_index('timestamp')
        else:
            raise ValueError("DataFrame must have datetime index or 'timestamp' column")

    # Extract day of week and hour
    df_temp = df.copy()
    df_temp['day_of_week'] = df_temp.index.dayofweek  # 0=Monday, 6=Sunday
    df_temp['hour'] = df_temp.index.hour

    # Create pivot table (Day of Week x Hour)
    heatmap_data = df_temp.pivot_table(values=column,
                                       index='day_of_week',
                                       columns='hour',
                                       aggfunc='mean')

    # Create figure
    fig, ax = plt.subplots(figsize=(14, 6))

    # Create heatmap
    im = ax.imshow(heatmap_data.values, aspect='auto', cmap='YlOrRd', interpolation='nearest')

    # Set ticks and labels
    ax.set_xticks(np.arange(24))
    ax.set_xticklabels([f'{h:02d}:00' for h in range(24)], rotation=45, ha='right')

    day_labels = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    ax.set_yticks(np.arange(7))
    ax.set_yticklabels(day_labels)

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, orientation='vertical', pad=0.02)
    cbar.set_label('Mean Apparent Power (MVA)', fontsize=12, fontweight='bold')

    # Add text annotations for values
    for i in range(len(day_labels)):
        for j in range(24):
            value = heatmap_data.values[i, j]
            if not np.isnan(value):
                text_color = 'white' if value > heatmap_data.values.max() * 0.6 else 'black'
                ax.text(j, i, f'{value:.1f}', ha='center', va='center',
                       color=text_color, fontsize=7, fontweight='bold')

    # Formatting
    ax.set_xlabel('Hour of Day', fontsize=13, fontweight='bold')
    ax.set_ylabel('Day of Week', fontsize=13, fontweight='bold')
    ax.set_title('Temporal Load Heatmap\nMean Apparent Power by Day-of-Week and Hour-of-Day',
                fontsize=14, fontweight='bold', pad=20)

    plt.tight_layout()

    # Save plot
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    plt.savefig(outdir / 'temporal_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"[OK] Temporal Heatmap saved to {outdir / 'temporal_heatmap.png'}")


def plot_distribution_analysis(df, outdir, column='S_TOTAL'):
    """
    Figure 3: Distribution Analysis with KDE and Normal Fit

    Histogram of load data with overlaid Kernel Density Estimate (KDE)
    and fitted Gaussian (Normal) distribution.

    Args:
        df: DataFrame with load data
        outdir: Output directory path
        column: Column name for load data (default: 'S_TOTAL')
    """
    # Extract load data
    load_data = df[column].dropna().values

    # Fit normal distribution
    mu, sigma = stats.norm.fit(load_data)

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 7))

    # Plot histogram
    n, bins, patches = ax.hist(load_data, bins=50, density=True, alpha=0.6,
                               color='skyblue', edgecolor='black', label='Load Data Histogram')

    # Plot KDE
    from scipy.stats import gaussian_kde
    kde = gaussian_kde(load_data)
    x_range = np.linspace(load_data.min(), load_data.max(), 1000)
    kde_values = kde(x_range)
    ax.plot(x_range, kde_values, color='blue', linewidth=2.5,
           label='Kernel Density Estimate (KDE)', zorder=5)

    # Plot Normal fit
    normal_fit = stats.norm.pdf(x_range, mu, sigma)
    ax.plot(x_range, normal_fit, color='red', linewidth=2.5, linestyle='--',
           label=f'Normal Fit (μ={mu:.2f}, σ={sigma:.2f})', zorder=5)

    # Calculate statistics
    skewness = stats.skew(load_data)
    kurtosis = stats.kurtosis(load_data)
    median = np.median(load_data)

    # Shapiro-Wilk test for normality (sample up to 5000 points)
    sample_size = min(5000, len(load_data))
    sample_data = np.random.choice(load_data, size=sample_size, replace=False)
    shapiro_stat, shapiro_p = stats.shapiro(sample_data)

    '''# Add statistics box
    stats_text = f'Distribution Statistics:\n'
    stats_text += f'Mean (μ): {mu:.2f} MVA\n'
    stats_text += f'Std Dev (σ): {sigma:.2f} MVA\n'
    stats_text += f'Median: {median:.2f} MVA\n'
    stats_text += f'Skewness: {skewness:.3f}\n'
    stats_text += f'Kurtosis: {kurtosis:.3f}\n'
    stats_text += f'Shapiro-Wilk p-value: {shapiro_p:.4f}\n'
    if shapiro_p < 0.05:
        stats_text += '(Non-normal distribution)'
    else:
        stats_text += '(Normal distribution)'

    ax.text(0.98, 0.98, stats_text, transform=ax.transAxes,
           verticalalignment='top', horizontalalignment='right',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9),
           fontsize=10, family='monospace')'''

    # Formatting
    ax.set_xlabel('Apparent Power (MVA)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Probability Density', fontsize=13, fontweight='bold')
    ax.set_title('Load Distribution Analysis\nHistogram with KDE and Normal Fit',
                fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='upper left', fontsize=11)
    ax.grid(True, alpha=0.3, linestyle='--', axis='y')

    plt.tight_layout()

    # Save plot
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    plt.savefig(outdir / 'distribution_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"[OK] Distribution Analysis saved to {outdir / 'distribution_analysis.png'}")


def plot_seasonal_decomposition(df, outdir, column='S_TOTAL', period=144):
    """
    Figure 4: Seasonal Decomposition (STL Decomposition)

    4-panel subplot showing Original time series, Trend, Seasonal component, and Residual.
    Uses STL (Seasonal and Trend decomposition using Loess).

    Args:
        df: DataFrame with datetime index and load data
        outdir: Output directory path
        column: Column name for load data (default: 'S_TOTAL')
        period: Seasonal period in timesteps (default: 144 = 1 day at 10-min resolution)
    """
    # Ensure datetime index
    if not isinstance(df.index, pd.DatetimeIndex):
        if 'timestamp' in df.columns:
            df = df.set_index('timestamp')
        else:
            raise ValueError("DataFrame must have datetime index or 'timestamp' column")

    # Extract load data
    load_series = df[column].dropna()

    # Take a representative sample (e.g., 1 month) for clearer visualization
    # If data is longer than 30 days, use first 30 days
    max_samples = 30 * 144  # 30 days
    if len(load_series) > max_samples:
        load_series = load_series.iloc[:max_samples]

    # Store original datetime index for plotting
    original_index = load_series.index

    # Create series with integer index to avoid statsmodels frequency detection issues
    # STL will use only the explicit period parameter
    values_for_stl = pd.Series(load_series.values)

    # Perform STL decomposition
    # Pass period explicitly - statsmodels requires this when no frequency is in index
    try:
        stl = STL(values_for_stl, period=period, seasonal=period+1, robust=True)
        result = stl.fit()
    except Exception as e:
        # Fallback: try with just seasonal parameter (for older statsmodels versions)
        stl = STL(values_for_stl, seasonal=period, robust=True)
        result = stl.fit()

    # Reconstruct results with original datetime index for plotting
    trend_series = pd.Series(result.trend.values, index=original_index)
    seasonal_series = pd.Series(result.seasonal.values, index=original_index)
    resid_series = pd.Series(result.resid.values, index=original_index)

    # Create figure with 4 subplots
    fig, axes = plt.subplots(4, 1, figsize=(14, 12))

    # Plot 1: Original time series
    axes[0].plot(load_series.index, load_series.values, color='black', linewidth=1.2)
    axes[0].set_ylabel('Load (MVA)', fontsize=11, fontweight='bold')
    axes[0].set_title('Original Time Series', fontsize=12, fontweight='bold')
    axes[0].grid(True, alpha=0.3)

    # Plot 2: Trend component
    axes[1].plot(trend_series.index, trend_series.values, color='blue', linewidth=1.5)
    axes[1].set_ylabel('Trend (MVA)', fontsize=11, fontweight='bold')
    axes[1].set_title('Trend Component', fontsize=12, fontweight='bold')
    axes[1].grid(True, alpha=0.3)

    # Plot 3: Seasonal component
    axes[2].plot(seasonal_series.index, seasonal_series.values, color='green', linewidth=1.2)
    axes[2].set_ylabel('Seasonal (MVA)', fontsize=11, fontweight='bold')
    axes[2].set_title(f'Seasonal Component', fontsize=12, fontweight='bold')
    #(Period = {period} steps = {period*10/60:.1f} hours)
    axes[2].grid(True, alpha=0.3)

    # Plot 4: Residual component
    axes[3].plot(resid_series.index, resid_series.values, color='red', linewidth=0.8, alpha=0.7)
    axes[3].set_ylabel('Residual (MVA)', fontsize=11, fontweight='bold')
    axes[3].set_xlabel('Time', fontsize=12, fontweight='bold')
    axes[3].set_title('Residual (Noise)', fontsize=12, fontweight='bold')
    axes[3].grid(True, alpha=0.3)
    axes[3].axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)

    # Calculate decomposition statistics
    trend_strength = 1 - (np.var(result.resid) / np.var(result.trend + result.resid))
    seasonal_strength = 1 - (np.var(result.resid) / np.var(result.seasonal + result.resid))

    # Add overall statistics box
    '''stats_text = f'Decomposition Strength:\n'
    stats_text += f'Trend: {trend_strength:.3f}\n'
    stats_text += f'Seasonal: {seasonal_strength:.3f}\n'
    stats_text += f'Period: {period*10} min

    fig.text(0.99, 0.99, stats_text, transform=fig.transFigure,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9),
            fontsize=10, family='monospace')'''

    # Main title
    fig.suptitle('STL Seasonal Decomposition\nTrend, Seasonal, and Residual Components',
                fontsize=15, fontweight='bold', y=0.995)

    plt.tight_layout(rect=[0, 0, 1, 0.99])

    # Save plot
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    plt.savefig(outdir / 'seasonal_decomposition.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"[OK] Seasonal Decomposition saved to {outdir / 'seasonal_decomposition.png'}")


def plot_power_factor_analysis(df, outdir):
    """
    Figure 5: Reactive Power vs Power Factor Scatter

    Scatter plot showing relationship between Active Power (P) and Power Factor (PF).
    Power Factor = P / S, where S is apparent power.

    Args:
        df: DataFrame with P_TOTAL, Q_TOTAL, and S_TOTAL columns
        outdir: Output directory path
    """
    # Check required columns
    required_cols = ['P_TOTAL', 'S_TOTAL']
    missing_cols = [col for col in required_cols if col not in df.columns]

    if missing_cols:
        print(f"[WARNING] Missing columns {missing_cols} for power factor analysis. Skipping this plot.")
        return

    # Calculate power factor
    df_temp = df.copy()
    df_temp['power_factor'] = df_temp['P_TOTAL'] / df_temp['S_TOTAL']

    # Remove invalid values (PF should be between 0 and 1)
    df_temp = df_temp[(df_temp['power_factor'] >= 0) & (df_temp['power_factor'] <= 1)]
    df_temp = df_temp.dropna(subset=['P_TOTAL', 'power_factor'])

    # If Q_TOTAL is available, use it for color coding
    if 'Q_TOTAL' in df.columns:
        color_data = df_temp['Q_TOTAL']
        color_label = 'Reactive Power Q (MVAr)'
        cmap = 'viridis'
    else:
        # Use density-based coloring
        from scipy.stats import gaussian_kde
        xy = np.vstack([df_temp['P_TOTAL'], df_temp['power_factor']])
        z = gaussian_kde(xy)(xy)
        color_data = z
        color_label = 'Point Density'
        cmap = 'YlOrRd'

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))

    # Create scatter plot
    scatter = ax.scatter(df_temp['P_TOTAL'], df_temp['power_factor'],
                        c=color_data, cmap=cmap, alpha=0.6, s=20, edgecolors='none')

    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax, orientation='vertical', pad=0.02)
    cbar.set_label(color_label, fontsize=12, fontweight='bold')

    # Add ideal power factor line (PF = 1.0)
    ax.axhline(y=1.0, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Ideal PF = 1.0')

    # Add typical power factor thresholds
    ax.axhline(y=0.95, color='orange', linestyle=':', linewidth=1.5, alpha=0.5, label='Good PF = 0.95')
    ax.axhline(y=0.85, color='yellow', linestyle=':', linewidth=1.5, alpha=0.5, label='Acceptable PF = 0.85')

    # Calculate statistics
    mean_pf = df_temp['power_factor'].mean()
    median_pf = df_temp['power_factor'].median()
    min_pf = df_temp['power_factor'].min()
    max_pf = df_temp['power_factor'].max()
    std_pf = df_temp['power_factor'].std()

    # Count samples in different PF ranges
    excellent_pf = (df_temp['power_factor'] >= 0.95).sum() / len(df_temp) * 100
    good_pf = ((df_temp['power_factor'] >= 0.85) & (df_temp['power_factor'] < 0.95)).sum() / len(df_temp) * 100
    poor_pf = (df_temp['power_factor'] < 0.85).sum() / len(df_temp) * 100

    # Add statistics box
    stats_text = f'Power Factor Statistics:\n'
    stats_text += f'Mean PF: {mean_pf:.3f}\n'
    stats_text += f'Median PF: {median_pf:.3f}\n'
    stats_text += f'Std Dev: {std_pf:.3f}\n'
    stats_text += f'Range: [{min_pf:.3f}, {max_pf:.3f}]\n'
    stats_text += f'\nPF Distribution:\n'
    stats_text += f'Excellent (≥0.95): {excellent_pf:.1f}%\n'
    stats_text += f'Good (0.85-0.95): {good_pf:.1f}%\n'
    stats_text += f'Poor (<0.85): {poor_pf:.1f}%'

    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
           verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9),
           fontsize=10, family='monospace')

    # Formatting
    ax.set_xlabel('Active Power P (MW)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Power Factor (PF = P/S)', fontsize=13, fontweight='bold')
    ax.set_title('Power Factor Analysis\nActive Power vs Power Factor Relationship',
                fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_ylim(max(0, min_pf - 0.05), min(1.1, max_pf + 0.05))

    plt.tight_layout()

    # Save plot
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    plt.savefig(outdir / 'power_factor_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"[OK] Power Factor Analysis saved to {outdir / 'power_factor_analysis.png'}")


def plot_yearly_seasonality(df, outdir, column='S_TOTAL'):
    """
    Figure 6: Yearly Seasonality Analysis

    Shows monthly patterns and variations in load throughout the year using
    boxplots and trend lines.

    Args:
        df: DataFrame with datetime index and load data
        outdir: Output directory path
        column: Column name for load data (default: 'S_TOTAL')
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Ensure datetime index
    if not isinstance(df.index, pd.DatetimeIndex):
        if 'timestamp' in df.columns:
            df = df.set_index('timestamp')
        else:
            raise ValueError("DataFrame must have datetime index or 'timestamp' column")

    # Extract month information
    df_temp = df.copy()
    df_temp['Month'] = df_temp.index.month
    df_temp['Month_Name'] = df_temp.index.strftime('%b')

    # Create figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

    # ========== Subplot 1: Monthly Boxplots ==========
    # Prepare data for boxplot
    monthly_data = [df_temp[df_temp['Month'] == month][column].dropna().values
                    for month in range(1, 13)]
    month_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

    # Create boxplot
    bp = ax1.boxplot(monthly_data, labels=month_labels, patch_artist=True,
                     showfliers=False,  # Don't show outliers for cleaner view
                     medianprops=dict(color='red', linewidth=2),
                     boxprops=dict(facecolor='lightblue', alpha=0.7),
                     whiskerprops=dict(linewidth=1.5),
                     capprops=dict(linewidth=1.5))

    # Add mean line
    monthly_means = [np.mean(data) if len(data) > 0 else 0 for data in monthly_data]
    ax1.plot(range(1, 13), monthly_means, 'go-', linewidth=2, markersize=8,
            label='Monthly Mean', zorder=5)

    # Add overall mean line
    overall_mean = df_temp[column].mean()
    ax1.axhline(overall_mean, color='orange', linestyle='--', linewidth=2,
               label=f'Overall Mean: {overall_mean:.2f} MVA', alpha=0.7)

    # Formatting subplot 1
    ax1.set_xlabel('Month', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Apparent Power (MVA)', fontsize=12, fontweight='bold')
    ax1.set_title('Monthly Load Distribution (Boxplots)',
                 fontsize=13, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=10)
    ax1.grid(True, alpha=0.3, axis='y')

    # ========== Subplot 2: Monthly Statistics with Error Bands ==========
    # Calculate monthly statistics
    monthly_stats = df_temp.groupby('Month')[column].agg(['mean', 'std', 'min', 'max', 'median'])

    months = np.arange(1, 13)
    means = monthly_stats['mean'].values
    stds = monthly_stats['std'].values
    medians = monthly_stats['median'].values
    mins = monthly_stats['min'].values
    maxs = monthly_stats['max'].values

    # Plot mean with std error band
    ax2.plot(months, means, 'b-', linewidth=2.5, marker='o', markersize=8,
            label='Mean Load', zorder=5)
    ax2.fill_between(months, means - stds, means + stds,
                     alpha=0.3, color='blue', label='±1 Std Dev')

    # Plot median
    ax2.plot(months, medians, 'g--', linewidth=2, marker='s', markersize=6,
            label='Median Load', zorder=5)

    # Plot min/max range as shaded area
    ax2.fill_between(months, mins, maxs, alpha=0.15, color='gray',
                     label='Min-Max Range')

    # Add seasonal annotations
    # Winter: Dec, Jan, Feb (months 12, 1, 2)
    # Spring: Mar, Apr, May (months 3, 4, 5)
    # Summer: Jun, Jul, Aug (months 6, 7, 8)
    # Fall: Sep, Oct, Nov (months 9, 10, 11)
    ax2.axvspan(0.5, 2.5, color='cyan', alpha=0.1, label='Winter')
    ax2.axvspan(2.5, 5.5, color='green', alpha=0.1, label='Spring')
    ax2.axvspan(5.5, 8.5, color='yellow', alpha=0.1, label='Summer')
    ax2.axvspan(8.5, 11.5, color='orange', alpha=0.1, label='Fall')
    ax2.axvspan(11.5, 12.5, color='cyan', alpha=0.1)  # December (Winter)

    # Formatting subplot 2
    ax2.set_xlabel('Month', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Apparent Power (MVA)', fontsize=12, fontweight='bold')
    ax2.set_title('Monthly Load Statistics with Seasonal Patterns',
                 fontsize=13, fontweight='bold')
    ax2.set_xticks(months)
    ax2.set_xticklabels(month_labels)
    ax2.legend(loc='upper right', fontsize=9, ncol=2)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0.5, 12.5)

    # ========== Add Statistics Summary ==========
    # Find peak and low months
    peak_month_idx = np.argmax(means)
    low_month_idx = np.argmin(means)
    peak_month = month_labels[peak_month_idx]
    low_month = month_labels[low_month_idx]

    # Calculate variability
    max_variability_idx = np.argmax(stds)
    min_variability_idx = np.argmin(stds)

    '''stats_text = f'Yearly Seasonality Summary:\n'
    stats_text += f'Peak Month: {peak_month} ({means[peak_month_idx]:.2f} MVA)\n'
    stats_text += f'Low Month: {low_month} ({means[low_month_idx]:.2f} MVA)\n'
    stats_text += f'Annual Range: {means[peak_month_idx] - means[low_month_idx]:.2f} MVA\n'
    stats_text += f'Most Variable: {month_labels[max_variability_idx]} (σ={stds[max_variability_idx]:.2f})\n'
    stats_text += f'Most Stable: {month_labels[min_variability_idx]} (σ={stds[min_variability_idx]:.2f})'

    fig.text(0.98, 0.48, stats_text, transform=fig.transFigure,
            verticalalignment='center', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9),
            fontsize=10, family='monospace')'''

    # Main title
    fig.suptitle('Yearly Seasonality Analysis of Apparent Power',
                fontsize=15, fontweight='bold', y=0.995)

    plt.tight_layout(rect=[0, 0, 0.97, 0.99])

    # Save plot
    plt.savefig(outdir / 'yearly_seasonality.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"[OK] Yearly Seasonality saved to {outdir / 'yearly_seasonality.png'}")


def generate_all_analysis_plots(df, outdir):
    """
    Generate all 6 data analysis visualization figures.

    Args:
        df: DataFrame with load data (should have columns: S_TOTAL, P_TOTAL, Q_TOTAL)
        outdir: Output directory path for saving plots

    Returns:
        None (saves all plots to outdir)
    """
    print("\n" + "="*70)
    print("Generating Data Analysis Visualizations")
    print("="*70 + "\n")

    # Create output directory
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    try:
        # Figure 1: Load Duration Curve
        print("Generating Figure 1: Augmented Load Duration Curve...")
        plot_load_duration_curve(df, outdir)

        # Figure 2: Temporal Heatmap
        print("Generating Figure 2: Temporal Heatmap...")
        plot_temporal_heatmap(df, outdir)

        # Figure 3: Distribution Analysis
        print("Generating Figure 3: Distribution Analysis...")
        plot_distribution_analysis(df, outdir)

        # Figure 4: Seasonal Decomposition
        print("Generating Figure 4: Seasonal Decomposition...")
        plot_seasonal_decomposition(df, outdir)

        # Figure 5: Power Factor Analysis
        print("Generating Figure 5: Power Factor Analysis...")
        plot_power_factor_analysis(df, outdir)

        # Figure 6: Yearly Seasonality
        print("Generating Figure 6: Yearly Seasonality...")
        plot_yearly_seasonality(df, outdir)

        print("\n" + "="*70)
        print("All Data Analysis Visualizations Complete!")
        print(f"Plots saved to: {outdir}")
        print("="*70 + "\n")

    except Exception as e:
        print(f"\n[ERROR] Error generating plots: {str(e)}")
        raise


if __name__ == "__main__":
    """
    Example usage for testing the visualization module.
    """
    import sys

    if len(sys.argv) < 2:
        print("Usage: python data_analysis_plots.py <path_to_excel_or_directory> [output_dir]")
        sys.exit(1)

    data_path = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "outputs/data_analysis"

    # Load data
    print(f"Loading data from: {data_path}")
    from data.load_and_clean import load_and_clean
    df = load_and_clean(data_path)

    # Generate all plots
    generate_all_analysis_plots(df, output_dir)
