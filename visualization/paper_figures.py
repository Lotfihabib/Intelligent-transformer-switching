"""
Publication Figures Module

Generates publication-quality figures for the scientific paper:
1. Representative weekly operation (3-panel: load+forecast, N step, loading ratio)
2. Critical transition event zoom (2-panel: load detail, N comparison)
3. Switching decision heatmap (2-panel: frequency + avg N, with marginals)
4. Load duration curve with operating regions (LDC + actual N color overlay)
5. Cumulative loss comparison (MPC vs legacy baseline)

All figures save as both PNG (300 DPI) and PDF.
"""

import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.lines import Line2D
from pathlib import Path

from config import CONFIG


plt.style.use('default')

# Consistent color scheme
N_COLORS = {1: '#66BB6A', 2: '#FF9800', 3: '#EF5350'}
N_LABELS = {1: 'N = 1', 2: 'N = 2', 3: 'N = 3'}
BP_12 = CONFIG['Sr'] * math.sqrt(2 * CONFIG['P0'] / CONFIG['Ploss_rated'])
BP_23 = CONFIG['Sr'] * math.sqrt(6 * CONFIG['P0'] / CONFIG['Ploss_rated'])
SR = CONFIG['Sr']


def _save_figure(fig, outdir, name):
    """Save figure as both PNG and PDF at 300 DPI."""
    outdir = Path(outdir)
    fig.savefig(outdir / f'{name}.png', dpi=300, bbox_inches='tight')
    fig.savefig(outdir / f'{name}.pdf', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"    Saved {name}.png/pdf")


def _compute_baseline_loss(load_mva):
    """Compute per-timestep loss at N=3 (legacy baseline), vectorized."""
    P0 = CONFIG['P0']
    Ploss_rated = CONFIG['Ploss_rated']
    Sr = CONFIG['Sr']
    load_mva = np.asarray(load_mva, dtype=float)
    return 3 * (P0 + Ploss_rated * (load_mva / (3 * Sr)) ** 2)


def _select_interesting_week(df):
    """
    Select a week with diverse regime transitions for the representative plot.

    Returns:
        int: start index of the best week (1008-sample window)
    """
    week_len = 7 * 144  # 7 days at 10-min resolution
    best_score = -1
    best_idx = 0
    step = 144  # Slide by 1 day at a time for efficiency

    for start in range(0, len(df) - week_len, step):
        chunk = df.iloc[start:start + week_len]

        # Skip if forecast data is mostly missing
        if chunk['forecast_q50'].isna().sum() > week_len * 0.5:
            continue

        n_unique = chunk['N_current'].nunique()
        n_switches = (chunk['N_current'] != chunk['N_current'].shift(1)).sum()
        load_range = chunk['S_TOTAL'].max() - chunk['S_TOTAL'].min()

        # Score: prioritize diversity, then switching activity, then load range
        score = n_unique * 100 + n_switches * 10 + load_range

        if score > best_score:
            best_score = score
            best_idx = start

    return best_idx


def _find_transition_window(df):
    """
    Find a 48-hour window with multiple switching events and breakpoint crossings.

    Returns:
        tuple: (start_idx, end_idx) for the window
    """
    window_len = 48 * 6  # 48 hours at 10-min resolution
    switch_mask = df['N_current'] != df['N_current'].shift(1)
    switch_indices = np.where(switch_mask.values)[0]

    if len(switch_indices) < 2:
        return 0, min(window_len, len(df))

    best_score = -1
    best_start = 0

    for idx in switch_indices:
        start = max(0, idx - window_len // 4)  # Start a bit before the switch
        end = min(len(df), start + window_len)
        if end - start < window_len // 2:
            continue

        chunk = df.iloc[start:end]

        # Skip if forecast data is mostly missing
        if chunk['forecast_q50'].isna().sum() > len(chunk) * 0.5:
            continue

        n_switches = switch_mask.iloc[start:end].sum()
        n_unique = chunk['N_current'].nunique()
        crosses_bp = (chunk['S_TOTAL'].min() < BP_12 < chunk['S_TOTAL'].max() or
                      chunk['S_TOTAL'].min() < BP_23 < chunk['S_TOTAL'].max())

        score = n_switches * 20 + n_unique * 50 + (30 if crosses_bp else 0)

        if score > best_score:
            best_score = score
            best_start = start

    return best_start, min(best_start + window_len, len(df))


# ─────────────────────────────────────────────────────────────────────
# Figure 1: Representative Weekly Operation
# ─────────────────────────────────────────────────────────────────────

def plot_weekly_operation(df, outdir):
    """3-panel weekly view: load+forecast, active transformers, loading ratio."""
    outdir = Path(outdir)
    week_start = _select_interesting_week(df)
    week_end = week_start + 7 * 144
    week = df.iloc[week_start:week_end].copy()
    ts = pd.to_datetime(week['timestamp'])

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(16, 12), sharex=True)

    date_range = f"{ts.iloc[0].strftime('%b %d')} – {ts.iloc[-1].strftime('%b %d, %Y')}"
    fig.suptitle(f'Representative Weekly Operation ({date_range})',
                 fontsize=14, fontweight='bold', y=0.98)

    # Panel A: Load with forecast envelope
    ax1.plot(ts, week['S_TOTAL'], color='black', linewidth=1.5,
             label='Actual Load', zorder=3)

    has_forecast = week['forecast_q50'].notna().any()
    if has_forecast:
        valid = week['forecast_q50'].notna()
        ax1.plot(ts[valid], week.loc[valid, 'forecast_q50'],
                 color='#1E88E5', linewidth=1, alpha=0.8, label='Forecast (q50)')
        ax1.fill_between(ts[valid],
                         week.loc[valid, 'forecast_q10'],
                         week.loc[valid, 'forecast_q90'],
                         color='#1E88E5', alpha=0.15, label='80% PI (q10–q90)')

    ax1.axhline(BP_23, color='#EF5350', linestyle='--', linewidth=1, alpha=0.7,
                label=f'bp$_{{2\\rightarrow3}}$ = {BP_23:.1f} MVA')
    ax1.axhline(BP_12, color='#66BB6A', linestyle='--', linewidth=1, alpha=0.7,
                label=f'bp$_{{1\\rightarrow2}}$ = {BP_12:.1f} MVA')

    ax1.set_ylabel('Apparent Power (MVA)', fontsize=11)
    ax1.legend(loc='upper right', fontsize=8, ncol=2)
    ax1.grid(True, alpha=0.3, linestyle='--')

    # Panel B: Active transformers (step plot with background coloring)
    ax2.step(ts, week['N_current'], where='post', color='#1565C0', linewidth=1.5)

    # Color background by N regime
    n_vals = week['N_current'].values
    for i in range(len(n_vals) - 1):
        ax2.axvspan(ts.iloc[i], ts.iloc[i + 1],
                    color=N_COLORS[n_vals[i]], alpha=0.12)

    ax2.set_ylabel('Active Transformers (N)', fontsize=11)
    ax2.set_ylim(0.5, 3.5)
    ax2.set_yticks([1, 2, 3])
    ax2.grid(True, alpha=0.3, linestyle='--')

    # Custom legend for background colors
    legend_elements = [Line2D([0], [0], color=N_COLORS[n], linewidth=6, alpha=0.4,
                              label=N_LABELS[n]) for n in [1, 2, 3]]
    ax2.legend(handles=legend_elements, loc='upper right', fontsize=8)

    # Panel C: Per-unit loading ratio
    loading_ratio = week['S_TOTAL'] / (week['N_current'] * SR)
    ax3.plot(ts, loading_ratio, color='#1565C0', linewidth=1, alpha=0.8)
    ax3.axhline(1.0, color='#EF5350', linestyle='--', linewidth=1.2,
                label='Rated Capacity (1.0 p.u.)')
    ax3.axhline(0.85, color='#FF9800', linestyle=':', linewidth=1,
                label='Warning Threshold (0.85 p.u.)')

    ax3.set_ylabel('Loading Ratio (p.u.)', fontsize=11)
    ax3.set_xlabel('Time', fontsize=11)
    ax3.legend(loc='upper right', fontsize=8)
    ax3.grid(True, alpha=0.3, linestyle='--')
    ax3.set_ylim(0, max(1.1, loading_ratio.max() * 1.1))

    # Format shared x-axis
    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%a %H:%M'))
    ax3.xaxis.set_major_locator(mdates.HourLocator(interval=12))
    ax3.xaxis.set_minor_locator(mdates.HourLocator(interval=6))
    plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45, ha='right')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    _save_figure(fig, outdir, 'fig_weekly_operation')


# ─────────────────────────────────────────────────────────────────────
# Figure 2: Critical Transition Event Zoom
# ─────────────────────────────────────────────────────────────────────

def plot_transition_zoom(df, outdir):
    """2-panel zoom into a 48h transition with forecast, decisions, and safety."""
    outdir = Path(outdir)
    start, end = _find_transition_window(df)
    window = df.iloc[start:end].copy()
    ts = pd.to_datetime(window['timestamp'])

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True,
                                    gridspec_kw={'height_ratios': [2, 1]})

    date_range = f"{ts.iloc[0].strftime('%b %d %H:%M')} – {ts.iloc[-1].strftime('%b %d %H:%M, %Y')}"
    fig.suptitle(f'Critical Transition Event ({date_range})',
                 fontsize=14, fontweight='bold', y=0.98)

    # Panel A: Load with forecast and breakpoints
    ax1.plot(ts, window['S_TOTAL'], color='black', linewidth=1.5,
             label='Actual Load', zorder=3)

    has_forecast = window['forecast_q50'].notna().any()
    if has_forecast:
        valid = window['forecast_q50'].notna()
        ax1.plot(ts[valid], window.loc[valid, 'forecast_q50'],
                 color='#1E88E5', linewidth=1, alpha=0.8, label='Forecast (q50)')
        ax1.fill_between(ts[valid],
                         window.loc[valid, 'forecast_q10'],
                         window.loc[valid, 'forecast_q90'],
                         color='#1E88E5', alpha=0.15, label='80% PI')

    ax1.axhline(BP_23, color='#EF5350', linestyle='--', linewidth=1, alpha=0.7,
                label=f'bp$_{{2\\rightarrow3}}$')
    ax1.axhline(BP_12, color='#66BB6A', linestyle='--', linewidth=1, alpha=0.7,
                label=f'bp$_{{1\\rightarrow2}}$')

    # Mark switching events with vertical lines and annotations
    switch_mask = window['N_current'] != window['N_current'].shift(1)
    switch_rows = window[switch_mask]
    for _, row in switch_rows.iterrows():
        t = pd.to_datetime(row['timestamp'])
        prev = int(row['prev_N']) if pd.notna(row.get('prev_N')) else '?'
        curr = int(row['N_current'])
        ax1.axvline(t, color='#AB47BC', linewidth=1.5, alpha=0.7, linestyle='-')
        ax1.annotate(f'N: {prev}→{curr}', xy=(t, ax1.get_ylim()[1] * 0.95),
                     fontsize=7, color='#AB47BC', fontweight='bold',
                     ha='center', va='top',
                     bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))

    # Mark safety override events
    safety_str = window['safety_rules'].fillna('').astype(str)
    override_mask = safety_str.str.strip() != ''
    if override_mask.any():
        override_ts = ts[override_mask]
        override_load = window.loc[override_mask, 'S_TOTAL']
        ax1.scatter(override_ts, override_load, color='#EF5350', marker='x',
                    s=30, zorder=4, label='Safety Override')

    ax1.set_ylabel('Apparent Power (MVA)', fontsize=11)
    ax1.legend(loc='upper right', fontsize=8, ncol=2)
    ax1.grid(True, alpha=0.3, linestyle='--')

    # Panel B: Transformer configuration comparison
    ax2.step(ts, window['N_current'], where='post', color='#1565C0',
             linewidth=2, label='Applied (N_current)')
    ax2.step(ts, window['N_mpc'], where='post', color='gray',
             linewidth=1.5, linestyle='--', alpha=0.7, label='MPC Recommendation')

    # Shade regions where safety overrode MPC
    mismatch = window['N_current'] != window['N_mpc']
    if mismatch.any():
        for i in range(len(window) - 1):
            if mismatch.iloc[i]:
                ax2.axvspan(ts.iloc[i], ts.iloc[i + 1],
                            color='#EF5350', alpha=0.15)
        # Add a single legend entry for override shading
        ax2.fill_between([], [], [], color='#EF5350', alpha=0.15,
                         label='Safety Override Region')

    ax2.set_ylabel('Active Transformers', fontsize=11)
    ax2.set_xlabel('Time', fontsize=11)
    ax2.set_ylim(0.5, 3.5)
    ax2.set_yticks([1, 2, 3])
    ax2.legend(loc='upper right', fontsize=8)
    ax2.grid(True, alpha=0.3, linestyle='--')

    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
    ax2.xaxis.set_major_locator(mdates.HourLocator(interval=4))
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    _save_figure(fig, outdir, 'fig_transition_zoom')


# ─────────────────────────────────────────────────────────────────────
# Figure 3: Switching Decision Heatmap
# ─────────────────────────────────────────────────────────────────────

def plot_switching_heatmap(df, outdir):
    """2-panel heatmap: switching frequency and average N by hour and day."""
    outdir = Path(outdir)
    ts = pd.to_datetime(df['timestamp'])
    df_temp = df.copy()
    df_temp['hour'] = ts.dt.hour
    df_temp['dow'] = ts.dt.dayofweek  # 0=Mon, 6=Sun
    df_temp['is_switch'] = (df_temp['N_current'] != df_temp['N_current'].shift(1)).astype(int)
    # First row is not a valid switch comparison
    df_temp.iloc[0, df_temp.columns.get_loc('is_switch')] = 0

    day_labels = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']

    # Pivot tables
    switch_pivot = df_temp.pivot_table(values='is_switch', index='dow',
                                       columns='hour', aggfunc='sum', fill_value=0)
    avg_n_pivot = df_temp.pivot_table(values='N_current', index='dow',
                                      columns='hour', aggfunc='mean')

    # Ensure full 7x24 grid
    switch_data = np.zeros((7, 24))
    avg_n_data = np.full((7, 24), np.nan)
    for d in range(7):
        for h in range(24):
            if d in switch_pivot.index and h in switch_pivot.columns:
                switch_data[d, h] = switch_pivot.loc[d, h]
            if d in avg_n_pivot.index and h in avg_n_pivot.columns:
                avg_n_data[d, h] = avg_n_pivot.loc[d, h]

    fig = plt.figure(figsize=(16, 8))
    fig.suptitle('Switching Decision Patterns by Hour and Day of Week',
                 fontsize=14, fontweight='bold', y=0.98)

    # Left panel: Switching frequency heatmap with marginals
    # Main heatmap
    ax_heat1 = fig.add_axes([0.06, 0.08, 0.35, 0.7])
    ax_top1 = fig.add_axes([0.06, 0.80, 0.35, 0.12])   # Top marginal
    ax_right1 = fig.add_axes([0.42, 0.08, 0.04, 0.7])   # Right marginal

    im1 = ax_heat1.imshow(switch_data, aspect='auto', cmap='YlOrRd',
                           interpolation='nearest')

    # Text annotations
    vmax = switch_data.max()
    for i in range(7):
        for j in range(24):
            val = int(switch_data[i, j])
            if val > 0:
                color = 'white' if val > vmax * 0.6 else 'black'
                ax_heat1.text(j, i, str(val), ha='center', va='center',
                              fontsize=6, color=color, fontweight='bold')

    ax_heat1.set_xticks(np.arange(24))
    ax_heat1.set_xticklabels([f'{h:02d}' for h in range(24)], fontsize=7)
    ax_heat1.set_yticks(np.arange(7))
    ax_heat1.set_yticklabels(day_labels, fontsize=9)
    ax_heat1.set_xlabel('Hour of Day', fontsize=10)
    ax_heat1.set_ylabel('Day of Week', fontsize=10)
    ax_heat1.set_title('Switching Frequency', fontsize=11, fontweight='bold')

    # Top marginal: hourly totals
    hourly_totals = switch_data.sum(axis=0)
    ax_top1.bar(np.arange(24), hourly_totals, color='#FF9800', edgecolor='white',
                width=0.8)
    ax_top1.set_xlim(-0.5, 23.5)
    ax_top1.set_xticks([])
    ax_top1.set_ylabel('Total', fontsize=8)
    ax_top1.tick_params(labelsize=7)

    # Right marginal: daily totals
    daily_totals = switch_data.sum(axis=1)
    ax_right1.barh(np.arange(7), daily_totals, color='#FF9800', edgecolor='white',
                   height=0.8)
    ax_right1.set_ylim(-0.5, 6.5)
    ax_right1.set_yticks([])
    ax_right1.invert_yaxis()
    ax_right1.set_xlabel('Total', fontsize=8)
    ax_right1.tick_params(labelsize=7)

    # Right panel: Average N heatmap
    ax_heat2 = fig.add_axes([0.56, 0.08, 0.35, 0.7])

    im2 = ax_heat2.imshow(avg_n_data, aspect='auto', cmap='RdYlGn_r',
                           interpolation='nearest', vmin=1.0, vmax=3.0)

    for i in range(7):
        for j in range(24):
            val = avg_n_data[i, j]
            if not np.isnan(val):
                color = 'white' if val > 2.3 else 'black'
                ax_heat2.text(j, i, f'{val:.1f}', ha='center', va='center',
                              fontsize=6, color=color, fontweight='bold')

    ax_heat2.set_xticks(np.arange(24))
    ax_heat2.set_xticklabels([f'{h:02d}' for h in range(24)], fontsize=7)
    ax_heat2.set_yticks(np.arange(7))
    ax_heat2.set_yticklabels(day_labels, fontsize=9)
    ax_heat2.set_xlabel('Hour of Day', fontsize=10)
    ax_heat2.set_title('Mean Active Transformers', fontsize=11, fontweight='bold')

    # Colorbars
    cax1 = fig.add_axes([0.42, 0.82, 0.04, 0.10])
    fig.colorbar(im1, cax=cax1, orientation='vertical')
    cax1.tick_params(labelsize=7)

    cax2 = fig.add_axes([0.92, 0.08, 0.015, 0.7])
    fig.colorbar(im2, cax=cax2, orientation='vertical')
    cax2.tick_params(labelsize=7)

    _save_figure(fig, outdir, 'fig_switching_heatmap')


# ─────────────────────────────────────────────────────────────────────
# Figure 4: Load Duration Curve with Operating Regions
# ─────────────────────────────────────────────────────────────────────

def plot_ldc_with_operations(df, outdir):
    """LDC with actual transformer configuration color overlay."""
    outdir = Path(outdir)

    load = df['S_TOTAL'].values
    n_current = df['N_current'].values

    # Sort by load descending, keep paired N values
    sort_idx = np.argsort(load)[::-1]
    sorted_load = load[sort_idx]
    sorted_n = n_current[sort_idx]
    time_pct = np.linspace(0, 100, len(sorted_load))

    fig, ax = plt.subplots(figsize=(12, 7))

    # Background theoretical zones
    n_pts = len(sorted_load)
    idx_bp23 = np.searchsorted(-sorted_load, -BP_23)
    idx_bp12 = np.searchsorted(-sorted_load, -BP_12)
    pct_bp23 = (idx_bp23 / n_pts) * 100
    pct_bp12 = (idx_bp12 / n_pts) * 100

    if pct_bp23 > 0:
        ax.axvspan(0, pct_bp23, color='#EF5350', alpha=0.06, zorder=0)
    ax.axvspan(pct_bp23, pct_bp12, color='#FF9800', alpha=0.06, zorder=0)
    if pct_bp12 < 100:
        ax.axvspan(pct_bp12, 100, color='#66BB6A', alpha=0.06, zorder=0)

    # LDC line
    ax.plot(time_pct, sorted_load, color='black', linewidth=1.5, zorder=2,
            label='Load Duration Curve')

    # Scatter overlay colored by actual N
    for n_val in [1, 2, 3]:
        mask = sorted_n == n_val
        if mask.any():
            ax.scatter(time_pct[mask], sorted_load[mask], c=N_COLORS[n_val],
                       s=2, alpha=0.5, zorder=3, label=f'Operated at N = {n_val}')

    # Breakpoint lines
    ax.axhline(BP_23, color='#EF5350', linestyle='--', linewidth=1.5, alpha=0.7,
               label=f'bp$_{{2\\rightarrow3}}$ = {BP_23:.1f} MVA')
    ax.axhline(BP_12, color='#66BB6A', linestyle='--', linewidth=1.5, alpha=0.7,
               label=f'bp$_{{1\\rightarrow2}}$ = {BP_12:.1f} MVA')

    # Zone percentage annotations
    # Theoretical
    theo_n3_pct = pct_bp23
    theo_n2_pct = pct_bp12 - pct_bp23
    theo_n1_pct = 100 - pct_bp12
    # Actual
    act_n3_pct = (n_current == 3).sum() / len(n_current) * 100
    act_n2_pct = (n_current == 2).sum() / len(n_current) * 100
    act_n1_pct = (n_current == 1).sum() / len(n_current) * 100

    stats_text = (
        f"Zone Distribution\n"
        f"{'':>12s} {'Theo.':>6s} {'Actual':>6s}\n"
        f"{'N=3':>12s} {theo_n3_pct:5.1f}% {act_n3_pct:5.1f}%\n"
        f"{'N=2':>12s} {theo_n2_pct:5.1f}% {act_n2_pct:5.1f}%\n"
        f"{'N=1':>12s} {theo_n1_pct:5.1f}% {act_n1_pct:5.1f}%"
    )
    ax.text(0.98, 0.55, stats_text, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', horizontalalignment='right',
            family='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9))

    ax.set_xlabel('Time Exceedance (%)', fontsize=12)
    ax.set_ylabel('Apparent Power (MVA)', fontsize=12)
    ax.set_title('Load Duration Curve with Actual Operating Configuration',
                 fontsize=14, fontweight='bold', pad=15)
    ax.legend(loc='upper right', fontsize=8,
              bbox_to_anchor=(0.98, 0.98))
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xlim(0, 100)

    plt.tight_layout()
    _save_figure(fig, outdir, 'fig_ldc_operating')


# ─────────────────────────────────────────────────────────────────────
# Figure 5: Cumulative Loss Comparison
# ─────────────────────────────────────────────────────────────────────

def plot_cumulative_losses(df, outdir):
    """Cumulative loss comparison: TFT-SMPC vs legacy N=3 baseline."""
    outdir = Path(outdir)
    ts = pd.to_datetime(df['timestamp'])

    # Compute cumulative losses
    mpc_loss_kwh = df['loss_kwh'].values
    baseline_loss_kw = _compute_baseline_loss(df['S_TOTAL'].values)
    baseline_loss_kwh = baseline_loss_kw / 6.0  # 10-min intervals

    cum_mpc = np.cumsum(mpc_loss_kwh) / 1000.0  # Convert to MWh
    cum_baseline = np.cumsum(baseline_loss_kwh) / 1000.0
    cum_savings = cum_baseline - cum_mpc

    fig, ax = plt.subplots(figsize=(14, 6))

    ax.plot(ts, cum_baseline, color='#EF5350', linewidth=1.5,
            label='Legacy Baseline (N = 3)')
    ax.plot(ts, cum_mpc, color='#1E88E5', linewidth=1.5,
            label='TFT-SMPC Strategy')
    ax.fill_between(ts, cum_mpc, cum_baseline, color='#66BB6A', alpha=0.25,
                    label='Cumulative Savings')

    # Annotate final savings
    final_savings = cum_savings.iloc[-1] if hasattr(cum_savings, 'iloc') else cum_savings[-1]
    ax.annotate(f'Total Savings: {final_savings:.1f} MWh',
                xy=(ts.iloc[-1], (cum_mpc[-1] + cum_baseline[-1]) / 2),
                fontsize=11, fontweight='bold', color='#2E7D32',
                ha='right', va='center',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                          edgecolor='#66BB6A', alpha=0.9))

    # Secondary y-axis for savings
    ax2 = ax.twinx()
    ax2.plot(ts, cum_savings, color='#2E7D32', linewidth=1, alpha=0.5,
             linestyle=':')
    ax2.set_ylabel('Cumulative Savings (MWh)', fontsize=11, color='#2E7D32')
    ax2.tick_params(axis='y', labelcolor='#2E7D32')

    ax.set_xlabel('Date', fontsize=11)
    ax.set_ylabel('Cumulative Energy Losses (MWh)', fontsize=11)
    ax.set_title('Cumulative Loss Comparison: TFT-SMPC vs Legacy Operation',
                 fontsize=14, fontweight='bold', pad=15)
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3, linestyle='--')

    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=0, ha='center')

    plt.tight_layout()
    _save_figure(fig, outdir, 'fig_cumulative_losses')


# ─────────────────────────────────────────────────────────────────────
# Orchestrator
# ─────────────────────────────────────────────────────────────────────

def generate_paper_figures(logs_path, outdir):
    """
    Generate all publication-quality figures from backtesting logs.

    Args:
        logs_path: Path to logs.csv from backtesting
        outdir: Output directory for figures
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print("\n--- Generating Publication Figures ---")
    df = pd.read_csv(logs_path)
    print(f"  Loaded {len(df)} timesteps from {logs_path}")

    # Ensure timestamp is parsed
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    print("  [1/5] Weekly operation...")
    plot_weekly_operation(df, outdir)

    print("  [2/5] Transition event zoom...")
    plot_transition_zoom(df, outdir)

    print("  [3/5] Switching decision heatmap...")
    plot_switching_heatmap(df, outdir)

    print("  [4/5] Load duration curve with operations...")
    plot_ldc_with_operations(df, outdir)

    print("  [5/5] Cumulative loss comparison...")
    plot_cumulative_losses(df, outdir)

    print("  All paper figures generated successfully.")
