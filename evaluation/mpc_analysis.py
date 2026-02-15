"""
MPC Performance Analysis Module

Post-hoc analysis of stochastic MPC switching decisions:
1. Enhanced baseline comparison with iron/copper loss separation
2. Switching statistics (frequency distributions, dwell times)
3. Operational regime analysis (time in each N, vs theoretical optimal)
4. Computational timing analysis (MPC solve times)

All analysis functions work from logs.csv and switching_events data,
requiring no access to the trained model or raw data.
"""

import json
import math
import numpy as np
import pandas as pd
from pathlib import Path

from config import CONFIG
from control.power_model import compute_breakpoints_and_thresholds


def _compute_raw_breakpoints():
    """
    Compute raw switching breakpoints (no hysteresis) for theoretical optimal.

    Returns:
        tuple: (bp_1_2, bp_2_3) in MVA
    """
    P0 = CONFIG['P0']
    Ploss_rated = CONFIG['Ploss_rated']
    Sr = CONFIG['Sr']
    bp_1_2 = Sr * math.sqrt(2 * P0 / Ploss_rated)
    bp_2_3 = Sr * math.sqrt(6 * P0 / Ploss_rated)
    return bp_1_2, bp_2_3


def _compute_loss_components(load_mva, N):
    """
    Decompose transformer losses into iron and copper components (vectorized).

    Args:
        load_mva: float or np.ndarray, total load in MVA
        N: int or np.ndarray, number of active transformers

    Returns:
        tuple: (iron_loss_kw, copper_loss_kw) as float or np.ndarray
    """
    P0 = CONFIG['P0']
    Ploss_rated = CONFIG['Ploss_rated']
    Sr = CONFIG['Sr']

    load_mva = np.asarray(load_mva, dtype=float)
    N = np.asarray(N, dtype=float)

    iron_loss = N * P0
    load_per_tr = np.where(N > 0, load_mva / N, 0.0)
    copper_loss = N * Ploss_rated * (load_per_tr / Sr) ** 2

    return iron_loss, copper_loss


def analyze_baseline_comparison(df):
    """
    Enhanced baseline comparison with iron/copper loss decomposition.

    For each timestep, computes iron and copper losses for both
    MPC strategy (N_current) and baseline strategy (N=3 always).

    Args:
        df: DataFrame with columns [S_TOTAL, N_current] from logs.csv

    Returns:
        dict: Loss decomposition and savings metrics
    """
    load = df['S_TOTAL'].values
    N_mpc = df['N_current'].values
    N_baseline = np.full_like(N_mpc, 3)
    Sr = CONFIG['Sr']

    # MPC strategy losses
    mpc_iron, mpc_copper = _compute_loss_components(load, N_mpc)
    # Handle overload: if load > N*Sr, assign penalty
    mpc_overload = load > N_mpc * Sr
    mpc_total = np.where(mpc_overload, 1000.0, mpc_iron + mpc_copper)
    mpc_iron = np.where(mpc_overload, 0.0, mpc_iron)
    mpc_copper = np.where(mpc_overload, 0.0, mpc_copper)

    # Baseline strategy losses (N=3 always)
    bl_iron, bl_copper = _compute_loss_components(load, N_baseline)
    bl_overload = load > 3 * Sr
    bl_total = np.where(bl_overload, 1000.0, bl_iron + bl_copper)
    bl_iron = np.where(bl_overload, 0.0, bl_iron)
    bl_copper = np.where(bl_overload, 0.0, bl_copper)

    # Convert kW to kWh (10-min intervals -> divide by 6)
    interval_factor = 1.0 / 6.0

    mpc_total_kwh = float(np.sum(mpc_total) * interval_factor)
    mpc_iron_kwh = float(np.sum(mpc_iron) * interval_factor)
    mpc_copper_kwh = float(np.sum(mpc_copper) * interval_factor)

    bl_total_kwh = float(np.sum(bl_total) * interval_factor)
    bl_iron_kwh = float(np.sum(bl_iron) * interval_factor)
    bl_copper_kwh = float(np.sum(bl_copper) * interval_factor)

    iron_savings = bl_iron_kwh - mpc_iron_kwh
    copper_savings = bl_copper_kwh - mpc_copper_kwh
    total_savings = bl_total_kwh - mpc_total_kwh

    return {
        'mpc_total_loss_kwh': mpc_total_kwh,
        'mpc_iron_loss_kwh': mpc_iron_kwh,
        'mpc_copper_loss_kwh': mpc_copper_kwh,
        'baseline_total_loss_kwh': bl_total_kwh,
        'baseline_iron_loss_kwh': bl_iron_kwh,
        'baseline_copper_loss_kwh': bl_copper_kwh,
        'iron_savings_kwh': iron_savings,
        'iron_savings_pct': (iron_savings / bl_iron_kwh * 100) if bl_iron_kwh > 0 else 0.0,
        'copper_savings_kwh': copper_savings,
        'copper_savings_pct': (copper_savings / bl_copper_kwh * 100) if bl_copper_kwh > 0 else 0.0,
        'total_savings_kwh': total_savings,
        'total_savings_pct': (total_savings / bl_total_kwh * 100) if bl_total_kwh > 0 else 0.0,
    }


def analyze_switching_statistics(df, switching_events):
    """
    Comprehensive switching frequency and dwell time analysis.

    Args:
        df: DataFrame from logs.csv with timestamp column
        switching_events: list of switching event dicts

    Returns:
        dict: Switching statistics including frequency distributions and dwell times
    """
    timestamps = pd.to_datetime(df['timestamp'])
    total_hours = len(df) / 6.0
    total_days = total_hours / 24.0
    total_weeks = total_days / 7.0

    num_switches = len(switching_events)

    # Switches per time period
    switches_per_day = num_switches / total_days if total_days > 0 else 0.0
    switches_per_week = num_switches / total_weeks if total_weeks > 0 else 0.0
    switches_per_month = num_switches * (30.44 * 24 / total_hours) if total_hours > 0 else 0.0

    # Transition type counts
    up_transitions = sum(1 for e in switching_events if e['to_N'] > e['from_N'])
    down_transitions = sum(1 for e in switching_events if e['to_N'] < e['from_N'])
    safety_overrides = sum(1 for e in switching_events if e.get('safety_override', False))

    # Dwell time analysis
    # Compute dwell times from consecutive switching event timestamps
    dwell_times_minutes = []
    if len(switching_events) >= 2:
        switch_ts = [pd.to_datetime(e['timestamp']) for e in switching_events]
        for j in range(1, len(switch_ts)):
            dt_minutes = (switch_ts[j] - switch_ts[j - 1]).total_seconds() / 60.0
            dwell_times_minutes.append(dt_minutes)

    # Also add dwell from start to first switch and last switch to end
    if len(switching_events) >= 1:
        first_switch = pd.to_datetime(switching_events[0]['timestamp'])
        last_switch = pd.to_datetime(switching_events[-1]['timestamp'])
        dwell_start = (first_switch - timestamps.iloc[0]).total_seconds() / 60.0
        dwell_end = (timestamps.iloc[-1] - last_switch).total_seconds() / 60.0
        all_dwells = [dwell_start] + dwell_times_minutes + [dwell_end]
    else:
        all_dwells = [total_hours * 60.0]  # No switches: entire period is one dwell

    dwell_stats = {
        'count': len(all_dwells),
        'mean_minutes': float(np.mean(all_dwells)),
        'median_minutes': float(np.median(all_dwells)),
        'std_minutes': float(np.std(all_dwells)) if len(all_dwells) > 1 else 0.0,
        'min_minutes': float(np.min(all_dwells)),
        'max_minutes': float(np.max(all_dwells)),
    }

    # Hourly distribution (which hour of day do switches occur)
    hourly_dist = {h: 0 for h in range(24)}
    for e in switching_events:
        hour = pd.to_datetime(e['timestamp']).hour
        hourly_dist[hour] += 1

    # Daily distribution (which day of week)
    daily_dist = {d: 0 for d in range(7)}  # 0=Monday, 6=Sunday
    for e in switching_events:
        dow = pd.to_datetime(e['timestamp']).dayofweek
        daily_dist[dow] += 1

    # Per-day switching counts (for time series)
    daily_counts = {}
    for e in switching_events:
        day_str = pd.to_datetime(e['timestamp']).strftime('%Y-%m-%d')
        daily_counts[day_str] = daily_counts.get(day_str, 0) + 1

    # Per-week switching counts
    weekly_counts = {}
    for e in switching_events:
        ts = pd.to_datetime(e['timestamp'])
        week_key = f"{ts.isocalendar()[0]}-W{ts.isocalendar()[1]:02d}"
        weekly_counts[week_key] = weekly_counts.get(week_key, 0) + 1

    return {
        'total_switches': num_switches,
        'switches_per_day': switches_per_day,
        'switches_per_week': switches_per_week,
        'switches_per_month': switches_per_month,
        'dwell_times': dwell_stats,
        'hourly_distribution': hourly_dist,
        'daily_distribution': daily_dist,
        'daily_counts': daily_counts,
        'weekly_counts': weekly_counts,
        'up_transitions': up_transitions,
        'down_transitions': down_transitions,
        'safety_overrides': safety_overrides,
    }


def analyze_operational_regimes(df):
    """
    Operational regime analysis: actual vs theoretical optimal N.

    For each timestep, determines the theoretical optimal N using raw breakpoints
    (perfect foresight, zero switching cost) and compares against actual MPC decisions.

    Args:
        df: DataFrame from logs.csv with [S_TOTAL, N_current]

    Returns:
        dict: Regime distribution, match rate, per-regime metrics, optimality gap
    """
    load = df['S_TOTAL'].values
    N_actual = df['N_current'].values.astype(int)
    total_steps = len(df)
    P0 = CONFIG['P0']
    Ploss_rated = CONFIG['Ploss_rated']
    Sr = CONFIG['Sr']

    # Raw breakpoints (no hysteresis)
    bp_1_2, bp_2_3 = _compute_raw_breakpoints()

    # Theoretical optimal N for each load value
    N_optimal = np.ones(total_steps, dtype=int)
    N_optimal[load > bp_1_2] = 2
    N_optimal[load > bp_2_3] = 3

    # Time distribution - actual
    time_dist_actual = {}
    for n in [1, 2, 3]:
        count = int(np.sum(N_actual == n))
        time_dist_actual[n] = {
            'count': count,
            'pct': count / total_steps * 100,
            'hours': count / 6.0,
        }

    # Time distribution - theoretical optimal
    time_dist_optimal = {}
    for n in [1, 2, 3]:
        count = int(np.sum(N_optimal == n))
        time_dist_optimal[n] = {
            'count': count,
            'pct': count / total_steps * 100,
        }

    # Regime match rate
    regime_match = int(np.sum(N_actual == N_optimal))
    regime_match_pct = regime_match / total_steps * 100

    # Confusion matrix: actual vs optimal
    confusion = {}
    for n_act in [1, 2, 3]:
        for n_opt in [1, 2, 3]:
            count = int(np.sum((N_actual == n_act) & (N_optimal == n_opt)))
            confusion[f'actual_{n_act}_optimal_{n_opt}'] = count

    # Per-regime efficiency metrics
    regime_metrics = {}
    for n in [1, 2, 3]:
        mask = N_actual == n
        if np.sum(mask) > 0:
            regime_load = load[mask]
            iron, copper = _compute_loss_components(regime_load, n)
            total_loss = iron + copper
            regime_metrics[n] = {
                'avg_load_mva': float(np.mean(regime_load)),
                'avg_load_factor': float(np.mean(regime_load / (n * Sr))),
                'avg_loss_kw': float(np.mean(total_loss)),
                'total_loss_kwh': float(np.sum(total_loss) / 6.0),
            }
        else:
            regime_metrics[n] = {
                'avg_load_mva': 0.0,
                'avg_load_factor': 0.0,
                'avg_loss_kw': 0.0,
                'total_loss_kwh': 0.0,
            }

    # Theoretical minimum loss (perfect foresight)
    opt_iron, opt_copper = _compute_loss_components(load, N_optimal)
    theoretical_loss_kwh = float(np.sum(opt_iron + opt_copper) / 6.0)

    # Actual loss
    act_iron, act_copper = _compute_loss_components(load, N_actual)
    overload_mask = load > N_actual * Sr
    act_total = np.where(overload_mask, 1000.0, act_iron + act_copper)
    actual_loss_kwh = float(np.sum(act_total) / 6.0)

    optimality_gap = actual_loss_kwh - theoretical_loss_kwh
    optimality_gap_pct = (optimality_gap / theoretical_loss_kwh * 100) if theoretical_loss_kwh > 0 else 0.0

    return {
        'time_distribution': {str(k): v for k, v in time_dist_actual.items()},
        'theoretical_optimal': {str(k): v for k, v in time_dist_optimal.items()},
        'regime_match_pct': regime_match_pct,
        'confusion_matrix': confusion,
        'regime_metrics': {str(k): v for k, v in regime_metrics.items()},
        'theoretical_loss_kwh': theoretical_loss_kwh,
        'actual_loss_kwh': actual_loss_kwh,
        'optimality_gap_kwh': optimality_gap,
        'optimality_gap_pct': optimality_gap_pct,
        'breakpoints': {'bp_1_2': bp_1_2, 'bp_2_3': bp_2_3},
    }


def analyze_computational_timing(df):
    """
    MPC solver timing analysis.

    Args:
        df: DataFrame from logs.csv (may or may not have mpc_solve_time_ms column)

    Returns:
        dict with timing statistics, or None if column not present
    """
    if 'mpc_solve_time_ms' not in df.columns:
        return None

    times = df['mpc_solve_time_ms'].dropna().values
    if len(times) == 0:
        return None

    return {
        'mean_ms': float(np.mean(times)),
        'std_ms': float(np.std(times)),
        'median_ms': float(np.median(times)),
        'max_ms': float(np.max(times)),
        'min_ms': float(np.min(times)),
        'p95_ms': float(np.percentile(times, 95)),
        'p99_ms': float(np.percentile(times, 99)),
        'total_calls': int(len(times)),
        'total_time_s': float(np.sum(times) / 1000.0),
    }


def run_mpc_analysis(logs_path, switching_events, outdir):
    """
    Main entry point for MPC performance analysis.

    Loads logs, runs all 4 sub-analyses, saves results as JSON,
    and generates visualization plots.

    Args:
        logs_path: Path to logs.csv from backtesting
        switching_events: list of dicts with keys:
            {timestamp, from_N, to_N, mpc_decision, safety_override}
        outdir: Output directory for results and plots

    Returns:
        dict: Complete analysis results (also saved as mpc_analysis_results.json)
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print("\nLoading backtesting logs...")
    df = pd.read_csv(logs_path)
    print(f"  Loaded {len(df)} timesteps from logs.csv")

    # 1. Baseline comparison with loss decomposition
    print("\n[1/4] Analyzing baseline comparison (iron/copper decomposition)...")
    baseline_results = analyze_baseline_comparison(df)
    print(f"  MPC total loss:      {baseline_results['mpc_total_loss_kwh']:.1f} kWh")
    print(f"    Iron losses:       {baseline_results['mpc_iron_loss_kwh']:.1f} kWh")
    print(f"    Copper losses:     {baseline_results['mpc_copper_loss_kwh']:.1f} kWh")
    print(f"  Baseline total loss: {baseline_results['baseline_total_loss_kwh']:.1f} kWh")
    print(f"    Iron losses:       {baseline_results['baseline_iron_loss_kwh']:.1f} kWh")
    print(f"    Copper losses:     {baseline_results['baseline_copper_loss_kwh']:.1f} kWh")
    print(f"  Iron savings:        {baseline_results['iron_savings_kwh']:.1f} kWh ({baseline_results['iron_savings_pct']:.1f}%)")
    print(f"  Copper savings:      {baseline_results['copper_savings_kwh']:.1f} kWh ({baseline_results['copper_savings_pct']:.1f}%)")
    print(f"  Total savings:       {baseline_results['total_savings_kwh']:.1f} kWh ({baseline_results['total_savings_pct']:.1f}%)")

    # 2. Switching statistics
    print("\n[2/4] Analyzing switching statistics...")
    switching_results = analyze_switching_statistics(df, switching_events)
    print(f"  Total switches:      {switching_results['total_switches']}")
    print(f"  Switches/day:        {switching_results['switches_per_day']:.2f}")
    print(f"  Switches/week:       {switching_results['switches_per_week']:.2f}")
    print(f"  Switches/month:      {switching_results['switches_per_month']:.1f}")
    print(f"  Up transitions:      {switching_results['up_transitions']}")
    print(f"  Down transitions:    {switching_results['down_transitions']}")
    print(f"  Safety overrides:    {switching_results['safety_overrides']}")
    print(f"  Mean dwell time:     {switching_results['dwell_times']['mean_minutes']:.0f} min")
    print(f"  Median dwell time:   {switching_results['dwell_times']['median_minutes']:.0f} min")

    # 3. Operational regime analysis
    print("\n[3/4] Analyzing operational regimes...")
    regime_results = analyze_operational_regimes(df)
    print(f"  Actual time distribution:")
    for n in ['1', '2', '3']:
        d = regime_results['time_distribution'][n]
        print(f"    N={n}: {d['pct']:.1f}% ({d['hours']:.1f} hours)")
    print(f"  Theoretical optimal:")
    for n in ['1', '2', '3']:
        d = regime_results['theoretical_optimal'][n]
        print(f"    N={n}: {d['pct']:.1f}%")
    print(f"  Regime match rate:   {regime_results['regime_match_pct']:.1f}%")
    print(f"  Optimality gap:      {regime_results['optimality_gap_kwh']:.1f} kWh ({regime_results['optimality_gap_pct']:.1f}%)")

    # 4. Computational timing
    print("\n[4/4] Analyzing computational timing...")
    timing_results = analyze_computational_timing(df)
    if timing_results is not None:
        print(f"  Mean solve time:     {timing_results['mean_ms']:.2f} ms")
        print(f"  Std solve time:      {timing_results['std_ms']:.2f} ms")
        print(f"  Max solve time:      {timing_results['max_ms']:.2f} ms")
        print(f"  P95 solve time:      {timing_results['p95_ms']:.2f} ms")
        print(f"  Total MPC time:      {timing_results['total_time_s']:.1f} s")
    else:
        print("  No timing data available (mpc_solve_time_ms column not in logs)")

    # Config snapshot
    bp_1_2, bp_2_3 = _compute_raw_breakpoints()
    config_snapshot = {
        'P0': CONFIG['P0'],
        'Ploss_rated': CONFIG['Ploss_rated'],
        'Sr': CONFIG['Sr'],
        'kappa': CONFIG['kappa'],
        'T': CONFIG['T'],
        'M': CONFIG['M'],
        'min_on_time': CONFIG['min_on_time'],
        'bp_1_2': round(bp_1_2, 2),
        'bp_2_3': round(bp_2_3, 2),
    }

    # Assemble results
    analysis_results = {
        'baseline_comparison': baseline_results,
        'switching_statistics': switching_results,
        'operational_regimes': regime_results,
        'timing': timing_results,
        'config_snapshot': config_snapshot,
    }

    # Save JSON
    json_path = outdir / 'mpc_analysis_results.json'
    with open(json_path, 'w') as f:
        json.dump(analysis_results, f, indent=2, default=str)
    print(f"\n[OK] Analysis results saved to {json_path}")

    # Generate plots
    try:
        from visualization.mpc_plots import plot_mpc_analysis
        plot_mpc_analysis(analysis_results, outdir)
    except Exception as e:
        print(f"[WARNING] Could not generate MPC analysis plots: {e}")
        import traceback
        traceback.print_exc()

    return analysis_results
