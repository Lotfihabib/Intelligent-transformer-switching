"""
Safety Layer Analysis Module

Post-hoc analysis of safety layer rule activations and overrides:
1. Override statistics (per-rule frequency, override types)
2. Temporal analysis (hourly/daily patterns, uncertainty correlation)
3. Safety-optimality trade-off (cost of safety per override)
4. Constraint validation (loading ratios, overload verification)

All analysis functions work from logs.csv data (with safety_rules and prev_N columns),
requiring no access to the trained model or raw data.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter

from config import CONFIG


def _compute_total_loss(load_mva, N):
    """
    Compute total transformer losses for given load and configuration (vectorized).

    Args:
        load_mva: float or np.ndarray, total load in MVA
        N: int or np.ndarray, number of active transformers

    Returns:
        float or np.ndarray: total loss in kW
    """
    P0 = CONFIG['P0']
    Ploss_rated = CONFIG['Ploss_rated']
    Sr = CONFIG['Sr']

    N = np.asarray(N, dtype=float)
    load_mva = np.asarray(load_mva, dtype=float)

    iron_loss = N * P0
    copper_loss = N * Ploss_rated * (load_mva / (N * Sr)) ** 2

    return iron_loss + copper_loss


def analyze_override_statistics(df):
    """
    Compute aggregate override statistics from safety rule activations.

    Args:
        df: DataFrame with columns [N_mpc, N_current, prev_N, safety_rules]

    Returns:
        dict: Override statistics including per-rule counts and override types
    """
    total_decisions = len(df)

    # Identify overrides: rows where safety modified the MPC decision
    override_mask = df['safety_rules'].astype(str).str.strip() != ''
    override_count = override_mask.sum()
    override_pct = (override_count / total_decisions * 100) if total_decisions > 0 else 0.0

    # Per-rule activation frequency
    per_rule_counts = {i: 0 for i in range(1, 7)}
    for rules_str in df.loc[override_mask, 'safety_rules']:
        rules_str = str(rules_str).strip()
        if rules_str:
            for r in rules_str.split(','):
                r = r.strip()
                try:
                    rule_num = int(float(r))
                    if 1 <= rule_num <= 6:
                        per_rule_counts[rule_num] += 1
                except (ValueError, TypeError):
                    continue

    # Override type classification
    override_rows = df[override_mask].copy()
    blocked = 0
    forced = 0
    modified = 0

    if len(override_rows) > 0:
        mpc_wanted_switch = override_rows['N_mpc'] != override_rows['prev_N']
        safety_kept = override_rows['N_current'] == override_rows['prev_N']
        safety_switched = override_rows['N_current'] != override_rows['prev_N']
        mpc_stayed = override_rows['N_mpc'] == override_rows['prev_N']
        mpc_diff_safety = override_rows['N_mpc'] != override_rows['N_current']

        # Blocked: MPC wanted switch, safety kept current
        blocked = int((mpc_wanted_switch & safety_kept).sum())
        # Forced: MPC wanted to stay, safety forced switch
        forced = int((mpc_stayed & safety_switched).sum())
        # Modified: Both switched but to different N
        modified = int((mpc_wanted_switch & safety_switched & mpc_diff_safety).sum())

    rule_names = {
        1: 'Minimum Transformer',
        2: 'High Load Protection',
        3: 'Dwell Time Enforcement',
        4: 'Quantile Guards',
        5: 'Uncertainty Deferral',
        6: 'Overload Protection',
    }

    return {
        'total_decisions': total_decisions,
        'override_count': int(override_count),
        'override_pct': round(override_pct, 2),
        'per_rule_counts': {str(k): v for k, v in per_rule_counts.items()},
        'per_rule_names': rule_names,
        'override_types': {
            'blocked_switch': blocked,
            'forced_switch': forced,
            'modified_switch': modified,
        },
    }


def analyze_safety_temporal(df):
    """
    Analyze temporal patterns of safety layer activations.

    Args:
        df: DataFrame with columns [timestamp, safety_rules, forecast_q10, forecast_q90]

    Returns:
        dict: Temporal distributions and uncertainty correlation
    """
    ts = pd.to_datetime(df['timestamp'])
    override_mask = df['safety_rules'].astype(str).str.strip() != ''

    # Hourly distribution of overrides
    override_hours = ts[override_mask].dt.hour
    hourly_dist = {str(h): 0 for h in range(24)}
    for h, count in override_hours.value_counts().items():
        hourly_dist[str(h)] = int(count)

    # Daily distribution (0=Mon, 6=Sun)
    override_days = ts[override_mask].dt.dayofweek
    day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    daily_dist = {str(d): 0 for d in range(7)}
    for d, count in override_days.value_counts().items():
        daily_dist[str(d)] = int(count)
    daily_dist_named = {day_names[int(k)]: v for k, v in daily_dist.items()}

    # Uncertainty correlation
    has_forecasts = 'forecast_q10' in df.columns and 'forecast_q90' in df.columns
    uncertainty_at_override = {'mean': None, 'median': None, 'std': None}
    uncertainty_at_no_override = {'mean': None, 'median': None, 'std': None}

    if has_forecasts:
        # Only consider rows with valid forecast data
        valid_mask = df['forecast_q10'].notna() & df['forecast_q90'].notna()
        uncertainty = df.loc[valid_mask, 'forecast_q90'] - df.loc[valid_mask, 'forecast_q10']

        override_unc = uncertainty[override_mask & valid_mask]
        no_override_unc = uncertainty[~override_mask & valid_mask]

        if len(override_unc) > 0:
            uncertainty_at_override = {
                'mean': round(float(override_unc.mean()), 3),
                'median': round(float(override_unc.median()), 3),
                'std': round(float(override_unc.std()), 3),
            }
        if len(no_override_unc) > 0:
            uncertainty_at_no_override = {
                'mean': round(float(no_override_unc.mean()), 3),
                'median': round(float(no_override_unc.median()), 3),
                'std': round(float(no_override_unc.std()), 3),
            }

    return {
        'hourly_distribution': hourly_dist,
        'daily_distribution': daily_dist,
        'daily_distribution_named': daily_dist_named,
        'uncertainty_at_override': uncertainty_at_override,
        'uncertainty_at_no_override': uncertainty_at_no_override,
    }


def analyze_safety_optimality(df):
    """
    Quantify the cost of safety layer overrides.

    For each override, computes the loss difference between the safe decision
    and the MPC recommendation using actual realized load.

    Args:
        df: DataFrame with columns [S_TOTAL, N_mpc, N_current, loss_kw, safety_rules]

    Returns:
        dict: Cost of safety metrics
    """
    override_mask = df['safety_rules'].astype(str).str.strip() != ''
    override_rows = df[override_mask].copy()

    if len(override_rows) == 0:
        return {
            'total_cost_of_safety_kwh': 0.0,
            'mean_cost_per_override_kw': 0.0,
            'override_count': 0,
            'per_rule_cost_kwh': {str(i): 0.0 for i in range(1, 7)},
            'cost_as_pct_of_savings': 0.0,
        }

    # Loss at safe decision (already in logs)
    loss_at_safe = override_rows['loss_kw'].values

    # Loss at MPC recommendation (counterfactual)
    load_mva = override_rows['S_TOTAL'].values
    n_mpc = override_rows['N_mpc'].values

    # Handle overload case for MPC recommendation
    Sr = CONFIG['Sr']
    mpc_overload = load_mva > (n_mpc * Sr)
    loss_at_mpc = np.where(
        mpc_overload,
        1000.0,
        _compute_total_loss(load_mva, np.maximum(n_mpc, 1))
    )

    # Cost delta: positive means safety costs more energy
    cost_delta_kw = loss_at_safe - loss_at_mpc
    cost_delta_kwh = cost_delta_kw / 6.0  # 10-min intervals

    total_cost_kwh = float(cost_delta_kwh.sum())
    mean_cost_kw = float(cost_delta_kw.mean())

    # Per-rule cost breakdown
    per_rule_cost_kwh = {str(i): 0.0 for i in range(1, 7)}
    for idx, rules_str in zip(override_rows.index, override_rows['safety_rules']):
        rules_str = str(rules_str).strip()
        if rules_str:
            row_cost = cost_delta_kwh[override_rows.index.get_loc(idx)]
            for r in rules_str.split(','):
                r = r.strip()
                try:
                    rule_num = int(float(r))
                    if 1 <= rule_num <= 6:
                        per_rule_cost_kwh[str(rule_num)] += float(row_cost)
                except (ValueError, TypeError):
                    continue

    # Round per-rule costs
    per_rule_cost_kwh = {k: round(v, 2) for k, v in per_rule_cost_kwh.items()}

    # Cost as percentage of total savings
    total_loss_kwh = df['loss_kwh'].sum()
    baseline_loss = _compute_total_loss(df['S_TOTAL'].values, np.full(len(df), 3))
    baseline_loss_kwh = float(baseline_loss.sum() / 6.0)
    savings_kwh = baseline_loss_kwh - total_loss_kwh
    cost_as_pct = (abs(total_cost_kwh) / savings_kwh * 100) if savings_kwh > 0 else 0.0

    return {
        'total_cost_of_safety_kwh': round(total_cost_kwh, 2),
        'mean_cost_per_override_kw': round(mean_cost_kw, 2),
        'override_count': len(override_rows),
        'per_rule_cost_kwh': per_rule_cost_kwh,
        'cost_as_pct_of_savings': round(cost_as_pct, 2),
    }


def analyze_constraint_validation(df):
    """
    Validate safety constraint compliance and identify near-violation events.

    Args:
        df: DataFrame with columns [timestamp, S_TOTAL, N_current]

    Returns:
        dict: Constraint validation results
    """
    Sr = CONFIG['Sr']
    load = df['S_TOTAL'].values
    N = df['N_current'].values

    loading_ratio = load / (N * Sr)

    # Max loading ratio
    max_idx = np.argmax(loading_ratio)
    max_ratio = float(loading_ratio[max_idx])
    max_timestamp = str(df.iloc[max_idx]['timestamp'])
    max_load = float(load[max_idx])
    max_N = int(N[max_idx])

    # Overload verification
    overload_count = int((loading_ratio > 1.0).sum())

    # Near-violation events (loading ratio > 0.85)
    near_mask = loading_ratio > 0.85
    near_count = int(near_mask.sum())

    # Top 10 closest-to-violation
    near_indices = np.where(near_mask)[0]
    if len(near_indices) > 0:
        sorted_near = near_indices[np.argsort(loading_ratio[near_indices])[::-1]][:10]
        near_violations = []
        for idx in sorted_near:
            near_violations.append({
                'timestamp': str(df.iloc[idx]['timestamp']),
                'load_mva': round(float(load[idx]), 2),
                'N': int(N[idx]),
                'loading_ratio': round(float(loading_ratio[idx]), 4),
            })
    else:
        near_violations = []

    # Loading ratio distribution stats
    ratio_stats = {
        'mean': round(float(np.mean(loading_ratio)), 4),
        'median': round(float(np.median(loading_ratio)), 4),
        'std': round(float(np.std(loading_ratio)), 4),
        'p95': round(float(np.percentile(loading_ratio, 95)), 4),
        'p99': round(float(np.percentile(loading_ratio, 99)), 4),
        'max': round(max_ratio, 4),
    }

    return {
        'max_loading_ratio': round(max_ratio, 4),
        'max_loading_timestamp': max_timestamp,
        'max_loading_load_mva': round(max_load, 2),
        'max_loading_N': max_N,
        'overload_count': overload_count,
        'near_violation_count': near_count,
        'near_violations_top10': near_violations,
        'loading_ratio_stats': ratio_stats,
    }


def run_safety_analysis(logs_path, switching_events, outdir):
    """
    Run complete safety layer analysis.

    Args:
        logs_path: Path to logs.csv from backtesting
        switching_events: List of switching event dicts from backtesting
        outdir: Output directory for results and plots

    Returns:
        dict: Complete analysis results (also saved as JSON)
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print("\n--- Safety Layer Analysis ---")

    # Load logs
    df = pd.read_csv(logs_path)
    print(f"Loaded {len(df)} timesteps from {logs_path}")

    # Check for required columns
    if 'safety_rules' not in df.columns:
        print("[WARNING] 'safety_rules' column not found in logs. "
              "Re-run backtesting to generate safety rule data.")
        return None

    if 'prev_N' not in df.columns:
        print("[WARNING] 'prev_N' column not found in logs. "
              "Re-run backtesting to generate prev_N data.")
        return None

    # Fill NaN safety_rules with empty string and ensure string type
    df['safety_rules'] = df['safety_rules'].fillna('').astype(str)
    # Handle pandas float-reading: "nan" strings from NaN conversion
    df.loc[df['safety_rules'] == 'nan', 'safety_rules'] = ''

    # 1. Override statistics
    print("  Computing override statistics...")
    override_results = analyze_override_statistics(df)
    print(f"    Total overrides: {override_results['override_count']} "
          f"({override_results['override_pct']:.1f}%)")
    for rule_num in range(1, 7):
        count = override_results['per_rule_counts'][str(rule_num)]
        if count > 0:
            print(f"    Rule {rule_num}: {count} activations")

    # 2. Temporal analysis
    print("  Computing temporal analysis...")
    temporal_results = analyze_safety_temporal(df)

    # 3. Safety-optimality trade-off
    print("  Computing safety-optimality trade-off...")
    optimality_results = analyze_safety_optimality(df)
    print(f"    Cost of safety: {optimality_results['total_cost_of_safety_kwh']:.1f} kWh "
          f"({optimality_results['cost_as_pct_of_savings']:.1f}% of savings)")

    # 4. Constraint validation
    print("  Computing constraint validation...")
    validation_results = analyze_constraint_validation(df)
    print(f"    Max loading ratio: {validation_results['max_loading_ratio']:.3f}")
    print(f"    Overload violations: {validation_results['overload_count']}")
    print(f"    Near-violations (>0.85): {validation_results['near_violation_count']}")

    # Assemble results
    analysis_results = {
        'override_statistics': override_results,
        'temporal_analysis': temporal_results,
        'safety_optimality': optimality_results,
        'constraint_validation': validation_results,
        'config_snapshot': {
            'min_on_time': CONFIG['min_on_time'],
            'min_off_time': CONFIG['min_off_time'],
            'uncertainty_limit': CONFIG['uncertainty_limit'],
            'high_load_threshold': CONFIG['high_load_threshold'],
            'overload_margin': CONFIG['overload_margin'],
            'Sr': CONFIG['Sr'],
        },
    }

    # Save JSON
    json_path = outdir / 'safety_analysis_results.json'
    with open(json_path, 'w') as f:
        json.dump(analysis_results, f, indent=2, default=str)
    print(f"  Results saved to {json_path}")

    # Generate plots
    try:
        from visualization.safety_plots import plot_safety_analysis
        plot_safety_analysis(analysis_results, outdir)
        print("  Plots generated successfully")
    except Exception as e:
        print(f"  [WARNING] Plot generation failed: {e}")
        import traceback
        traceback.print_exc()

    return analysis_results
