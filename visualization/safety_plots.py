"""
Safety Layer Visualization Module

Generates multi-panel plots for safety layer analysis:
1. Rule activation summary (per-rule frequency, override types, cost, summary)
2. Safety temporal patterns (hourly/daily distributions, uncertainty, loading ratios)
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


plt.style.use('default')


RULE_NAMES = {
    '1': 'R1: Min Transformer',
    '2': 'R2: High Load',
    '3': 'R3: Dwell Time',
    '4': 'R4: Quantile Guards',
    '5': 'R5: Uncertainty',
    '6': 'R6: Overload',
}

RULE_COLORS = {
    '1': '#42A5F5',
    '2': '#EF5350',
    '3': '#FF9800',
    '4': '#AB47BC',
    '5': '#26A69A',
    '6': '#FFA726',
}


def plot_safety_analysis(analysis_results, outdir):
    """
    Orchestrator: generate all safety analysis plots.

    Args:
        analysis_results: dict from run_safety_analysis()
        outdir: Output directory path
    """
    outdir = Path(outdir)

    plot_rule_activation_summary(
        analysis_results['override_statistics'],
        analysis_results['safety_optimality'],
        outdir,
    )

    plot_safety_temporal_patterns(
        analysis_results['temporal_analysis'],
        analysis_results['constraint_validation'],
        outdir,
    )


def plot_rule_activation_summary(override_data, optimality_data, outdir):
    """
    2x2 panel: per-rule frequency, override type pie, per-rule cost, summary text.

    Args:
        override_data: dict from analyze_override_statistics()
        optimality_data: dict from analyze_safety_optimality()
        outdir: Output directory
    """
    outdir = Path(outdir)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Safety Layer Rule Activation Summary', fontsize=14, fontweight='bold', y=0.98)

    # Panel 1: Per-rule activation frequency (horizontal bar)
    ax = axes[0, 0]
    rule_nums = [str(i) for i in range(1, 7)]
    counts = [override_data['per_rule_counts'].get(r, 0) for r in rule_nums]
    labels = [RULE_NAMES.get(r, f'Rule {r}') for r in rule_nums]
    colors = [RULE_COLORS.get(r, '#90A4AE') for r in rule_nums]

    y_pos = np.arange(len(rule_nums))
    bars = ax.barh(y_pos, counts, color=colors, edgecolor='white', height=0.6)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel('Activation Count')
    ax.set_title('Per-Rule Activation Frequency')
    ax.invert_yaxis()

    # Add count labels on bars
    for bar, count in zip(bars, counts):
        if count > 0:
            ax.text(bar.get_width() + max(counts) * 0.02, bar.get_y() + bar.get_height() / 2,
                    str(count), va='center', fontsize=9, fontweight='bold')

    # Panel 2: Override type breakdown (pie chart)
    ax = axes[0, 1]
    override_types = override_data['override_types']
    type_labels = []
    type_sizes = []
    type_colors = []

    type_map = {
        'blocked_switch': ('Blocked Switch', '#EF5350'),
        'forced_switch': ('Forced Switch', '#42A5F5'),
        'modified_switch': ('Modified Switch', '#FF9800'),
    }

    for key, (label, color) in type_map.items():
        val = override_types.get(key, 0)
        if val > 0:
            type_labels.append(f'{label}\n({val})')
            type_sizes.append(val)
            type_colors.append(color)

    if sum(type_sizes) > 0:
        wedges, texts, autotexts = ax.pie(
            type_sizes, labels=type_labels, colors=type_colors,
            autopct='%1.1f%%', startangle=90, textprops={'fontsize': 9}
        )
        for autotext in autotexts:
            autotext.set_fontsize(9)
            autotext.set_fontweight('bold')
    else:
        ax.text(0.5, 0.5, 'No Overrides', transform=ax.transAxes,
                ha='center', va='center', fontsize=12, color='gray')
    ax.set_title('Override Type Breakdown')

    # Panel 3: Per-rule cost of safety (horizontal bar, color-coded)
    ax = axes[1, 0]
    per_rule_cost = optimality_data.get('per_rule_cost_kwh', {})
    cost_values = [per_rule_cost.get(r, 0.0) for r in rule_nums]
    bar_colors = ['#EF5350' if v > 0 else '#66BB6A' for v in cost_values]

    bars = ax.barh(y_pos, cost_values, color=bar_colors, edgecolor='white', height=0.6)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel('Cost of Safety (kWh)')
    ax.set_title('Per-Rule Cost of Safety')
    ax.invert_yaxis()
    ax.axvline(x=0, color='black', linewidth=0.8, linestyle='-')

    # Add value labels
    for bar, val in zip(bars, cost_values):
        if abs(val) > 0.01:
            x_pos = bar.get_width()
            ha = 'left' if val >= 0 else 'right'
            offset = max(abs(v) for v in cost_values) * 0.02 if cost_values else 0
            ax.text(x_pos + (offset if val >= 0 else -offset),
                    bar.get_y() + bar.get_height() / 2,
                    f'{val:.1f}', va='center', ha=ha, fontsize=8, fontweight='bold')

    # Panel 4: Summary statistics text box
    ax = axes[1, 1]
    total_decisions = override_data.get('total_decisions', 0)
    override_count = override_data.get('override_count', 0)
    override_pct = override_data.get('override_pct', 0)
    total_cost = optimality_data.get('total_cost_of_safety_kwh', 0)
    mean_cost = optimality_data.get('mean_cost_per_override_kw', 0)
    cost_pct = optimality_data.get('cost_as_pct_of_savings', 0)

    blocked = override_types.get('blocked_switch', 0)
    forced = override_types.get('forced_switch', 0)
    modified = override_types.get('modified_switch', 0)

    stats_text = (
        f"SAFETY LAYER SUMMARY\n"
        f"{'='*35}\n\n"
        f"Total Decisions:      {total_decisions:,}\n"
        f"Total Overrides:      {override_count:,}\n"
        f"Override Rate:        {override_pct:.1f}%\n\n"
        f"Override Types:\n"
        f"  Blocked Switches:   {blocked:,}\n"
        f"  Forced Switches:    {forced:,}\n"
        f"  Modified Switches:  {modified:,}\n\n"
        f"Cost of Safety:\n"
        f"  Total:              {total_cost:.1f} kWh\n"
        f"  Mean per Override:  {mean_cost:.2f} kW\n"
        f"  % of Savings:       {cost_pct:.1f}%\n"
    )

    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', family='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))
    ax.axis('off')
    ax.set_title('Summary Statistics')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(outdir / 'safety_rule_activations.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_safety_temporal_patterns(temporal_data, validation_data, outdir):
    """
    2x2 panel: hourly/daily override patterns, uncertainty comparison, loading ratios.

    Args:
        temporal_data: dict from analyze_safety_temporal()
        validation_data: dict from analyze_constraint_validation()
        outdir: Output directory
    """
    outdir = Path(outdir)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Safety Layer Temporal Patterns & Constraint Validation',
                 fontsize=14, fontweight='bold', y=0.98)

    # Panel 1: Hourly override distribution
    ax = axes[0, 0]
    hourly = temporal_data.get('hourly_distribution', {})
    hours = list(range(24))
    hour_counts = [hourly.get(str(h), 0) for h in hours]

    ax.bar(hours, hour_counts, color='#42A5F5', edgecolor='white', width=0.8)
    ax.set_xlabel('Hour of Day')
    ax.set_ylabel('Override Count')
    ax.set_title('Hourly Override Distribution')
    ax.set_xticks(range(0, 24, 2))
    ax.set_xlim(-0.5, 23.5)

    # Panel 2: Daily override distribution
    ax = axes[0, 1]
    daily = temporal_data.get('daily_distribution', {})
    day_labels = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    day_counts = [daily.get(str(d), 0) for d in range(7)]
    day_colors = ['#42A5F5'] * 5 + ['#66BB6A'] * 2  # Weekday blue, weekend green

    ax.bar(range(7), day_counts, color=day_colors, edgecolor='white', width=0.7)
    ax.set_xticks(range(7))
    ax.set_xticklabels(day_labels)
    ax.set_ylabel('Override Count')
    ax.set_title('Daily Override Distribution')

    # Panel 3: Uncertainty at override vs non-override (box plot style)
    ax = axes[1, 0]
    unc_override = temporal_data.get('uncertainty_at_override', {})
    unc_no_override = temporal_data.get('uncertainty_at_no_override', {})

    has_data = (unc_override.get('mean') is not None and
                unc_no_override.get('mean') is not None)

    if has_data:
        categories = ['At Override', 'No Override']
        means = [unc_override['mean'], unc_no_override['mean']]
        medians = [unc_override['median'], unc_no_override['median']]
        stds = [unc_override.get('std', 0), unc_no_override.get('std', 0)]

        x = np.arange(len(categories))
        width = 0.3

        bars_mean = ax.bar(x - width / 2, means, width, label='Mean',
                           color='#42A5F5', edgecolor='white')
        bars_median = ax.bar(x + width / 2, medians, width, label='Median',
                             color='#FF9800', edgecolor='white')

        # Error bars for std
        ax.errorbar(x - width / 2, means, yerr=stds, fmt='none',
                     ecolor='black', capsize=4, capthick=1.5)

        ax.set_xticks(x)
        ax.set_xticklabels(categories)
        ax.set_ylabel('Uncertainty (q90 - q10) [MVA]')
        ax.legend(fontsize=9)

        # Add value labels
        for bar, val in zip(bars_mean, means):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                    f'{val:.2f}', ha='center', va='bottom', fontsize=9)
        for bar, val in zip(bars_median, medians):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                    f'{val:.2f}', ha='center', va='bottom', fontsize=9)
    else:
        ax.text(0.5, 0.5, 'No forecast data available',
                transform=ax.transAxes, ha='center', va='center',
                fontsize=12, color='gray')
    ax.set_title('Forecast Uncertainty: Override vs Non-Override')

    # Panel 4: Loading ratio distribution
    ax = axes[1, 1]
    ratio_stats = validation_data.get('loading_ratio_stats', {})

    if ratio_stats:
        # Create synthetic histogram from stats
        mean_r = ratio_stats.get('mean', 0)
        std_r = ratio_stats.get('std', 0)
        p95 = ratio_stats.get('p95', 0)
        p99 = ratio_stats.get('p99', 0)
        max_r = ratio_stats.get('max', 0)

        # Summary text instead of full histogram (we don't have raw data)
        stats_text = (
            f"LOADING RATIO STATISTICS\n"
            f"{'='*30}\n\n"
            f"Mean:     {mean_r:.4f}\n"
            f"Median:   {ratio_stats.get('median', 0):.4f}\n"
            f"Std Dev:  {std_r:.4f}\n"
            f"P95:      {p95:.4f}\n"
            f"P99:      {p99:.4f}\n"
            f"Max:      {max_r:.4f}\n\n"
            f"Max Load: {validation_data.get('max_loading_load_mva', 0):.1f} MVA "
            f"(N={validation_data.get('max_loading_N', '?')})\n"
            f"At:       {validation_data.get('max_loading_timestamp', 'N/A')}\n\n"
            f"Overload Violations: {validation_data.get('overload_count', 0)}\n"
            f"Near-Violations:     {validation_data.get('near_violation_count', 0)}\n"
            f"  (loading ratio > 0.85)"
        )

        ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', family='monospace',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

        # Visual indicator bar using plot (axhline doesn't support transform)
        bar_y = 0.08
        ax.plot([0.05, 0.95], [bar_y, bar_y], color='lightgray',
                linewidth=8, solid_capstyle='round', transform=ax.transAxes)

        # Mark key points on the indicator
        for val, label, color in [(mean_r, 'Mean', '#42A5F5'),
                                   (p95, 'P95', '#FF9800'),
                                   (max_r, 'Max', '#EF5350'),
                                   (1.0, 'Limit', 'black')]:
            x_pos = 0.05 + 0.9 * min(val / 1.1, 1.0)
            ax.plot(x_pos, bar_y, 'v', color=color, markersize=10,
                    transform=ax.transAxes)
            ax.text(x_pos, bar_y - 0.04, label, ha='center', fontsize=7,
                    color=color, transform=ax.transAxes, fontweight='bold')

    ax.axis('off')
    ax.set_title('Loading Ratio & Constraint Validation')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(outdir / 'safety_temporal_patterns.png', dpi=300, bbox_inches='tight')
    plt.close()
