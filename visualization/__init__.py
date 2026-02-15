"""
Visualization Module
====================

This module contains all plotting and visualization functions for:
- Training metrics and model evaluation
- Prediction analysis and uncertainty quantification
- Backtest results and KPI dashboards
- Data analysis and exploratory visualizations

Exports:
    plot_training_curves: Plot training/validation loss curves with metrics
    plot_predictions: Plot model predictions vs actual with uncertainty bands
    plot_multistep_predictions: Plot full forecast horizons for multiple sequences
    generate_summary_plots: Generate backtest analysis and KPI summary plots
    generate_all_analysis_plots: Generate all data analysis visualization figures
    plot_load_duration_curve: Load Duration Curve (LDC) analysis
    plot_temporal_heatmap: Day-of-week vs Hour-of-day heatmap
    plot_distribution_analysis: Distribution with KDE and Gaussian fit
    plot_seasonal_decomposition: STL decomposition (Trend, Seasonal, Residual)
    plot_power_factor_analysis: Reactive power vs Power Factor scatter
"""

from visualization.training_plots import (
    plot_training_curves,
    plot_predictions,
    plot_multistep_predictions
)

from visualization.results_plots import generate_summary_plots

from visualization.data_analysis_plots import (
    generate_all_analysis_plots,
    plot_load_duration_curve,
    plot_temporal_heatmap,
    plot_distribution_analysis,
    plot_seasonal_decomposition,
    plot_power_factor_analysis,
    plot_yearly_seasonality
)

from visualization.mpc_plots import plot_mpc_analysis
from visualization.safety_plots import plot_safety_analysis
from visualization.paper_figures import generate_paper_figures

__all__ = [
    'plot_training_curves',
    'plot_predictions',
    'plot_multistep_predictions',
    'generate_summary_plots',
    'generate_all_analysis_plots',
    'plot_load_duration_curve',
    'plot_temporal_heatmap',
    'plot_distribution_analysis',
    'plot_seasonal_decomposition',
    'plot_power_factor_analysis',
    'plot_yearly_seasonality',
    'plot_mpc_analysis',
    'plot_safety_analysis',
    'generate_paper_figures',
]
