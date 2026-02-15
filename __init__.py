"""
Grid Transformer Switching Optimization System

A modular implementation of stochastic Model Predictive Control (MPC)
with probabilistic load forecasting for optimizing power transformer operations.
"""

__version__ = "1.0.0"

# Re-export key components for easy access
from .config import CONFIG
from .evaluation import compute_evaluation_metrics, run_backtest, analyze_median_prediction_errors

__all__ = [
    'CONFIG',
    'compute_evaluation_metrics',
    'run_backtest',
    'analyze_median_prediction_errors',
]
