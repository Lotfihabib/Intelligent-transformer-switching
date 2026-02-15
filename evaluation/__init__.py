"""
Evaluation Package

Contains all evaluation and analysis functionality:
- Metrics computation (point and probabilistic)
- Backtesting simulation
- Error analysis
"""

from .metrics import compute_evaluation_metrics
from .backtesting import run_backtest
from .analysis import analyze_median_prediction_errors
from .mpc_analysis import run_mpc_analysis
from .safety_analysis import run_safety_analysis

__all__ = [
    'compute_evaluation_metrics',
    'run_backtest',
    'analyze_median_prediction_errors',
    'run_mpc_analysis',
    'run_safety_analysis',
]
