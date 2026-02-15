"""
Control Package

This package contains the Model Predictive Control (MPC) system for transformer switching
optimization. It includes:

- Power loss modeling and breakpoint calculations
- Probabilistic forecasting with trajectory sampling
- Stochastic MPC optimization
- Rule-based safety layer for constraint enforcement

Key Components:
- power_model: Transformer loss calculations and switching breakpoints
- forecasting: Trajectory sampling from quantile predictions
- mpc: Stochastic MPC optimization algorithm
- safety: Safety layer with hysteresis and constraint enforcement
"""

from .power_model import compute_breakpoints_and_thresholds, transformer_loss_model
from .forecasting import sample_trajectories_from_quantiles
from .mpc import stochastic_mpc
from .safety import safety_layer

__all__ = [
    'compute_breakpoints_and_thresholds',
    'transformer_loss_model',
    'sample_trajectories_from_quantiles',
    'stochastic_mpc',
    'safety_layer',
]
