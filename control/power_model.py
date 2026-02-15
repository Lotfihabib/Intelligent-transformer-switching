"""
Power Model Module

This module implements transformer power loss calculations and switching breakpoint
determination. It provides the physical model that underlies the optimization problem.

Key Functions:
- compute_breakpoints_and_thresholds: Calculate optimal switching points with hysteresis
- transformer_loss_model: Calculate transformer losses based on load and configuration

Mathematical Background:
- Switching breakpoints are where loss curves intersect (1→2, 2→3 transformers)
- Hysteresis buffers prevent rapid oscillation between states
- Loss model: Loss(N, L) = N × [P0 + Ploss_rated × (L/N/Sr)²]
"""

import math
from config import CONFIG


def compute_breakpoints_and_thresholds():
    """Compute switching breakpoints and hysteresis thresholds"""
    P0, Ploss_rated, Sr = CONFIG['P0'], CONFIG['Ploss_rated'], CONFIG['Sr']

    # Breakpoints
    bp_1_2 = Sr * math.sqrt(2 * P0 / Ploss_rated)
    bp_2_3 = Sr * math.sqrt(6 * P0 / Ploss_rated)

    # Hysteresis
    buffer_1_2 = max(0.05 * bp_1_2, 3.0)
    buffer_2_3 = max(0.05 * bp_2_3, 3.0)

    return {
        '1_2_up': bp_1_2 + buffer_1_2,
        '1_2_down': bp_1_2 - buffer_1_2,
        '2_3_up': bp_2_3 + buffer_2_3,
        '2_3_down': bp_2_3 - buffer_2_3,
    }


def transformer_loss_model(load_mva, P0, Ploss_rated, Sr):
    """Calculate transformer loss"""
    if load_mva <= 0:
        return P0
    load_factor = load_mva / Sr
    return P0 + Ploss_rated * (load_factor ** 2)
