"""
Model Predictive Control (MPC) Module

This module implements the stochastic MPC optimization algorithm for transformer
switching decisions. It evaluates switching options over a receding horizon and
selects the action that minimizes expected costs.

Key Function:
- stochastic_mpc: Optimize switching decision using sampled load trajectories

Optimization Objective:
- Minimize: switching_cost + expected_operational_cost
- Switching cost: kappa penalty for changing state
- Operational cost: Expected energy losses over horizon T
- Overload penalty: Large penalty for capacity violations

Constraints:
- Dwell time: Minimum on/off time before switching allowed
- Transformer count: N ∈ {1, 2, 3}
- Capacity: Load must not exceed N × Sr (enforced via penalty)
"""

import numpy as np
from config import CONFIG
from .power_model import transformer_loss_model


def stochastic_mpc(samples, current_state):
    """Stochastic MPC"""
    M, horizon = samples.shape
    T = min(CONFIG['T'], horizon)
    Sr = CONFIG['Sr']
    kappa = CONFIG['kappa']

    # Current state
    current_N = current_state['N']

    # Evaluate options
    best_cost = float('inf')
    best_N = current_N

    for N in [1, 2, 3]:
        # Check feasibility
        if current_state['dwell_time'] < 24 and N != current_N:
            continue

        # Switching cost
        switching_cost = kappa if N != current_N else 0

        # Expected operational cost
        operational_cost = 0
        for t in range(T):
            step_cost = 0
            for m in range(M):  # Sample subset for speed
                load = samples[m, t]
                if load > N * Sr:
                    step_cost += 1000  # Overload penalty
                else:
                    load_per_transformer = load / N
                    loss = transformer_loss_model(load_per_transformer, CONFIG['P0'], CONFIG['Ploss_rated'], Sr)
                    step_cost += N * loss
            operational_cost += step_cost / M

        total_cost = switching_cost + operational_cost

        if total_cost < best_cost:
            best_cost = total_cost
            best_N = N
    return best_N
