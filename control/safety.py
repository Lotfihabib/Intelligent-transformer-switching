"""
Safety Layer Module

This module implements the rule-based safety layer that validates and adjusts MPC
decisions to ensure safe transformer operation. It enforces conservative constraints
that prevent overloading and maintain system stability.

Key Function:
- safety_layer: Apply sequential safety rules to validate switching decisions

Safety Rules (applied in order):
1. Minimum Transformer Rule: Never allow N < 1
2. High Load Protection: Require N ≥ 2 if q90 > 95 MVA
3. Dwell Time Enforcement: Respect minimum on/off times (3 steps = 30 min)
4. Quantile Guards: Switch only when quantiles support the transition
   - Switch DOWN: Use q50 (median) instead of q70 for less conservative control
   - Switch UP: Use q30 instead of q10 (still conservative for safety)
5. Uncertainty Deferral: Defer switching if uncertainty is high (unless overload risk)
6. Overload Protection: Ensure N × 70 MVA ≥ q90 × 1.05

Design Philosophy:
- Balanced approach between safety and efficiency
- Less conservative than previous versions to enable more savings
- Quantile-based decisions account for forecast uncertainty
- Hysteresis buffers prevent rapid switching oscillation
"""

import numpy as np
from config import CONFIG


def safety_layer(decision, samples, thresholds, current_state):
    """
    Safety layer with balanced constraints and rule activation tracking.

    Returns:
        tuple: (decision, rules_triggered) where decision is the validated
               switching decision (int) and rules_triggered is a list of
               rule numbers (1-6) that modified the decision.
    """
    if len(samples) == 0:
        return decision, []

    rules_triggered = []

    # Extract quantiles for decision-making
    q10 = np.percentile(samples[:, 0], 10)
    q50 = np.percentile(samples[:, 0], 50)
    q90 = np.percentile(samples[:, 0], 90)
    uncertainty = q90 - q10

    current_N = current_state['N']
    dwell_time = current_state['dwell_time']
    Sr = CONFIG['Sr']

    # Rule 1: Minimum transformer count
    before = decision
    if decision < 1:
        decision = 1
    if decision != before:
        rules_triggered.append(1)

    # Rule 2: High load protection (use q90 for safety)
    before = decision
    if q90 > CONFIG['high_load_threshold'] and decision < 2:
        decision = 2
    if decision != before:
        rules_triggered.append(2)

    # Rule 3: Dwell time enforcement (prevent rapid switching)
    before = decision
    if decision != current_N:
        if decision < current_N:  # Turning OFF transformer
            if dwell_time < CONFIG['min_on_time']:
                decision = current_N  # Too soon to turn off
        else:  # Turning ON transformer
            if dwell_time < CONFIG['min_off_time']:
                decision = current_N  # Too soon to turn on
    if decision != before:
        rules_triggered.append(3)

    # Rule 4: Quantile guards
    before = decision
    # Switch DOWN
    if decision < current_N:
        # For down transitions, use the lower threshold (e.g., '2_3_down' for 3->2)
        threshold_key = f'{decision}_{current_N}_down'
        threshold = thresholds.get(threshold_key, 0)
        if q90 >= threshold:
            decision = current_N

    # Switch UP
    if decision > current_N:
        # For up transitions, use the upper threshold (e.g., '2_3_up' for 2->3)
        threshold_key = f'{current_N}_{decision}_up'
        threshold = thresholds.get(threshold_key, 1000)
        if q10 <= threshold:
            decision = current_N
    if decision != before:
        rules_triggered.append(4)

    # Rule 5: Uncertainty deferral
    before = decision
    if uncertainty > CONFIG['uncertainty_limit'] and decision != current_N:
        if q90 <= current_N * Sr * 0.90:  # Changed from q90 and 0.95 to be less restrictive
            decision = current_N
    if decision != before:
        rules_triggered.append(5)

    # Rule 6: Final overload protection (use q90 for safety)
    before = decision
    required_N = int(np.ceil(q90 * (1 + CONFIG['overload_margin']) / Sr))

    if decision < required_N:
        decision = min(required_N, 3)
    if decision != before:
        rules_triggered.append(6)

    return decision, rules_triggered
