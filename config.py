"""
Configuration Module

Central configuration for the Grid Transformer Switching Optimization System.
All system parameters are defined here for easy tuning.

Last Updated: 2024-10-28
"""

# =============================================================================
# SYSTEM CONFIGURATION
# =============================================================================

CONFIG = {
    # =========================================================================
    # Power System Parameters
    # =========================================================================
    'P0': 38.0,                    # No-load loss per transformer (kW)
    'Ploss_rated': 250.0,          # Rated load loss per transformer (kW)
    'Sr': 70.0,                    # Rated power per transformer (MVA)

    # =========================================================================
    # Control Parameters
    # =========================================================================
    'T': 24,                       # MPC horizon (4 hours at 10-min resolution)
    'H': 24,                       # Forecast horizon (4 hours)
    'M': 100,                      # Number of trajectory samples
    'kappa': 50.0,                 # Switching penalty (kW)

    # =========================================================================
    # Safety Constraints
    # =========================================================================
    'min_on_time': 24,                  # Minimum on time (4 hours = 24 steps)
    'min_off_time': 24,                 # Minimum off time (4 hours = 24 steps)
    'uncertainty_limit': 8.0,           # Uncertainty threshold for deferral (MVA)
    'high_load_threshold': 95.0,        # High load protection threshold (MVA)
    'overload_margin': 0.05,            # Overload protection margin (5%)
    'forecast_bias_correction': 1.32,   # MVA bias correction

    # =========================================================================
    # Model Parameters
    # =========================================================================
    'sequence_length': 144,                 # Input sequence length (24 hours at 10-min)
    'quantiles': [0.1, 0.5, 0.9],           # Forecast quantiles
    'quantile_weights': [1.0, 1.0, 1.0],    # Quantile loss weights (weighted median)
    'batch_size': 64,                       # Training batch size
    'learning_rate': 0.0001,                # Optimized learning rate for stability
    'patience': 10,                         # Early stopping patience (epochs)
    'weight_decay': 0.0001,                 # L2 regularization strength
    'dropout_rate': 0.10,                   # Dropout rate for regularization

    # =========================================================================
    # Evaluation Parameters
    # =========================================================================
    'compute_multistep_metrics': False,      # Track multistep metrics during training (slower)
    'save_full_horizon_predictions': False,  # Save all 24 horizons in backtesting logs (large files)
    'evaluation_horizons': [1, 6, 12, 24],   # Key horizons for stratified analysis
    'load_regime_breakpoints': [38.6, 66.9], # MVA breakpoints for low/medium/high load classification

    # Advanced Calibration Parameters (Phase 3)
    'advanced_calibration_analysis': False,  # Run detailed per-horizon calibration analysis
    'calibration_horizons': [1, 6, 12, 24],  # Horizons for detailed calibration analysis
    'pit_num_bins': 10,                      # Number of bins for PIT histograms
    'calibration_confidence_level': 0.80,    # Confidence level for interval scores (80%)
}


def get_config():
    """
    Get a copy of the configuration dictionary.

    Returns:
        dict: Copy of CONFIG dictionary
    """
    return CONFIG.copy()


def validate_config():
    """
    Validate configuration parameters.

    Raises:
        ValueError: If configuration parameters are invalid
    """
    # Power system validation
    assert CONFIG['P0'] > 0, "P0 must be positive"
    assert CONFIG['Ploss_rated'] > 0, "Ploss_rated must be positive"
    assert CONFIG['Sr'] > 0, "Sr must be positive"

    # Control parameters validation
    assert CONFIG['T'] > 0, "T must be positive"
    assert CONFIG['H'] > 0, "H must be positive"
    assert CONFIG['M'] > 0, "M must be positive"
    assert CONFIG['kappa'] >= 0, "kappa must be non-negative"

    # Safety constraints validation
    assert CONFIG['min_on_time'] > 0, "min_on_time must be positive"
    assert CONFIG['min_off_time'] > 0, "min_off_time must be positive"
    assert CONFIG['uncertainty_limit'] > 0, "uncertainty_limit must be positive"
    assert 0 < CONFIG['overload_margin'] < 1, "overload_margin must be in (0, 1)"

    # Model parameters validation
    assert CONFIG['sequence_length'] > 0, "sequence_length must be positive"
    assert len(CONFIG['quantiles']) == 3, "Must have 3 quantiles"
    assert len(CONFIG['quantile_weights']) == 3, "Must have 3 quantile weights"
    assert CONFIG['batch_size'] > 0, "batch_size must be positive"
    assert CONFIG['learning_rate'] > 0, "learning_rate must be positive"
    assert 0 < CONFIG['dropout_rate'] < 1, "dropout_rate must be in (0, 1)"

    # Evaluation parameters validation
    if 'evaluation_horizons' in CONFIG:
        assert all(h > 0 for h in CONFIG['evaluation_horizons']), \
            "evaluation_horizons must all be positive"
        assert all(h <= CONFIG['H'] for h in CONFIG['evaluation_horizons']), \
            f"evaluation_horizons must be within [1, {CONFIG['H']}]"
    if 'load_regime_breakpoints' in CONFIG:
        assert len(CONFIG['load_regime_breakpoints']) == 2, \
            "Must have exactly 2 breakpoints (low/medium/high)"
        assert CONFIG['load_regime_breakpoints'][0] < CONFIG['load_regime_breakpoints'][1], \
            "Breakpoints must be in ascending order"

    # Advanced calibration parameters validation
    if 'calibration_horizons' in CONFIG:
        assert all(h > 0 for h in CONFIG['calibration_horizons']), \
            "calibration_horizons must all be positive"
        assert all(h <= CONFIG['H'] for h in CONFIG['calibration_horizons']), \
            f"calibration_horizons must be within [1, {CONFIG['H']}]"
    if 'pit_num_bins' in CONFIG:
        assert CONFIG['pit_num_bins'] > 0, "pit_num_bins must be positive"
    if 'calibration_confidence_level' in CONFIG:
        assert 0 < CONFIG['calibration_confidence_level'] < 1, \
            "calibration_confidence_level must be in (0, 1)"

    return True


# Validate on import
validate_config()
