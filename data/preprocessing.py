"""
Data Preprocessing Module

Advanced feature engineering and normalization for power system time series data.
Generates lag features, rolling statistics, and calendar features.
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple


def preprocess_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    """
    Advanced preprocessing with comprehensive feature engineering for power system variables.

    Features generated:
    - Lagged features: [10min, 20min, 30min, 40min, 50min, 1Hour, 2Hours, 4Hours,24Hours, 168Hours] for all power variables
    - Rolling means: [1h, 4h, 24h] for all power variables
    - Rolling std: [24h] for volatility capture
    - Calendar features: hour_sin, hour_cos, minute_sin, minute_cos, day_of_week, is_weekend

    CRITICAL: Applies Z-score normalization to all power features using training set statistics
    to prevent gradient issues and feature dominance.

    Args:
        df: Raw DataFrame with power system data

    Returns:
        Tuple of (processed_df, metadata):
            - processed_df: DataFrame with engineered and normalized features
            - metadata: Dictionary with feature information and normalization stats
    """

    print("Preprocessing data with advanced feature engineering...")

    # Create basic features
    processed_df = df.copy()

    # Define power variables to engineer features for
    # Check which columns exist in the dataset
    power_variables = ['P_TOTAL', 'Q_TOTAL', 'S_TOTAL']
    available_variables = [var for var in power_variables if var in processed_df.columns]
    print(f"Available power variables: {available_variables}")

    # Lag features
    lag_steps = [1, 2, 3, 4, 5, 6, 12, 24, 144, 1008]
    lag_labels = ['10min', '20min', '30min', '40min', '50min', '1Hour', '2Hours', '4Hours','24Hours', '168Hours']
    for var in available_variables:
        for lag, label in zip(lag_steps, lag_labels):
            processed_df[f'{var}_lag_{label}'] = processed_df[var].shift(lag)

    # Rolling mean features (1h, 4h, 24h at 10-min resolution = 6, 24, 144 steps)
    rolling_windows = [6, 24, 144]
    rolling_labels = ['1h', '4h', '24h']
    for var in available_variables:
        for window, label in zip(rolling_windows, rolling_labels):
            processed_df[f'{var}_roll_{label}_mean'] = processed_df[var].rolling(window, min_periods=1).mean()

    # Rolling standard deviation for volatility (24h window to capture daily variability)
    volatility_window = 144  # 24 hours at 10-min resolution
    for var in available_variables:
        processed_df[f'{var}_roll_24h_std'] = processed_df[var].rolling(volatility_window, min_periods=1).std()

    # Calendar features for temporal patterns
    processed_df['hour_sin'] = np.sin(2 * np.pi * processed_df.index.hour / 24)
    processed_df['hour_cos'] = np.cos(2 * np.pi * processed_df.index.hour / 24)
    processed_df['day_of_week'] = processed_df.index.dayofweek / 6.0  # Normalize to [0, 1]
    processed_df['is_weekend'] = (processed_df.index.dayofweek >= 5).astype(float)

    # Add minute of hour feature for 10-minute resolution patterns
    processed_df['minute_sin'] = np.sin(2 * np.pi * processed_df.index.minute / 60)
    processed_df['minute_cos'] = np.cos(2 * np.pi * processed_df.index.minute / 60)

    # Drop NaN rows (from lag features)
    initial_length = len(processed_df)
    processed_df.dropna(inplace=True)
    print(f"Dropped {initial_length - len(processed_df)} rows with NaN values")

    # Create train/val/test splits (80/10/10)
    n = len(processed_df)
    train_end = int(0.8 * n)
    val_end = int(0.9 * n)
    processed_df['split'] = 'test'
    processed_df.iloc[:train_end, processed_df.columns.get_loc('split')] = 'train'
    processed_df.iloc[train_end:val_end, processed_df.columns.get_loc('split')] = 'val'

    # Metadata
    # Feature columns include all except target and split
    # (includes original power variables + engineered features)
    feature_cols = [col for col in processed_df.columns if col not in ['S_TOTAL', 'split']]

    # Organize features by type for interpretability
    feature_groups = {
        'original_variables': available_variables,
        'lag_features': [col for col in feature_cols if '_lag_' in col],
        'rolling_mean_features': [col for col in feature_cols if '_roll_' in col and '_mean' in col],
        'rolling_std_features': [col for col in feature_cols if '_roll_' in col and '_std' in col],
        'calendar_features': [col for col in feature_cols if col in ['hour_sin', 'hour_cos', 'day_of_week', 'is_weekend', 'minute_sin', 'minute_cos']]
    }

    # Calendar features are already properly scaled [-1,1] or [0,1], so we skip them
    print("\n" + "="*60)
    print("APPLYING FEATURE NORMALIZATION (Z-SCORE)")
    print("="*60)

    # Features to normalize (exclude calendar features which are already scaled)
    calendar_features = feature_groups['calendar_features']
    features_to_normalize = [col for col in feature_cols if col not in calendar_features]

    # Compute mean and std from TRAINING data only (critical: prevent data leakage!)
    train_data = processed_df[processed_df['split'] == 'train']
    feature_stats = {}

    print(f"\nNormalizing {len(features_to_normalize)} power features using training stats...")
    print(f"Skipping {len(calendar_features)} calendar features (already scaled)\n")
    for col in features_to_normalize:
        mean_val = train_data[col].mean()
        std_val = train_data[col].std()

        # Prevent division by zero for constant features
        if std_val == 0 or np.isnan(std_val):
            print(f"  WARNING: {col} has zero std, using std=1.0")
            std_val = 1.0

        feature_stats[col] = {'mean': mean_val, 'std': std_val}

        # Apply z-score normalization to ALL splits: z = (x - mean) / std
        processed_df[col] = (processed_df[col] - mean_val) / std_val

    # Show sample statistics
    print(f"Feature normalization examples:")

    sample_features = features_to_normalize[:min(3, len(features_to_normalize))]
    for col in sample_features:
        orig_mean = feature_stats[col]['mean']
        orig_std = feature_stats[col]['std']
        new_mean = processed_df[processed_df['split'] == 'train'][col].mean()
        new_std = processed_df[processed_df['split'] == 'train'][col].std()

        print(f"  {col}:")
        print(f"    Original: mean={orig_mean:.2f}, std={orig_std:.2f}")
        print(f"    Normalized: mean={new_mean:.4f}, std={new_std:.4f}")


    print(f"\nNormalizing target variable (S_TOTAL)...")

    target_mean = train_data['S_TOTAL'].mean()
    target_std = train_data['S_TOTAL'].std()
    if target_std == 0 or np.isnan(target_std):
        print(f"  WARNING: S_TOTAL has zero std, using std=1.0")
        target_std = 1.0

    # Apply z-score normalization to target
    processed_df['S_TOTAL'] = (processed_df['S_TOTAL'] - target_mean) / target_std

    # Verify normalization
    train_target_mean = processed_df[processed_df['split'] == 'train']['S_TOTAL'].mean()
    train_target_std = processed_df[processed_df['split'] == 'train']['S_TOTAL'].std()

    print(f"  Original: mean={target_mean:.2f} MVA, std={target_std:.2f} MVA")
    print(f"  Normalized: mean={train_target_mean:.4f}, std={train_target_std:.4f}")
    print(f"  [OK] Target normalized successfully!")

    # Store normalization statistics in metadata for inverse transform during backtesting
    normalization_stats = {
        'feature_stats': feature_stats,
        'target_mean': target_mean,
        'target_std': target_std,
        'normalized_features': features_to_normalize,
        'calendar_features': calendar_features
    }

    metadata = {
        'feature_columns': feature_cols,
        'target_column': 'S_TOTAL',
        'n_features': len(feature_cols),
        'feature_groups': feature_groups,
        'available_power_variables': available_variables,
        'normalization': normalization_stats
    }

    print(f"\n{'='*60}")
    print(f"Preprocessing complete: {len(processed_df)} samples")
    print(f"Feature breakdown:")
    print(f"  - Original power variables: {len(feature_groups['original_variables'])}")
    print(f"  - Lag features: {len(feature_groups['lag_features'])}")
    print(f"  - Rolling mean features: {len(feature_groups['rolling_mean_features'])}")
    print(f"  - Rolling std features (volatility): {len(feature_groups['rolling_std_features'])}")
    print(f"  - Calendar features: {len(feature_groups['calendar_features'])}")
    print(f"  - Total features: {len(feature_cols)}")
    print(f"  - Normalized features: {len(features_to_normalize)}")

    return processed_df, metadata

