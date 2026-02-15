"""
Dataset Module

PyTorch Dataset class for time series forecasting with historical context.
"""

import torch
from torch.utils.data import Dataset


class SimpleTimeSeriesDataset(Dataset):
    """
    Dataset for time series forecasting.

    Creates sequences with historical features and targets for multi-horizon forecasting.

    Args:
        df: Preprocessed DataFrame with features and target
        target_col: Name of target column
        seq_len: Length of historical sequence
        horizon: Forecast horizon length
        split: Data split to use ('train', 'val', or 'test')
    """

    def __init__(self, df, target_col, seq_len, horizon, split):
        self.data = df[df['split'] == split].copy()
        self.target_col = target_col
        self.seq_len = seq_len
        self.horizon = horizon
        self.feature_cols = [col for col in df.columns if col not in [target_col, 'split']]

    def __len__(self):
        return max(0, len(self.data) - self.seq_len - self.horizon + 1)

    def __getitem__(self, idx):
        # Features
        features = self.data.iloc[idx:idx+self.seq_len][self.feature_cols].values

        # Historical target
        hist_target = self.data.iloc[idx:idx+self.seq_len][self.target_col].values

        # Future target
        future_target = self.data.iloc[idx+self.seq_len:idx+self.seq_len+self.horizon][self.target_col].values

        return (torch.FloatTensor(features),
                torch.FloatTensor(hist_target),
                torch.FloatTensor(future_target))
