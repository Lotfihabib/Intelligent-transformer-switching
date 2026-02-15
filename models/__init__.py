"""
Models Module

Neural network architectures, loss functions, and training procedures for
probabilistic load forecasting.
"""

try:
    import torch
    TORCH_AVAILABLE = True

    from .dataset import SimpleTimeSeriesDataset
    from .networks import TemporalFusionTransformer
    from .loss import pinball_loss
    from .training import (
        train_model,
        train_model_from_dataframe,
        create_datasets_and_loaders,
        create_tft_model,
        save_model_and_history
    )

    __all__ = [
        'SimpleTimeSeriesDataset',
        'TemporalFusionTransformer',
        'pinball_loss',
        'train_model',
        'train_model_from_dataframe',
        'create_datasets_and_loaders',
        'create_tft_model',
        'save_model_and_history',
        'TORCH_AVAILABLE',
    ]

except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: PyTorch not available. Model training will not be available.")

    __all__ = ['TORCH_AVAILABLE']
