"""
Feature extraction and preprocessing for weather data.
"""

import torch
import numpy as np
from embed_utils import add_time_embeddings, KoppenEmbedding
import xarray as xr


def diagnose_nan_percentage(ds):
    """Print NaN percentage for each variable."""
    print("\n" + "=" * 60)
    print("NaN PERCENTAGE BY VARIABLE:")
    print("=" * 60)
    all_vars = [
        "u10",
        "v10",
        "d2m",
        "t2m",
        "sp",
        "tp",
        "u100",
        "v100",
        "str",
        "hcc",
        "lcc",
        "mcc",
        "e",
    ]

    for var in all_vars:
        if var in ds:
            data = ds[var].values
            nan_pct = (np.isnan(data).sum() / data.size) * 100
            status = "✅" if nan_pct < 5 else "⚠️" if nan_pct < 15 else "❌"
            print(f"{status} {var:6s}: {nan_pct:5.1f}% NaN")
    print("=" * 60 + "\n")


# ✅ CLEANED: Only variables with low NaN percentage
WEATHER_VARS = [
    "u10",
    "v10",
    "t2m",
    "d2m",
    "sp",
    "tp",
    "u100",
    "v100",
    "str",
    "hcc",
    "lcc",
    "e",
]
# Total: 12 weather variables
# Input features: 12 (weather) + 4 (time) + 8 (Köppen) = 24 total

TIME_VARS = [
    "year_progress_sin",
    "year_progress_cos",
    "day_progress_sin",
    "day_progress_cos",
]


def prepare_model_input(ds, koppen_code, koppen_embed_layer, normalize=True):
    """
    Extract and prepare all features for ST-GCN input.
    HANDLES NaN VALUES in weather data.

    Args:
        ds: xarray Dataset (from dataLoader.main_dataloader)
        koppen_code: Integer Köppen code
        koppen_embed_layer: KoppenEmbedding instance
        normalize: Whether to normalize weather features

    Returns:
        features: Tensor [time, nodes, total_features]
        stats: Normalization statistics dict
    """

    # Extract weather data
    weather_data = np.stack([ds[var].values for var in WEATHER_VARS], axis=-1)
    # Shape: [time, lat, lon, ... (12 variables)]

    # ✅ DIAGNOSTIC 1: PRINT VARIABLE ORDER
    print("\n" + "=" * 60)
    print("VARIABLE ORDER IN FEATURES:")
    print("=" * 60)
    for i, var in enumerate(WEATHER_VARS):
        print(f"Index {i}: {var}")
    print("=" * 60 + "\n")

    # ✅ CHECK FOR NaN
    nan_count = np.isnan(weather_data).sum()
    if nan_count > 0:
        print(
            f"      ⚠️  Found {nan_count} NaN values in weather data. Filling with means..."
        )

        # Fill NaN with column-wise mean (per variable)
        for i in range(weather_data.shape[-1]):
            var_data = weather_data[..., i]
            var_mean = np.nanmean(var_data)  # Mean ignoring NaN
            if np.isnan(var_mean):
                var_mean = 0.0  # If all NaN, use 0
            weather_data[..., i] = np.nan_to_num(var_data, nan=var_mean)

        print(f"      ✅ NaN values filled")

    # Extract time data
    time_data = np.stack([ds[var].values for var in TIME_VARS], axis=-1)
    # Shape: [time, 4]

    # Get dimensions
    num_time, num_lat, num_lon, num_weather = weather_data.shape
    num_nodes = num_lat * num_lon

    # Reshape weather: [time, lat, lon, 12] -> [time, nodes, 12]
    weather_features = weather_data.reshape(num_time, num_nodes, num_weather)

    # Normalize weather features
    stats = {}
    if normalize:
        mean = weather_features.mean(axis=(0, 1))
        std = (
            weather_features.std(axis=(0, 1)) + 1e-8
        )  # Add epsilon to avoid div by zero

        # Check for problematic values
        if np.any(np.isnan(mean)) or np.any(np.isnan(std)):
            print(f"      ⚠️  NaN in statistics! Using safe defaults.")
            mean = np.nan_to_num(mean, nan=0.0)
            std = np.nan_to_num(std, nan=1.0)

        weather_features = (weather_features - mean) / std
        stats = {"mean": mean, "std": std}

        # ✅ DIAGNOSTIC 2: PRINT NORMALIZATION STATS
        print("\n" + "=" * 60)
        print("NORMALIZATION STATS:")
        print("=" * 60)
        for i, var in enumerate(WEATHER_VARS):
            print(f"{var:6s}: mean={mean[i]:8.2f}, std={std[i]:8.2f}")
            if var == "t2m":
                print(f"  👆 t2m normalized: (X - {mean[i]:.2f}) / {std[i]:.2f}")
        print("=" * 60 + "\n")

    # Convert to tensors
    weather_tensor = torch.tensor(weather_features, dtype=torch.float32)

    # Expand time features to all nodes
    time_expanded = np.tile(time_data[:, np.newaxis, :], (1, num_nodes, 1))
    time_tensor = torch.tensor(time_expanded, dtype=torch.float32)

    # Get Köppen embeddings
    device = next(koppen_embed_layer.parameters()).device
    koppen_tensor = torch.tensor([koppen_code], dtype=torch.long).to(device)
    koppen_embed = koppen_embed_layer(koppen_tensor)
    koppen_embed = koppen_embed.cpu()
    koppen_expanded = koppen_embed.unsqueeze(0).expand(num_time, num_nodes, -1)

    # Concatenate
    combined = torch.cat([weather_tensor, time_tensor, koppen_expanded], dim=-1)

    # Final NaN check
    if torch.isnan(combined).any():
        print(f"      ⚠️  WARNING: Still have NaN after processing!")
        combined = torch.nan_to_num(combined, nan=0.0)

    return combined, stats


def denormalize_predictions(
    predictions, stats, target_var_idx=2
):  # ✅ CHANGED from 3 to 2!
    """
    Denormalize predictions back to original scale.

    Args:
        predictions: Normalized predictions (tensor or numpy array)
        stats: Dict with 'mean' and 'std' from prepare_model_input
        target_var_idx: Index of target variable (default 2 = t2m in current WEATHER_VARS)

    Returns:
        Denormalized predictions in original units
    """
    if "mean" in stats and "std" in stats:
        mean = stats["mean"][target_var_idx]  # Get mean for target variable
        std = stats["std"][target_var_idx]  # Get std for target variable

        # Convert to same type as predictions
        if isinstance(predictions, torch.Tensor):
            mean = torch.tensor(
                mean, dtype=predictions.dtype, device=predictions.device
            )
            std = torch.tensor(std, dtype=predictions.dtype, device=predictions.device)

        return predictions * std + mean

    return predictions


def denormalize_all_predictions(predictions, stats):
    """
    Denormalize ALL 12 weather variables.

    Args:
        predictions: [samples, 12] or [12] numpy array
        stats: dict with 'mean' and 'std' arrays of shape [12]

    Returns:
        Denormalized predictions in same shape
    """
    mean = stats["mean"]
    std = stats["std"]

    # predictions shape: [samples, 12] or [12]
    if predictions.ndim == 1:
        # Single sample
        denorm = predictions * std + mean
    else:
        # Multiple samples
        denorm = predictions * std[np.newaxis, :] + mean[np.newaxis, :]

    return denorm
