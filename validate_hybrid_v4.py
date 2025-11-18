import torch
import xarray as xr
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
from model import STGCN
from hybrid_model import HybridSTGCN_LSTM
from graphBuilder import build_spatial_graph
from featurePreprocessor import prepare_model_input
from dataset import WeatherGraphDataset
from embed_utils import add_time_embeddings, KoppenEmbedding

# Configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

VAR_NAMES = [
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


def load_model(region_coords, region_name):
    """Load the adapted hybrid model"""
    # Try adapted model first
    adapted_path = (
        f"./Out_Data/AdaptedModels/hybrid_v4_adapted_{region_name}_{region_coords}.pt"
    )
    base_path = "./Out_Data/SavedModels/hybrid_maml_model_v4.pt"

    model_path = adapted_path if os.path.exists(adapted_path) else base_path

    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        print("Please train the model first or run adaptation")
        sys.exit(1)

    print(f"Loading model: {model_path}")
    checkpoint = torch.load(model_path, map_location=DEVICE, weights_only=False)

    # Get config from checkpoint or use defaults
    if "config" in checkpoint:
        config = checkpoint["config"]
        hybrid_config = checkpoint["hybrid_config"]
    else:
        config = {
            "input_channels": 24,
            "hidden_channels": 128,
            "output_channels": 12,
            "window_size": 24,
            "forecast_horizon": 8,
        }
        hybrid_config = {
            "lstm_hidden_size": 64,
            "lstm_num_layers": 2,
            "lstm_dropout": 0.2,
        }

    # Create models
    base_stgcn = STGCN(
        in_channels=config["input_channels"],
        hidden_channels=config["hidden_channels"],
        out_channels=config["output_channels"],
        window_size=config["window_size"],
        forecast_horizon=config["forecast_horizon"],
        dropout_rate=0.2,
    ).to(DEVICE)

    hybrid_model = HybridSTGCN_LSTM(
        base_stgcn=base_stgcn,
        lstm_hidden_size=hybrid_config["lstm_hidden_size"],
        lstm_num_layers=hybrid_config["lstm_num_layers"],
        lstm_dropout=hybrid_config["lstm_dropout"],
        out_channels=config["output_channels"],
        forecast_horizon=config["forecast_horizon"],
        freeze_base=False,
    ).to(DEVICE)

    koppen_embed = KoppenEmbedding(embedding_dim=8).to(DEVICE)

    # Load weights
    hybrid_model.load_state_dict(checkpoint["hybrid_model_state_dict"])
    koppen_embed.load_state_dict(checkpoint["koppen_embed_state_dict"])

    hybrid_model.eval()
    koppen_embed.eval()

    model_type = "Adapted" if "adapted" in model_path else "Base"
    print(f"✅ {model_type} model loaded successfully")

    return hybrid_model, koppen_embed, config


def validateAdapted(region_coords, region_name):
    """Validate adapted model on 2025 data

    Args:
        region_coords: Tuple of (lat_min, lat_max, lon_min, lon_max)
        region_name: String name for the region

    Returns:
        dict: Validation results
    """
    print("=" * 80)
    print(f"🎯 MODEL V4 VALIDATION: {region_name}")
    print("=" * 80)
    print(f"Region: {region_coords}")
    print(f"Device: {DEVICE}")
    print(f"Using 2025 data for validation")
    print("=" * 80)

    # Load model
    hybrid_model, koppen_embed, config = load_model(region_coords, region_name)

    lat_min, lat_max, lon_min, lon_max = region_coords

    # Load 2025 validation data
    data_dir = r"E:/Study/5th Sem Mini Project/Datasets/2025/Jan2Mar"
    accum_file = os.path.join(data_dir, "data_stream-oper_stepType-accum.nc")
    instant_file = os.path.join(data_dir, "data_stream-oper_stepType-instant.nc")

    ds_accum = xr.open_dataset(accum_file)
    ds_instant = xr.open_dataset(instant_file)
    ds = xr.merge([ds_accum, ds_instant])

    # Extract region
    ds_reg = ds.sel(latitude=slice(lat_max, lat_min), longitude=slice(lon_min, lon_max))
    ds_reg = ds_reg.isel(valid_time=slice(0, 35))

    if "day_of_year_sin" not in ds_reg:
        ds_reg = add_time_embeddings(ds_reg)

    # Prepare data
    edge_index, num_nodes, _ = build_spatial_graph(ds_reg, k_neighbors=4)
    features, stats = prepare_model_input(ds_reg, 0, koppen_embed, normalize=True)
    dataset = WeatherGraphDataset(
        features,
        edge_index,
        window_size=config["window_size"],
        forecast_horizon=config["forecast_horizon"],
    )

    print(f"   Dataset size: {len(dataset)} samples")
    print(f"   Grid nodes: {num_nodes}")

    # Get predictions
    window = dataset[0]
    x_cpu = window.x.cpu().numpy()
    y_true = window.y.cpu().numpy()

    with torch.no_grad():
        y_pred = (
            hybrid_model(window.x.to(DEVICE), window.edge_index.to(DEVICE))
            .cpu()
            .numpy()
        )

    # Process statistics
    if isinstance(stats, dict):
        mean = np.array(stats["mean"])
        std = np.array(stats["std"])
    else:
        mean, std = stats

    # Reshape for analysis
    x_reshaped = x_cpu.reshape(config["window_size"], num_nodes, -1)
    x_avg = x_reshaped.mean(axis=1)

    y_true_reshaped = y_true.reshape(config["forecast_horizon"], num_nodes, 12)
    y_true_avg = y_true_reshaped.mean(axis=1)

    y_pred_reshaped = y_pred.reshape(config["forecast_horizon"], num_nodes, 12)
    y_pred_avg = y_pred_reshaped.mean(axis=1)

    # Get timestamps
    input_times = ds_reg["valid_time"].values[: config["window_size"]]
    forecast_times = ds_reg["valid_time"].values[
        config["window_size"] : config["window_size"] + config["forecast_horizon"]
    ]

    # Temperature analysis
    temp_input = x_avg[:, 2] * std[2] + mean[2]
    temp_true = y_true_avg[:, 2] * std[2] + mean[2]
    temp_pred = y_pred_avg[:, 2] * std[2] + mean[2]

    print(f"\n🌡️ 2025 TEMPERATURE FORECAST ({region_name}):")
    print("-" * 70)
    print("Step | Timestamp           | TrueK | PredK | ErrorK")
    print("-" * 70)
    for i, (true_t, pred_t, ts) in enumerate(zip(temp_true, temp_pred, forecast_times)):
        error = abs(pred_t - true_t)
        print(
            f"{i+1:>4} | {str(ts)[:19]} | {true_t:5.1f} | {pred_t:5.1f} | {error:6.1f}"
        )

    # Create temperature graph with timestamps
    plt.figure(figsize=(14, 6))

    # Combine all timestamps
    all_times = np.concatenate([input_times, forecast_times])
    all_temps_input = np.concatenate([temp_input, np.full(len(temp_true), np.nan)])
    all_temps_true = np.concatenate([np.full(len(temp_input), np.nan), temp_true])
    all_temps_pred = np.concatenate([np.full(len(temp_input), np.nan), temp_pred])

    # Plot with actual timestamps
    plt.plot(
        all_times,
        all_temps_input,
        "b-",
        label="Input Temperature",
        linewidth=2,
        alpha=0.7,
    )
    plt.plot(
        all_times, all_temps_true, "g-", label="True Forecast", linewidth=2, marker="o"
    )
    plt.plot(
        all_times,
        all_temps_pred,
        "r--",
        label="Predicted Forecast",
        linewidth=2,
        marker="s",
    )

    plt.axvline(
        x=forecast_times[0],
        color="black",
        linestyle=":",
        alpha=0.5,
        label="Forecast Start",
    )
    plt.xlabel("Time")
    plt.ylabel("Temperature (K)")
    plt.title(f"2025 Temperature Analysis - {region_name}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    os.makedirs("./Out_Data", exist_ok=True)
    plt.savefig(
        f"./Out_Data/{region_name}_temperature.png", dpi=150, bbox_inches="tight"
    )
    plt.close()

    # Create all variables graph
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    # Define forecast steps for x-axis
    forecast_steps = range(1, len(temp_true) + 1)

    for v_idx, var_name in enumerate(VAR_NAMES[:6]):
        if v_idx < y_true_avg.shape[1]:
            true_vals = y_true_avg[:, v_idx] * std[v_idx] + mean[v_idx]
            pred_vals = y_pred_avg[:, v_idx] * std[v_idx] + mean[v_idx]

            axes[v_idx].plot(forecast_steps, true_vals, "g-", label="True", marker="o")
            axes[v_idx].plot(
                forecast_steps, pred_vals, "r--", label="Predicted", marker="s"
            )
            axes[v_idx].set_title(f"{var_name}")
            axes[v_idx].set_xlabel("Forecast Step")
            axes[v_idx].legend()
            axes[v_idx].grid(True, alpha=0.3)

    plt.suptitle(f"2025 All Variables Forecast - {region_name}")
    plt.tight_layout()
    plt.savefig(
        f"./Out_Data/{region_name}_all_variables.png", dpi=150, bbox_inches="tight"
    )
    plt.close()

    # Calculate metrics
    total_mse = 0
    results = {}

    for v_idx, var_name in enumerate(VAR_NAMES[:6]):
        if v_idx < y_true_avg.shape[1]:
            true_vals = y_true_avg[:, v_idx] * std[v_idx] + mean[v_idx]
            pred_vals = y_pred_avg[:, v_idx] * std[v_idx] + mean[v_idx]

            mse = np.mean((pred_vals - true_vals) ** 2)
            mae = np.mean(np.abs(pred_vals - true_vals))

            results[var_name] = {"mse": mse, "mae": mae}
            total_mse += mse

    avg_mse = total_mse / 6
    results["average_mse"] = avg_mse

    print(f"\n📈 2025 PERFORMANCE SUMMARY FOR {region_name}")
    print("-" * 50)
    for var_name in VAR_NAMES[:6]:
        if var_name in results:
            mse = results[var_name]["mse"]
            mae = results[var_name]["mae"]
            print(f"{var_name:>8}: MSE={mse:8.3f}, MAE={mae:6.3f}")

    print(f"\nAverage MSE: {avg_mse:.3f}")
    print(f"\n✅ 2025 Validation complete for {region_name}")

    return results


def main():
    """Example usage"""
    # Example: Validate adapted model for New York
    region_coords = (40, 45, 285, 290)
    region_name = "NewYork2024"

    results = validateAdapted(region_coords, region_name)
    print(f"\n✅ Validation results: Average MSE = {results['average_mse']:.3f}")


if __name__ == "__main__":
    main()
