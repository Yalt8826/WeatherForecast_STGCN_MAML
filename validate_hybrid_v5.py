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
        f"./Out_Data/AdaptedModels/hybrid_v5_adapted_{region_name}_{region_coords}.pt"
    )
    base_path = "./Out_Data/SavedModels/hybrid_maml_model_v5_best.pt"

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
            "hidden_channels": 256,
            "output_channels": 12,
            "window_size": 24,
            "forecast_horizon": 8,
        }
        hybrid_config = {
            "lstm_hidden_size": 128,
            "lstm_num_layers": 4,
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
    
    # Extract saved normalization stats
    saved_stats = checkpoint.get("stats", None)
    if saved_stats is None:
        print("⚠️ No saved stats found, will compute new ones")
    else:
        print("✅ Using saved normalization stats from adaptation")

    return hybrid_model, koppen_embed, config, saved_stats


def validateAdapted(region_coords, region_name):
    """Validate adapted model on 2025 data

    Args:
        region_coords: Tuple of (lat_min, lat_max, lon_min, lon_max)
        region_name: String name for the region

    Returns:
        dict: Validation results
    """
    print("=" * 80)
    print(f"🎯 MODEL V5 VALIDATION: {region_name}")
    print("=" * 80)
    print(f"Region: {region_coords}")
    print(f"Device: {DEVICE}")
    print(f"Using 2025 data for validation")
    print("=" * 80)

    # Load model and saved stats
    hybrid_model, koppen_embed, config, saved_stats = load_model(region_coords, region_name)

    lat_min, lat_max, lon_min, lon_max = region_coords

    # Load 2025 validation data
    data_dir = r"E:/Study/5th Sem Mini Project/Datasets/2025/Jan2Mar"
    accum_file = os.path.join(data_dir, "data_stream-oper_stepType-accum.nc")
    instant_file = os.path.join(data_dir, "data_stream-oper_stepType-instant.nc")

    ds_accum = xr.open_dataset(accum_file)
    ds_instant = xr.open_dataset(instant_file)
    ds = xr.merge([ds_accum, ds_instant], compat="override")

    # Extract region - ensure we have valid data
    ds_reg = ds.sel(latitude=slice(lat_max, lat_min), longitude=slice(lon_min, lon_max))
    
    # Check available time steps and use a reasonable subset
    total_time_steps = len(ds_reg.valid_time)
    print(f"   Total available time steps: {total_time_steps}")
    
    if total_time_steps < config["window_size"] + config["forecast_horizon"]:
        print(f"   ❌ Not enough time steps for validation (need {config['window_size'] + config['forecast_horizon']})")
        return {"average_mse": float('inf')}
    
    # Use middle portion of data for validation (more stable)
    start_idx = max(0, total_time_steps // 4)
    end_idx = min(total_time_steps, start_idx + 50)  # Use up to 50 samples
    ds_reg = ds_reg.isel(valid_time=slice(start_idx, end_idx))

    if "day_of_year_sin" not in ds_reg:
        ds_reg = add_time_embeddings(ds_reg)

    # Prepare data with saved stats
    edge_index, num_nodes, _ = build_spatial_graph(ds_reg, k_neighbors=4)
    
    if saved_stats is not None:
        # Use saved normalization stats from adaptation
        features, _ = prepare_model_input(ds_reg, 0, koppen_embed, normalize=True, stats=saved_stats)
        stats = saved_stats
        print("✅ Applied saved normalization stats")
    else:
        # Fallback to computing new stats
        features, stats = prepare_model_input(ds_reg, 0, koppen_embed, normalize=True)
        print("⚠️ Computing new normalization stats")
    dataset = WeatherGraphDataset(
        features,
        edge_index,
        window_size=config["window_size"],
        forecast_horizon=config["forecast_horizon"],
    )

    print(f"   Dataset size: {len(dataset)} samples")
    print(f"   Grid nodes: {num_nodes}")
    
    if len(dataset) == 0:
        print("   ❌ No valid samples in dataset")
        return {"average_mse": float('inf')}

    # Get predictions - batch process for better performance
    all_predictions = []
    all_targets = []
    
    num_samples = min(3, len(dataset))  # Use fewer samples for speed
    print(f"   Processing {num_samples} samples for validation")
    
    with torch.no_grad():
        for i in range(num_samples):
            window = dataset[i]
            y_pred = hybrid_model(window.x.to(DEVICE), window.edge_index.to(DEVICE)).cpu().numpy()
            all_predictions.append(y_pred)
            all_targets.append(window.y.cpu().numpy())
    
    # Average predictions for stability
    y_pred = np.mean(all_predictions, axis=0)
    y_true = np.mean(all_targets, axis=0)
    
    # Use first sample for input analysis
    x_cpu = dataset[0].x.cpu().numpy()

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
    
    # Set fixed y-axis range for consistent temperature scaling (1K per unit)
    all_temp_data = np.concatenate([all_temps_input[~np.isnan(all_temps_input)], 
                                   all_temps_true[~np.isnan(all_temps_true)], 
                                   all_temps_pred[~np.isnan(all_temps_pred)]])
    temp_min = np.floor(np.min(all_temp_data))
    temp_max = np.ceil(np.max(all_temp_data))
    plt.ylim(temp_min - 2, temp_max + 2)  # Add 2K buffer
    plt.yticks(np.arange(temp_min - 2, temp_max + 3, 1))  # 1K intervals
    
    plt.xlabel("Time")
    plt.ylabel("Temperature (K)")
    plt.title(f"2025 Temperature Analysis - {region_name}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    os.makedirs("./Out_Data/V5 Validation", exist_ok=True)
    plt.savefig(
        f"./Out_Data/V5 Validation/{region_name}_temperature.png", dpi=150, bbox_inches="tight"
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
        f"./Out_Data/V5 Validation/{region_name}_all_variables.png", dpi=150, bbox_inches="tight"
    )
    plt.close()

    # Calculate metrics (exclude surface pressure from average)
    total_mse = 0
    results = {}
    mse_count = 0

    for v_idx, var_name in enumerate(VAR_NAMES[:6]):
        if v_idx < y_true_avg.shape[1]:
            true_vals = y_true_avg[:, v_idx] * std[v_idx] + mean[v_idx]
            pred_vals = y_pred_avg[:, v_idx] * std[v_idx] + mean[v_idx]

            mse = np.mean((pred_vals - true_vals) ** 2)
            mae = np.mean(np.abs(pred_vals - true_vals))

            results[var_name] = {"mse": mse, "mae": mae}
            
            # Exclude surface pressure from average (it has huge values)
            if var_name != "sp":
                total_mse += mse
                mse_count += 1

    avg_mse = total_mse / mse_count if mse_count > 0 else 0
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
    region_name = "NewYork2025"

    results = validateAdapted(region_coords, region_name)
    print(f"\n✅ Validation results: Average MSE = {results['average_mse']:.3f}")


if __name__ == "__main__":
    main()