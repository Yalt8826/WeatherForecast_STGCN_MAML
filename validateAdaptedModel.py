import torch
import xarray as xr
import numpy as np
import os
import csv

from model import STGCN
from graphBuilder import build_spatial_graph
from featurePreprocessor import prepare_model_input
from dataset import WeatherGraphDataset
from embed_utils import add_time_embeddings, KoppenEmbedding

MODEL_PATH = "./Out_Data/AdaptedModels/adapted_model_2023_2024_(-40, -35, 140, 145).pt"
DATA_DIR = r"E:\Study\5th Sem Mini Project\Datasets\2025\Jan2Mar"
ACCUM_FILE = os.path.join(DATA_DIR, "data_stream-oper_stepType-accum.nc")
INSTANT_FILE = os.path.join(DATA_DIR, "data_stream-oper_stepType-instant.nc")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

lat_min, lat_max = -40, -35
lon_min, lon_max = 140, 145


def wrap_lon(lon):
    return lon if lon >= 0 else lon + 360


wrapped_lon_min = wrap_lon(lon_min)
wrapped_lon_max = wrap_lon(lon_max)

WINDOW_SIZE = 24
FORECAST_HORIZON = 8
TIME_WINDOW = WINDOW_SIZE + FORECAST_HORIZON + 1  # Need one extra for valid sample

print("Loading model checkpoint:", MODEL_PATH)
checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
config = checkpoint["config"]

model = STGCN(
    in_channels=config["input_channels"],
    hidden_channels=config["hidden_channels"],
    out_channels=config["output_channels"],
    window_size=config["window_size"],
    forecast_horizon=config["forecast_horizon"],
    dropout_rate=0.3,
).to(DEVICE)
koppen_embed = KoppenEmbedding(embedding_dim=8).to(DEVICE)
model.load_state_dict(checkpoint["model_state_dict"])
koppen_embed.load_state_dict(checkpoint["koppen_embed_state_dict"])
model.eval()
koppen_embed.eval()
print("Model loaded and set to eval.")
print(f"Testing 2023+2024-adapted model on 2025 data (multi-year adaptation)...")

ds_accum = xr.open_dataset(ACCUM_FILE)
ds_instant = xr.open_dataset(INSTANT_FILE)
ds = xr.merge([ds_accum, ds_instant])
lat_name = "latitude"
lon_name = "longitude"
time_name = "valid_time"

ds_reg = ds.sel(
    **{
        lat_name: slice(lat_max, lat_min),
        lon_name: slice(wrapped_lon_min, wrapped_lon_max),
    }
)
if ds_reg[lat_name].size == 0 or ds_reg[lon_name].size == 0:
    print("ERROR: region crop returns no gridpoints! Check coverage and fix bounds.")
    exit(1)
if time_name in ds_reg.coords:
    ds_reg = ds_reg.isel({time_name: slice(0, TIME_WINDOW)})
    print(
        f"Using first {TIME_WINDOW} timesteps from 2025 data (need {WINDOW_SIZE} input + {FORECAST_HORIZON} forecast + 1 buffer)"
    )
else:
    print("valid_time not found!")
    exit(1)
if "day_of_year_sin" not in ds_reg:
    ds_reg = add_time_embeddings(ds_reg)

edge_index, num_nodes, _ = build_spatial_graph(ds_reg, k_neighbors=4)
koppen_code = checkpoint.get("koppen_code", 0) if "koppen_code" in checkpoint else 0
features, stats = prepare_model_input(ds_reg, koppen_code, koppen_embed, normalize=True)
needed_timesteps = WINDOW_SIZE + FORECAST_HORIZON
print(f"Dataset info:")
print(f"  Total timesteps available: {features.shape[0]}")
print(f"  Window size (input): {WINDOW_SIZE}")
print(f"  Forecast horizon: {FORECAST_HORIZON}")
print(f"  Minimum needed: {WINDOW_SIZE + FORECAST_HORIZON}")
print(f"  Valid sample range: {WINDOW_SIZE} to {features.shape[0] - FORECAST_HORIZON}")

dataset = WeatherGraphDataset(
    features, edge_index, window_size=WINDOW_SIZE, forecast_horizon=FORECAST_HORIZON
)
print("WeatherGraphDataset size:", len(dataset))
if len(dataset) == 0:
    print("ERROR: Not enough timesteps for even one sample.")
    print(
        f"Need at least {WINDOW_SIZE + FORECAST_HORIZON + 1} timesteps, but have {features.shape[0]}"
    )
    exit(1)
else:
    print(f"Successfully created {len(dataset)} samples")

# --- Get normalization stats for all variables
if isinstance(stats, dict):
    mean = np.array(stats["mean"])
    std = np.array(stats["std"])
else:
    mean, std = stats

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
]  # Update if your order is different!
nvars = len(VAR_NAMES)

# ========== DISPLAY FIRST 3 DAYS INPUT + OUTPUT ==========
print("\n" + "=" * 80)
print("FIRST 3 DAYS INPUT + 1 DAY FORECAST OUTPUT")
print("=" * 80)

# Use only the first sample (idx=0)
window = dataset[0]
x_cpu = window.x.cpu().numpy()
y_true = window.y.cpu().numpy()
with torch.no_grad():
    y_pred = model(window.x.to(DEVICE), window.edge_index.to(DEVICE)).cpu().numpy()

# Average over nodes and handle shapes
print(f"x_cpu shape: {x_cpu.shape}")
print(f"y_true shape: {y_true.shape}")
print(f"y_pred shape: {y_pred.shape}")

# Reshape x_cpu to get features per timestep
if x_cpu.ndim == 2:  # [window*nodes, features]
    x_reshaped = x_cpu.reshape(WINDOW_SIZE, num_nodes, -1)
    x_avg = x_reshaped.mean(axis=1)  # [window, features]
else:
    x_avg = x_cpu

# Reshape y_true and y_pred
if y_true.ndim == 2:  # [horizon*nodes, features]
    y_true_reshaped = y_true.reshape(FORECAST_HORIZON, num_nodes, nvars)
    y_true_avg = y_true_reshaped.mean(axis=1)  # [horizon, nvars]
else:
    y_true_avg = y_true.mean(axis=0)

if y_pred.ndim == 2:  # [horizon*nodes, features]
    y_pred_reshaped = y_pred.reshape(FORECAST_HORIZON, num_nodes, nvars)
    y_pred_avg = y_pred_reshaped.mean(axis=1)  # [horizon, nvars]
else:
    y_pred_avg = y_pred.mean(axis=0)

# Get timestamps
input_timesteps = ds_reg[time_name].values[0:WINDOW_SIZE]
forecast_timesteps = ds_reg[time_name].values[
    WINDOW_SIZE : WINDOW_SIZE + FORECAST_HORIZON
]

print("\nINPUT (Past 3 days - 72 hours):")
print("-" * 60)
for t_idx, ts in enumerate(input_timesteps):
    if t_idx >= x_avg.shape[0]:
        continue
    print(f"Time: {ts}")
    for v_idx in range(min(3, nvars)):  # Show first 3 variables
        if v_idx < x_avg.shape[1]:  # Check bounds
            true_phys = x_avg[t_idx, v_idx] * std[v_idx] + mean[v_idx]
            print(f"  {VAR_NAMES[v_idx]}: {true_phys:.2f}")
    print()

print("\nOUTPUT (Next 1 day - 24 hours):")
print("-" * 60)
for t_idx, ts in enumerate(forecast_timesteps):
    if t_idx >= y_true_avg.shape[0] or t_idx >= y_pred_avg.shape[0]:
        continue
    print(f"Time: {ts}")
    for v_idx in range(min(3, nvars)):  # Show first 3 variables
        if v_idx < y_true_avg.shape[1] and v_idx < y_pred_avg.shape[1]:  # Check bounds
            true_phys = y_true_avg[t_idx, v_idx] * std[v_idx] + mean[v_idx]
            pred_phys = y_pred_avg[t_idx, v_idx] * std[v_idx] + mean[v_idx]
            print(
                f"  {VAR_NAMES[v_idx]} - True: {true_phys:.2f}, Pred: {pred_phys:.2f}"
            )
    print()

# Save detailed CSV
timeline_save = "./Out_Data/first_3days_forecast.csv"
with open(timeline_save, "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    header = ["TIMESTAMP", "TYPE"]
    for var in VAR_NAMES:
        header += [f"{var}_TRUE", f"{var}_PRED"]
    writer.writerow(header)

    # Input data
    for t_idx, ts in enumerate(input_timesteps):
        if t_idx >= x_avg.shape[0]:
            continue
        row = [str(ts), "INPUT"]
        for v_idx in range(nvars):
            true_phys = x_avg[t_idx, v_idx] * std[v_idx] + mean[v_idx]
            row += [f"{true_phys:.4f}", ""]
        writer.writerow(row)

    # Forecast data
    for t_idx, ts in enumerate(forecast_timesteps):
        if t_idx >= y_true_avg.shape[0] or t_idx >= y_pred_avg.shape[0]:
            continue
        row = [str(ts), "FORECAST"]
        for v_idx in range(nvars):
            true_phys = y_true_avg[t_idx, v_idx] * std[v_idx] + mean[v_idx]
            pred_phys = y_pred_avg[t_idx, v_idx] * std[v_idx] + mean[v_idx]
            row += [f"{true_phys:.4f}", f"{pred_phys:.4f}"]
        writer.writerow(row)

print(f"Detailed data saved to: {timeline_save}")

# ========== FORECAST ACCURACY FOR FIRST SAMPLE ==========
print("\n" + "=" * 80)
print("FORECAST ACCURACY (First Sample Only)")
print("=" * 80)

for v_idx, var in enumerate(VAR_NAMES):
    true_vals = y_true_avg[:, v_idx] * std[v_idx] + mean[v_idx]
    pred_vals = y_pred_avg[:, v_idx] * std[v_idx] + mean[v_idx]
    rmse = np.sqrt(np.mean((pred_vals - true_vals) ** 2))
    bias = np.mean(pred_vals - true_vals)
    print(f"{var:6s}: RMSE={rmse:8.2f}, Bias={bias:8.2f}")

print("=" * 80)
