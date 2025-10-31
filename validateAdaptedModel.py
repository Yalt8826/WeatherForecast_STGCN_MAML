import torch
import xarray as xr
import numpy as np
import os
import csv
import pandas as pd

from model import STGCN
from graphBuilder import build_spatial_graph
from featurePreprocessor import prepare_model_input
from dataset import WeatherGraphDataset
from embed_utils import add_time_embeddings, KoppenEmbedding

MODEL_PATH = "./Out_Data/AdaptedModels/adapted_model_2021_(-40, -35, 140, 145).pt"
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
TIME_WINDOW = WINDOW_SIZE + FORECAST_HORIZON + 8

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
else:
    print("valid_time not found!")
    exit(1)
if "day_of_year_sin" not in ds_reg:
    ds_reg = add_time_embeddings(ds_reg)

edge_index, num_nodes, _ = build_spatial_graph(ds_reg, k_neighbors=4)
koppen_code = checkpoint.get("koppen_code", 0) if "koppen_code" in checkpoint else 0
features, stats = prepare_model_input(ds_reg, koppen_code, koppen_embed, normalize=True)
needed_timesteps = WINDOW_SIZE + FORECAST_HORIZON
print(
    f"Needed timesteps for one sample: {needed_timesteps} (you have {features.shape[0]})"
)

dataset = WeatherGraphDataset(
    features, edge_index, window_size=WINDOW_SIZE, forecast_horizon=FORECAST_HORIZON
)
print("WeatherGraphDataset size:", len(dataset))
if len(dataset) == 0:
    print("ERROR: Not enough timesteps for even one sample. Increase TIME_WINDOW.")
    exit(1)

# --- Get normalization stats for all variables
if isinstance(stats, dict):
    mean = np.array(stats["mean"])
    std = np.array(stats["std"])
else:
    mean, std = stats

# Variable names (align with order in features: update if your order is different)
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
]  # Update with your actual order!

nvars = len(VAR_NAMES)
header = (
    ["TIMESTEP"]
    + [f"{name} PRED" for name in VAR_NAMES]
    + [f"{name} TRUE" for name in VAR_NAMES]
    + [f"{name} PRED (PHYS)" for name in VAR_NAMES]
    + [f"{name} TRUE (PHYS)" for name in VAR_NAMES]
)

all_preds_phys = [[] for _ in range(nvars)]
all_trues_phys = [[] for _ in range(nvars)]

for idx in range(len(dataset)):
    window = dataset[idx]
    x = window.x.to(DEVICE)
    edge_idx = window.edge_index.to(DEVICE)
    y_true = window.y.cpu().numpy()
    with torch.no_grad():
        y_pred = model(x, edge_idx).cpu().numpy()
    if y_true.ndim == 2:
        y_true = y_true.reshape(num_nodes, FORECAST_HORIZON, nvars)
    if y_pred.ndim == 2:
        y_pred = y_pred.reshape(num_nodes, FORECAST_HORIZON, nvars)
    print(f"\n--- Sample {idx}: Averaged forecast per timestep (all variables) ---")
    print(" | ".join(header))
    table_save = f"./Out_Data/timestep_pred_true_sample{idx}_allvars_denorm.csv"
    with open(table_save, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(header)
        for t in range(FORECAST_HORIZON):
            row = [t]
            pred_norms, true_norms, pred_phys, true_phys = [], [], [], []
            for v_idx, var in enumerate(VAR_NAMES):
                mean_pred = y_pred[:, t, v_idx].mean()
                mean_true = y_true[:, t, v_idx].mean()
                pred_norms.append(mean_pred)
                true_norms.append(mean_true)
                mean_pred_phys = mean_pred * std[v_idx] + mean[v_idx]
                mean_true_phys = mean_true * std[v_idx] + mean[v_idx]
                pred_phys.append(mean_pred_phys)
                true_phys.append(mean_true_phys)
                # Save for RMSE stats
                all_preds_phys[v_idx].append(mean_pred_phys)
                all_trues_phys[v_idx].append(mean_true_phys)
            row += [f"{v:.4f}" for v in pred_norms]
            row += [f"{v:.4f}" for v in true_norms]
            row += [f"{v:.4f}" for v in pred_phys]
            row += [f"{v:.4f}" for v in true_phys]
            print(" | ".join(str(cell) for cell in row))
            writer.writerow(row)
    print(f"(Saved: {table_save})")

# ---- RMSE/Bias summary for all variables ----
print(
    "\n====== RMSE and Bias Statistics (Phys Units, averaged over all nodes, all samples, all horizons) ======"
)
for v_idx, var in enumerate(VAR_NAMES):
    arr_pred = np.array(all_preds_phys[v_idx])
    arr_true = np.array(all_trues_phys[v_idx])
    rmse = np.sqrt(np.mean((arr_pred - arr_true) ** 2))
    bias = np.mean(arr_pred - arr_true)
    print(
        f"{var}:\tRMSE = {rmse:.4f}\tbias = {bias:.4f}\tTrue avg = {arr_true.mean():.4f}\tPred avg = {arr_pred.mean():.4f}"
    )
print(
    "====================================================================================="
)
