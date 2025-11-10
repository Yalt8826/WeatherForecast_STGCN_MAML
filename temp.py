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

MODEL_PATH = "./Out_Data/AdaptedModels/adapted_model_2021_(-40, -35, 140, 145).pt"
DATA_DIR = r"E:\\Study\\5th Sem Mini Project\\Datasets\\2025\\Jan2Mar"
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

# --- Generate timeline-format CSV
timeline_save = "./Out_Data/timeline_window_allvars.csv"

with open(timeline_save, "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    header = ["TIMESTAMP"]
    for var in VAR_NAMES:
        header += [
            f"{var} TRUE NORM",
            f"{var} TRUE PHYS",
            f"{var} PRED NORM",
            f"{var} PRED PHYS",
        ]
    writer.writerow(header)

    for idx in range(len(dataset)):
        window = dataset[idx]
        x_cpu = window.x.cpu().numpy()  # [window, nodes, nvars] or [window, nvars]
        y_true = window.y.cpu().numpy()  # [nodes, horizon, nvars]
        with torch.no_grad():
            y_pred = (
                model(window.x.to(DEVICE), window.edge_index.to(DEVICE)).cpu().numpy()
            )  # [nodes, horizon, nvars]

        # Normalize shapes
        if len(x_cpu.shape) == 3:
            x_avg = x_cpu.mean(axis=1)  # [window, nvars]
        else:
            x_avg = x_cpu  # [window, nvars]
        y_true_avg = y_true.mean(axis=0)  # [horizon, nvars]
        y_pred_avg = y_pred.mean(axis=0)  # [horizon, nvars]

        input_timesteps = ds_reg[time_name].values[idx : idx + WINDOW_SIZE]
        forecast_timesteps = ds_reg[time_name].values[
            idx + WINDOW_SIZE : idx + WINDOW_SIZE + FORECAST_HORIZON
        ]

        # --- History rows (input, no prediction)
        for t_idx, ts in enumerate(input_timesteps):
            row = [str(ts)]
            for v_idx in range(nvars):
                true_norm = x_avg[t_idx, v_idx]
                true_phys = true_norm * std[v_idx] + mean[v_idx]
                row += [f"{true_norm:.4f}", f"{true_phys:.4f}", "", ""]
            writer.writerow(row)

        # --- Prediction rows (forecast)
        for t_idx, ts in enumerate(forecast_timesteps):
            row = [str(ts)]
            for v_idx in range(nvars):
                true_norm = y_true_avg[t_idx, v_idx]
                true_phys = true_norm * std[v_idx] + mean[v_idx]
                pred_norm = y_pred_avg[t_idx, v_idx]
                pred_phys = pred_norm * std[v_idx] + mean[v_idx]
                row += [
                    f"{true_norm:.4f}",
                    f"{true_phys:.4f}",
                    f"{pred_norm:.4f}",
                    f"{pred_phys:.4f}",
                ]
            writer.writerow(row)

print(f"(Saved: {timeline_save})")
