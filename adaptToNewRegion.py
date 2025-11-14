import torch
import xarray as xr
import os
from torch_geometric.loader import DataLoader
from torch.utils.data import Subset
import gc

from embed_utils import add_time_embeddings, KoppenEmbedding
from graphBuilder import build_spatial_graph
from featurePreprocessor import prepare_model_input
from dataset import WeatherGraphDataset
from model import STGCN
from hybrid_model import HybridSTGCN_LSTM

# Convert -75 to -70 longitude to 0-360 format
# -75° = 360° - 75° = 285°
# -70° = 360° - 70° = 290°
NEW_REGION = (8, 13, 98, 103)
REGION_NAME = "Thailand"


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"ADAPTING HYBRID MODEL TO {REGION_NAME}")
print(f"Region coordinates: {NEW_REGION}")

lat_min, lat_max, lon_min, lon_max = NEW_REGION


def load_region_data():
    datasets = []

    # 2023 Q1
    base_path = "E:/Study/5th Sem Mini Project/Datasets/2023/Jan2Mar"
    accum_file = os.path.join(base_path, "data_stream-oper_stepType-accum.nc")
    instant_file = os.path.join(base_path, "data_stream-oper_stepType-instant.nc")

    print("Loading 2023 Q1...")
    ds_accum = xr.open_dataset(accum_file)
    ds_instant = xr.open_dataset(instant_file)
    ds_2023 = xr.merge([ds_accum, ds_instant])
    ds_2023 = ds_2023.sel(
        latitude=slice(lat_max, lat_min), longitude=slice(lon_min, lon_max)
    )
    datasets.append(ds_2023)

    # 2024 Q1
    base_path = "E:/Study/5th Sem Mini Project/Datasets/2024/Jan2Mar"
    accum_file = os.path.join(base_path, "data_stream-oper_stepType-accum.nc")
    instant_file = os.path.join(base_path, "data_stream-oper_stepType-instant.nc")

    print("Loading 2024 Q1...")
    ds_accum = xr.open_dataset(accum_file)
    ds_instant = xr.open_dataset(instant_file)
    ds_2024 = xr.merge([ds_accum, ds_instant])
    ds_2024 = ds_2024.sel(
        latitude=slice(lat_max, lat_min), longitude=slice(lon_min, lon_max)
    )
    datasets.append(ds_2024)

    return xr.concat(datasets, dim="valid_time")


# Load data
ds = load_region_data()
print(f"Data shape for {REGION_NAME}: {dict(ds.sizes)}")

if "day_of_year_sin" not in ds:
    ds = add_time_embeddings(ds)

# Build graph
edge_index, num_nodes, _ = build_spatial_graph(ds, k_neighbors=4)
koppen_embed = KoppenEmbedding(embedding_dim=8).to(DEVICE)
features, stats = prepare_model_input(ds, 0, koppen_embed, normalize=True)

# Create dataset
dataset = WeatherGraphDataset(features, edge_index, window_size=24, forecast_horizon=8)
print(f"Dataset size: {len(dataset)}")

# Training split
max_samples = min(600, len(dataset))
train_indices = list(range(0, int(0.8 * max_samples)))
val_indices = list(range(int(0.8 * max_samples), max_samples))

train_ds = Subset(dataset, train_indices)
val_ds = Subset(dataset, val_indices)

print(f"Training samples: {len(train_ds)}")
print(f"Validation samples: {len(val_ds)}")

# Load base model
MODEL_PATH = "./Out_Data/SavedModels/maml_model_multivar_(LongerWindowSize&LRRate).pt"
checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)

base_model = STGCN(
    in_channels=checkpoint["config"]["input_channels"],
    hidden_channels=checkpoint["config"]["hidden_channels"],
    out_channels=checkpoint["config"]["output_channels"],
    window_size=checkpoint["config"]["window_size"],
    forecast_horizon=checkpoint["config"]["forecast_horizon"],
    dropout_rate=0.3,
).to(DEVICE)

base_model.load_state_dict(checkpoint["model_state_dict"])
koppen_embed.load_state_dict(checkpoint["koppen_embed_state_dict"])

# Create hybrid model
hybrid_model = HybridSTGCN_LSTM(
    base_stgcn=base_model,
    lstm_hidden_size=64,
    lstm_num_layers=2,
    lstm_dropout=0.2,
    out_channels=12,
    forecast_horizon=8,
    freeze_base=True,
).to(DEVICE)

print(f"Hybrid model created for {REGION_NAME}")

# Training
hybrid_model.train()
optimizer = torch.optim.Adam(hybrid_model.get_trainable_parameters(), lr=0.002)
criterion = torch.nn.MSELoss()

train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=0)

print(f"\nTraining hybrid model for {REGION_NAME}...")
epochs = 10

for epoch in range(epochs):
    epoch_losses = []

    for batch_idx, batch in enumerate(train_loader):
        if batch_idx % 20 == 0:
            torch.cuda.empty_cache()

        batch = batch.to(DEVICE)
        optimizer.zero_grad()

        out = hybrid_model(batch.x, batch.edge_index)
        loss = criterion(out, batch.y)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            hybrid_model.get_trainable_parameters(), max_norm=1.0
        )
        optimizer.step()

        epoch_losses.append(loss.item())

    avg_loss = sum(epoch_losses) / len(epoch_losses)
    print(f"Epoch {epoch+1}/{epochs}: Loss = {avg_loss:.6f}")

# Validation
hybrid_model.eval()
val_loader = DataLoader(val_ds, batch_size=1, shuffle=False)

total_loss = 0.0
count = 0

with torch.no_grad():
    for batch in val_loader:
        batch = batch.to(DEVICE)
        out = hybrid_model(batch.x, batch.edge_index)
        loss = criterion(out, batch.y)
        total_loss += loss.item()
        count += 1

avg_val_loss = total_loss / count
print(f"Validation MSE for {REGION_NAME}: {avg_val_loss:.6f}")

# Save adapted model
save_dir = "./Out_Data/AdaptedModels"
os.makedirs(save_dir, exist_ok=True)
save_path = os.path.join(save_dir, f"hybrid_adapted_{REGION_NAME}_{NEW_REGION}.pt")

torch.save(
    {
        "hybrid_model_state_dict": hybrid_model.state_dict(),
        "base_model_state_dict": base_model.state_dict(),
        "koppen_embed_state_dict": koppen_embed.state_dict(),
        "region": NEW_REGION,
        "region_name": REGION_NAME,
        "stats": stats,
        "config": checkpoint["config"],
        "hybrid_config": {
            "lstm_hidden_size": 64,
            "lstm_num_layers": 2,
            "lstm_dropout": 0.2,
            "forecast_horizon": 8,
        },
    },
    save_path,
)

print(f"\n✅ Model adapted for {REGION_NAME} saved to: {save_path}")
print(f"Ready for validation on {REGION_NAME} region!")
torch.cuda.empty_cache()
