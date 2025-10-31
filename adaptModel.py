import torch
import xarray as xr
import os
from torch_geometric.loader import DataLoader
from torch.utils.data import Subset
import copy

# Import your project modules
from embed_utils import add_time_embeddings, KoppenEmbedding

from dataLoader import main_dataloader
from graphBuilder import build_spatial_graph
from featurePreprocessor import prepare_model_input
from dataset import WeatherGraphDataset
from model import STGCN

# Configurable parameters
YEAR = 2021
REGION_BOUNDS = (-40, -35, 140, 145)
MODEL_PATH = "./Out_Data/SavedModels/maml_model_multivar_(LongerWindowSize&LRRate).pt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 1. Load region data
lat_min, lat_max, lon_min, lon_max = REGION_BOUNDS
ds, koppen_code, _ = main_dataloader(lat_min, lat_max, lon_min, lon_max)

# Ensure time embedding is present
if "day_of_year_sin" not in ds:
    ds = add_time_embeddings(ds)

# 2. Filter to one year using 'valid_time' coordinate (common in weather datasets)
if "valid_time" in ds.coords:
    # If valid_time is datetime64 type, use .dt.year
    try:
        ds = ds.sel(valid_time=ds["valid_time"].dt.year == YEAR)
    except Exception as e:
        # In case 'valid_time' is not datetime64, show diagnostic info
        print("Error filtering by year, 'valid_time' type:", ds["valid_time"].dtype)
        raise
else:
    print("No 'valid_time' coordinate found. Available coords:", list(ds.coords))
    raise KeyError("Did not find 'valid_time' for time filtering.")

# 3. Build graph, features, windowed dataset
edge_index, num_nodes, _ = build_spatial_graph(ds, k_neighbors=4)
koppen_embed_dim = 8
koppen_embed = KoppenEmbedding(embedding_dim=koppen_embed_dim).to(DEVICE)
features, stats = prepare_model_input(ds, koppen_code, koppen_embed, normalize=True)

WINDOW_SIZE = 24
FORECAST_HORIZON = 8
dataset = WeatherGraphDataset(
    features, edge_index, window_size=WINDOW_SIZE, forecast_horizon=FORECAST_HORIZON
)

# 4. Chronological split for adaptation
n = len(dataset)
split = int(0.8 * n)
support_idx = list(range(0, split))
query_idx = list(range(split, n))
support_ds = Subset(dataset, support_idx)
query_ds = Subset(dataset, query_idx)

# 5. Load meta-trained model and embedding
checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
model = STGCN(
    in_channels=checkpoint["config"]["input_channels"],
    hidden_channels=checkpoint["config"]["hidden_channels"],
    out_channels=checkpoint["config"]["output_channels"],
    window_size=checkpoint["config"]["window_size"],
    forecast_horizon=checkpoint["config"]["forecast_horizon"],
    dropout_rate=0.3,
).to(DEVICE)
koppen_embed = KoppenEmbedding(embedding_dim=koppen_embed_dim).to(DEVICE)
model.load_state_dict(checkpoint["model_state_dict"])
koppen_embed.load_state_dict(checkpoint["koppen_embed_state_dict"])
model.eval()
koppen_embed.eval()

# 6. Adaptation (inner loop) on support set
inner_epochs = 3
inner_lr = 0.005
weight_decay = checkpoint["config"]["weight_decay"]
max_grad_norm = checkpoint["config"]["max_grad_norm"]

temp_model = copy.deepcopy(model)
temp_koppen = copy.deepcopy(koppen_embed)
temp_model.train()
temp_koppen.train()
optimizer = torch.optim.SGD(
    list(temp_model.parameters()) + list(temp_koppen.parameters()),
    lr=inner_lr,
    weight_decay=weight_decay,
)
criterion = torch.nn.MSELoss()
support_loader = DataLoader(
    support_ds,
    batch_size=1,
    shuffle=False,
    num_workers=0,
    pin_memory=(DEVICE == "cuda"),
)

for epoch in range(inner_epochs):
    for batch in support_loader:
        batch = batch.to(DEVICE)
        optimizer.zero_grad()
        out = temp_model(batch.x, batch.edge_index)
        loss = criterion(out, batch.y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(temp_model.parameters()) + list(temp_koppen.parameters()),
            max_grad_norm,
        )
        optimizer.step()

# 7. Evaluation (MSE on query set—no further training)
temp_model.eval()
temp_koppen.eval()
query_loader = DataLoader(
    query_ds, batch_size=1, shuffle=False, num_workers=0, pin_memory=(DEVICE == "cuda")
)
total_loss = 0.0
n_batches = 0

with torch.no_grad():
    for batch in query_loader:
        batch = batch.to(DEVICE)
        out = temp_model(batch.x, batch.edge_index)
        loss = criterion(out, batch.y)
        total_loss += loss.item()
        n_batches += 1

avg_query_loss = total_loss / n_batches if n_batches > 0 else float("nan")
print(
    f"Meta-adapted model's query MSE on {YEAR} ({REGION_BOUNDS}): {avg_query_loss:.4f}"
)

save_dir = "./Out_Data/AdaptedModels"
os.makedirs(save_dir, exist_ok=True)
save_path = os.path.join(save_dir, f"adapted_model_{YEAR}_{REGION_BOUNDS}.pt")
torch.save(
    {
        "model_state_dict": temp_model.state_dict(),
        "koppen_embed_state_dict": temp_koppen.state_dict(),
        "year": YEAR,
        "region": REGION_BOUNDS,
        "stats": stats,  # Optional: save normalization stats
        "support_size": len(support_ds),
        "query_size": len(query_ds),
        "config": checkpoint["config"],
    },
    save_path,
)
print(f"Adapted model saved to: {save_path}")
