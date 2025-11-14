import torch
import torch.nn as nn
import numpy as np
import copy
import time
import os
import xarray as xr
import random
from torch_geometric.loader import DataLoader
from torch.utils.data import random_split, Subset
from dataLoader import main_dataloader
from embed_utils import add_time_embeddings, KoppenEmbedding
from graphBuilder import build_spatial_graph
from featurePreprocessor import prepare_model_input, WEATHER_VARS
from dataset import WeatherGraphDataset
from model import STGCN

# ========== Reproducibility ==========
SEED = 42
np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# ========== CONFIGURATION ==========
NUM_EPOCHS = 50
BATCH_SIZE = 3
INNER_BATCH_SIZE = 1
INNER_EPOCHS_PER_TASK = 3
INNER_LR = 0.005
OUTER_LR = 0.0005
MAX_GRAD_NORM = 1.0
WEIGHT_DECAY = 1e-4
WINDOW_SIZE = 24
FORECAST_HORIZON = 8
NUM_WEATHER_VARS = 12

KOPPEN_EMBED_DIM = 8
INPUT_CHANNELS = NUM_WEATHER_VARS + 4 + KOPPEN_EMBED_DIM
HIDDEN_CHANNELS = 128
OUTPUT_CHANNELS = NUM_WEATHER_VARS

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

META_TRAIN_REGIONS = [
    (-9.5, -4.5, -67.5, -62.5),  # Amazon Basin, Brazil
    (12.5, 17.5, 102.5, 107.5),  # Thailand/Cambodia
    (22.5, 27.5, 19.5, 24.5),  # Libya/Egypt
    (-23.5, -18.5, 132.5, 137.5),  # Northern Territory, Australia
    (43.5, 48.5, 7.5, 12.5),  # Southern France
    (35.5, 40.5, -5.5, -0.5),  # Spain/Mediterranean
    (53.5, 58.5, 34.5, 39.5),  # Central Russia
    (44.5, 49.5, 125.5, 130.5),  # Northeast China/Manchuria
    (67.5, 72.5, -32.5, -27.5),  # Greenland
    (-20, -15, -70, -65),  # Peru/Western Amazon
    (32.5, 37.5, 137.5, 142.5),  # Tokyo/Eastern Japan
    (-35.5, -30.5, 16.5, 21.5),  # South Africa
    (51.5, 56.5, -112.5, -107.5),  # Alberta, Canada
    (29.5, 34.5, -101.5, -96.5),  # Texas, USA
    (11.5, 16.5, 86.5, 91.5),  # Bangladesh/India
]


print("\n" + "=" * 80)
print("MAML TRAINING: MULTI-VARIABLE WEATHER FORECASTING")
print("FINAL OPTIMIZED VERSION WITH STABILITY & PERFORMANCE FIXES")
print("=" * 80)
print(f"Device: {DEVICE}")
print(f"Architecture: 4-layer GCN, {HIDDEN_CHANNELS} hidden")
print(f"Epochs: {NUM_EPOCHS}")
print(f"Inner LR: {INNER_LR}, Outer LR: {OUTER_LR}")
print(f"Gradient Clipping: {MAX_GRAD_NORM}")
print(f"Weight Decay: {WEIGHT_DECAY}")
print("=" * 80 + "\n")


# ========== DATA LOADING AND TASK CREATION ==========
def load_cached_or_create(region_bounds, koppen_embed):
    """
    Loads a region dataset from cache if available, else loads fresh and embeds time.
    """
    lat_min, lat_max, lon_min, lon_max = region_bounds
    cache_path = f"Out_Data/lat{lat_min}-{lat_max}_lon{lon_min}-{lon_max}.nc"
    if os.path.exists(cache_path):
        print(f"  ✅ Loading cached: {cache_path}")
        ds = xr.open_dataset(cache_path)
        koppen_code = int(ds.attrs.get("koppen_code", 0))
    else:
        print(f"  ⚠️  No cache, loading from scratch...")
        ds, koppen_code, _ = main_dataloader(lat_min, lat_max, lon_min, lon_max)
        ds = add_time_embeddings(ds)
    if "day_of_year_sin" not in ds:
        ds = add_time_embeddings(ds)
    return ds, koppen_code


def chronological_split(dataset, support_ratio=0.8, gap=0):
    """
    Chronological split with anti-leakage optional gap: [support][gap][query]
    """
    n = len(dataset)
    split = int(support_ratio * n)
    q_start = min(split + gap, n)
    support_idx = list(range(0, split))
    query_idx = list(range(q_start, n))
    return Subset(dataset, support_idx), Subset(dataset, query_idx)


def create_task(region_bounds, koppen_embed, window_size=WINDOW_SIZE, support_size=0.8):
    ds, koppen_code = load_cached_or_create(region_bounds, koppen_embed)
    edge_index, num_nodes, _ = build_spatial_graph(ds, k_neighbors=4)
    features, stats = prepare_model_input(ds, koppen_code, koppen_embed, normalize=True)
    dataset = WeatherGraphDataset(
        features, edge_index, window_size=window_size, forecast_horizon=FORECAST_HORIZON
    )
    support_len = int(support_size * len(dataset))
    support_ds, query_ds = chronological_split(
        dataset, support_ratio=0.8, gap=WINDOW_SIZE
    )
    return support_ds, query_ds, stats


# ========== INNER LOOP ==========
def inner_loop(model, koppen_embed, support_ds, inner_epochs, inner_lr, device):
    """
    Simulates fast adaptation (inner loop of MAML); only sees support set.
    """
    temp_model = copy.deepcopy(model)
    temp_koppen = copy.deepcopy(koppen_embed)
    temp_model.train()
    temp_koppen.train()
    optimizer = torch.optim.SGD(
        list(temp_model.parameters()) + list(temp_koppen.parameters()),
        lr=inner_lr,
        weight_decay=WEIGHT_DECAY,
    )
    criterion = nn.MSELoss()
    support_loader = DataLoader(
        support_ds,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=(device == "cuda"),
    )
    losses = []
    for epoch in range(inner_epochs):
        for batch in support_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            out = temp_model(batch.x, batch.edge_index)
            loss = criterion(out, batch.y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(temp_model.parameters()) + list(temp_koppen.parameters()),
                max_norm=MAX_GRAD_NORM,
            )
            optimizer.step()
            losses.append(loss.item())
    return temp_model, temp_koppen, losses


# ========== OUTER LOOP ==========
def meta_update(model, koppen_embed, tasks, inner_epochs, inner_lr, outer_lr, device):
    """
    Performs the meta-update (outer loop of MAML): adapts per task,
    evaluates on one query batch per task (low memory, correct gradients).
    """
    meta_loss = 0.0
    criterion = nn.MSELoss()
    meta_optimizer = torch.optim.Adam(
        list(model.parameters()) + list(koppen_embed.parameters()),
        lr=outer_lr,
        weight_decay=WEIGHT_DECAY,
    )
    meta_optimizer.zero_grad()
    for support_ds, query_ds, stats in tasks:
        adapted_model, adapted_koppen, _ = inner_loop(
            model, koppen_embed, support_ds, inner_epochs, inner_lr, device
        )
        adapted_model.eval()
        query_loader = DataLoader(
            query_ds,
            batch_size=1,
            shuffle=False,
            num_workers=0,
            pin_memory=(device == "cuda"),
        )
        # Use ONLY the first batch in query set to compute meta-loss (with gradient tracking)
        query_batch = next(iter(query_loader))
        query_batch = query_batch.to(device)
        query_out = adapted_model(query_batch.x, query_batch.edge_index)
        query_loss = criterion(query_out, query_batch.y)
        meta_loss += query_loss
        del adapted_model, adapted_koppen
        torch.cuda.empty_cache()
    if len(tasks) > 0:
        meta_loss = meta_loss / len(tasks)
    meta_loss.backward()
    torch.nn.utils.clip_grad_norm_(
        list(model.parameters()) + list(koppen_embed.parameters()),
        max_norm=MAX_GRAD_NORM,
    )
    meta_optimizer.step()
    return meta_loss.item()


# ========== MAIN ==========
def main():
    print("Initializing model...")
    model = STGCN(
        in_channels=INPUT_CHANNELS,
        hidden_channels=HIDDEN_CHANNELS,
        out_channels=OUTPUT_CHANNELS,
        window_size=WINDOW_SIZE,
        forecast_horizon=FORECAST_HORIZON,
        dropout_rate=0.3,
    ).to(DEVICE)
    koppen_embed = KoppenEmbedding(embedding_dim=KOPPEN_EMBED_DIM).to(DEVICE)
    print(f"✅ STGCN initialized:")
    print(f"   Input channels: {INPUT_CHANNELS}")
    print(f"   Hidden channels: {HIDDEN_CHANNELS}")
    print(f"   Output channels: {OUTPUT_CHANNELS}")
    print(f"   Window size: {WINDOW_SIZE}")
    print(f"   Forecast horizon: {FORECAST_HORIZON}")
    print(f"✅ Total parameters: {sum(p.numel() for p in model.parameters()):,}\n")
    print("Loading tasks...")
    all_tasks = []
    for i, region in enumerate(META_TRAIN_REGIONS):
        print(f"\nRegion {i+1}: {region}")
        try:
            support_ds, query_ds, stats = create_task(region, koppen_embed)
            all_tasks.append((support_ds, query_ds, stats))
            print(f"  ✅ Support: {len(support_ds)}, Query: {len(query_ds)}")
        except Exception as e:
            print(f"  ❌ Error: {e}")
            continue
    if not all_tasks:
        print("\n❌ No tasks loaded!")
        return
    print(f"\n✅ Loaded {len(all_tasks)} tasks\n")
    print("=" * 80)
    print("STARTING TRAINING")
    print("=" * 80)
    best_loss = float("inf")
    start = time.time()
    DO_LOG = True
    log_file = "./Out_Data/maml_training_log.csv"
    if DO_LOG:
        with open(log_file, "w") as f:
            f.write("epoch,meta_loss\n")
    for epoch in range(NUM_EPOCHS):
        epoch_start = time.time()
        if len(all_tasks) > BATCH_SIZE:
            indices = np.random.choice(len(all_tasks), BATCH_SIZE, replace=False)
            batch_tasks = [all_tasks[i] for i in indices]
        else:
            batch_tasks = all_tasks
        loss = meta_update(
            model,
            koppen_embed,
            batch_tasks,
            INNER_EPOCHS_PER_TASK,
            INNER_LR,
            OUTER_LR,
            DEVICE,
        )
        epoch_time = time.time() - epoch_start
        print(
            f"Epoch {epoch+1}/{NUM_EPOCHS} - MetaLoss: {loss:.4f} - Time: {epoch_time:.1f}s"
        )
        if DO_LOG:
            with open(log_file, "a") as f:
                f.write(f"{epoch+1},{loss}\n")
        if loss < best_loss:
            best_loss = loss
            save_dir = "./Out_Data/SavedModels"
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(
                save_dir, "maml_model_multivar_(LongerWindowSize&LRRate).pt"
            )
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "koppen_embed_state_dict": koppen_embed.state_dict(),
                    "meta_loss": best_loss,
                    "epoch": epoch,
                    "config": {
                        "input_channels": INPUT_CHANNELS,
                        "hidden_channels": HIDDEN_CHANNELS,
                        "output_channels": OUTPUT_CHANNELS,
                        "window_size": WINDOW_SIZE,
                        "forecast_horizon": FORECAST_HORIZON,
                        "inner_lr": INNER_LR,
                        "outer_lr": OUTER_LR,
                        "max_grad_norm": MAX_GRAD_NORM,
                        "weight_decay": WEIGHT_DECAY,
                    },
                },
                save_path,
            )
        # Extra safety: clear cache after each epoch
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
    total = time.time() - start
    print("\n" + "=" * 80)
    print("COMPLETE!")
    print("=" * 80)
    print(f"Time: {total/3600:.2f} hours")
    print(f"Best loss achieved: {best_loss:.4f}")
    print("=" * 80)


if __name__ == "__main__":
    main()
