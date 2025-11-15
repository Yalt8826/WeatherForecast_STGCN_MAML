import torch
import torch.nn as nn
import numpy as np
import copy
import time
import os
import xarray as xr
import random
from torch_geometric.loader import DataLoader
from torch.utils.data import Subset
from dataLoader import main_dataloader
from embed_utils import add_time_embeddings, KoppenEmbedding
from graphBuilder import build_spatial_graph
from featurePreprocessor import prepare_model_input
from dataset import WeatherGraphDataset
from model import STGCN
from hybrid_model import HybridSTGCN_LSTM

# ========== MODEL 4.0 ULTRA SCALED CONFIG ==========
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

NUM_EPOCHS = 40  # Extended training
BATCH_SIZE = 2  # Keep manageable
INNER_EPOCHS_PER_TASK = 4  # Deep inner adaptation
INNER_LR = 0.005
OUTER_LR = 0.0008  # Slightly lower for stability
WINDOW_SIZE = 24  # Full temporal context
FORECAST_HORIZON = 8  # Extended predictions
HIDDEN_CHANNELS = 128  # Maximum GCN capacity
LSTM_HIDDEN_SIZE = 64  # Large LSTM
LSTM_NUM_LAYERS = 2  # Multi-layer temporal processing

INPUT_CHANNELS = 12 + 4 + 8  # 24 total
OUTPUT_CHANNELS = 12
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Complete global coverage - 15 regions
MODEL_4_REGIONS = [
    (18, 23, 75, 80),  # India
    (8, 13, 98, 103),  # Thailand
    (53, 58, 35, 40),  # Russia
    (12.5, 17.5, 102.5, 107.5),  # Thailand/Cambodia
    (22.5, 27.5, 19.5, 24.5),  # Libya/Egypt
    (43.5, 48.5, 7.5, 12.5),  # Southern France
    (35.5, 40.5, -5.5, -0.5),  # Spain/Mediterranean
    (32.5, 37.5, 137.5, 142.5),  # Tokyo/Eastern Japan
    (-23.5, -18.5, 132.5, 137.5),  # Australia
    (-20, -15, -70, -65),  # Peru
    (44.5, 49.5, 125.5, 130.5),  # Northeast China
    (29.5, 34.5, -101.5, -96.5),  # Texas
    (-9.5, -4.5, -67.5, -62.5),  # Amazon Basin
    (67.5, 72.5, -32.5, -27.5),  # Greenland
    (51.5, 56.5, -112.5, -107.5),  # Alberta, Canada
]

print("=" * 80)
print("🚀 HYBRID STGCN+LSTM MAML MODEL 4.0 - ULTRA SCALED")
print("=" * 80)
print(f"Device: {DEVICE}")
print(f"Regions: {len(MODEL_4_REGIONS)} (Complete Global Coverage)")
print(f"Window: {WINDOW_SIZE}, Forecast: {FORECAST_HORIZON}")
print(
    f"Hidden: {HIDDEN_CHANNELS}, LSTM: {LSTM_HIDDEN_SIZE} (x{LSTM_NUM_LAYERS} layers)"
)
print(f"Epochs: {NUM_EPOCHS}")
print("=" * 80)


def create_v4_task(region_bounds, koppen_embed):
    print(f"Loading region: {region_bounds}")
    lat_min, lat_max, lon_min, lon_max = region_bounds
    cache_path = f"Out_Data/lat{lat_min}-{lat_max}_lon{lon_min}-{lon_max}.nc"

    if os.path.exists(cache_path):
        ds = xr.open_dataset(cache_path)
        koppen_code = int(ds.attrs.get("koppen_code", 0))
    else:
        print("  Loading from scratch...")
        ds, koppen_code, _ = main_dataloader(lat_min, lat_max, lon_min, lon_max)
        ds = add_time_embeddings(ds)

    if "day_of_year_sin" not in ds:
        ds = add_time_embeddings(ds)

    edge_index, num_nodes, _ = build_spatial_graph(ds, k_neighbors=4)
    features, stats = prepare_model_input(ds, koppen_code, koppen_embed, normalize=True)

    dataset = WeatherGraphDataset(
        features, edge_index, window_size=WINDOW_SIZE, forecast_horizon=FORECAST_HORIZON
    )

    # Rich training data for Model 4.0
    max_samples = min(400, len(dataset))
    support_size = int(0.8 * max_samples)

    support_indices = list(range(0, support_size))
    query_indices = list(range(support_size, max_samples))

    support_ds = Subset(dataset, support_indices)
    query_ds = Subset(dataset, query_indices)

    print(f"  ✅ Support: {len(support_ds)}, Query: {len(query_ds)}")
    return support_ds, query_ds, stats


def inner_loop_v4(hybrid_model, koppen_embed, support_ds, device):
    temp_model = copy.deepcopy(hybrid_model)
    temp_koppen = copy.deepcopy(koppen_embed)
    temp_model.train()
    temp_koppen.train()

    optimizer = torch.optim.SGD(
        list(temp_model.parameters()) + list(temp_koppen.parameters()), lr=INNER_LR
    )
    criterion = nn.MSELoss()

    support_loader = DataLoader(support_ds, batch_size=1, shuffle=False)

    # Deep inner training for Model 4.0
    for epoch in range(INNER_EPOCHS_PER_TASK):
        for batch_idx, batch in enumerate(support_loader):
            if batch_idx >= 12:  # More batches for thorough adaptation
                break

            batch = batch.to(device)
            optimizer.zero_grad()
            out = temp_model(batch.x, batch.edge_index)
            loss = criterion(out, batch.y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(temp_model.parameters()) + list(temp_koppen.parameters()),
                max_norm=1.0,
            )
            optimizer.step()

    return temp_model, temp_koppen


def meta_update_v4(hybrid_model, koppen_embed, tasks, device):
    meta_optimizer = torch.optim.Adam(
        list(hybrid_model.parameters()) + list(koppen_embed.parameters()), lr=OUTER_LR
    )
    meta_optimizer.zero_grad()
    criterion = nn.MSELoss()
    meta_loss = 0.0

    for support_ds, query_ds, stats in tasks:
        if support_ds is None:
            continue

        # Inner adaptation
        adapted_model, adapted_koppen = inner_loop_v4(
            hybrid_model, koppen_embed, support_ds, device
        )
        adapted_model.train()

        # Query evaluation
        query_loader = DataLoader(query_ds, batch_size=1, shuffle=False)
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
            list(hybrid_model.parameters()) + list(koppen_embed.parameters()),
            max_norm=1.0,
        )
        meta_optimizer.step()
        return meta_loss.item()
    return 0.0


def main():
    print("Creating Model 4.0 Ultra Scaled Hybrid...")

    # Maximum capacity STGCN
    base_stgcn = STGCN(
        in_channels=INPUT_CHANNELS,
        hidden_channels=HIDDEN_CHANNELS,
        out_channels=OUTPUT_CHANNELS,
        window_size=WINDOW_SIZE,
        forecast_horizon=FORECAST_HORIZON,
        dropout_rate=0.2,
    ).to(DEVICE)

    # Ultra scaled hybrid with multi-layer LSTM
    hybrid_model = HybridSTGCN_LSTM(
        base_stgcn=base_stgcn,
        lstm_hidden_size=LSTM_HIDDEN_SIZE,
        lstm_num_layers=LSTM_NUM_LAYERS,
        lstm_dropout=0.2,
        out_channels=OUTPUT_CHANNELS,
        forecast_horizon=FORECAST_HORIZON,
        freeze_base=False,
    ).to(DEVICE)

    koppen_embed = KoppenEmbedding(embedding_dim=8).to(DEVICE)
    hybrid_model.lstm.flatten_parameters()

    total_params = sum(p.numel() for p in hybrid_model.parameters())
    print(f"✅ Model 4.0 created: {total_params:,} parameters")
    print(
        f"   Architecture: {HIDDEN_CHANNELS}H + {LSTM_HIDDEN_SIZE}x{LSTM_NUM_LAYERS}L"
    )
    print(f"   Sequence: {WINDOW_SIZE}→{FORECAST_HORIZON}")

    # Load Model 4.0 tasks
    print("\nLoading Model 4.0 tasks...")
    all_tasks = []
    for region in MODEL_4_REGIONS:
        try:
            task = create_v4_task(region, koppen_embed)
            if task[0] is not None:
                all_tasks.append(task)
        except Exception as e:
            print(f"  ❌ Error loading {region}: {e}")
            continue

    if not all_tasks:
        print("❌ No tasks loaded!")
        return

    print(f"\n✅ Loaded {len(all_tasks)} Model 4.0 tasks")

    # Model 4.0 training
    print("\n" + "=" * 80)
    print("🚀 STARTING MODEL 4.0 ULTRA TRAINING")
    print("=" * 80)

    best_loss = float("inf")
    log_file = "./Out_Data/hybrid_maml_v4_log.csv"

    with open(log_file, "w") as f:
        f.write("epoch,meta_loss\n")

    for epoch in range(NUM_EPOCHS):
        start_time = time.time()

        # Sample batch of tasks
        if len(all_tasks) > BATCH_SIZE:
            indices = np.random.choice(len(all_tasks), BATCH_SIZE, replace=False)
            batch_tasks = [all_tasks[i] for i in indices]
        else:
            batch_tasks = all_tasks

        loss = meta_update_v4(hybrid_model, koppen_embed, batch_tasks, DEVICE)

        epoch_time = time.time() - start_time
        print(
            f"Epoch {epoch+1}/{NUM_EPOCHS} - Loss: {loss:.4f} - Time: {epoch_time:.1f}s"
        )

        # Log progress
        with open(log_file, "a") as f:
            f.write(f"{epoch+1},{loss}\n")

        # Save best model
        if loss < best_loss:
            best_loss = loss
            save_path = "./Out_Data/SavedModels/hybrid_maml_model_v4.pt"
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(
                {
                    "hybrid_model_state_dict": hybrid_model.state_dict(),
                    "koppen_embed_state_dict": koppen_embed.state_dict(),
                    "meta_loss": best_loss,
                    "epoch": epoch,
                    "model_version": "4.0",
                    "total_params": total_params,
                    "config": {
                        "input_channels": INPUT_CHANNELS,
                        "hidden_channels": HIDDEN_CHANNELS,
                        "output_channels": OUTPUT_CHANNELS,
                        "window_size": WINDOW_SIZE,
                        "forecast_horizon": FORECAST_HORIZON,
                    },
                    "hybrid_config": {
                        "lstm_hidden_size": LSTM_HIDDEN_SIZE,
                        "lstm_num_layers": LSTM_NUM_LAYERS,
                        "lstm_dropout": 0.2,
                    },
                },
                save_path,
            )

        # Memory cleanup every 5 epochs
        if epoch % 5 == 0:
            torch.cuda.empty_cache()

    print("\n" + "=" * 80)
    print("✅ MODEL 4.0 ULTRA TRAINING COMPLETE!")
    print("=" * 80)
    print(f"Model Version: 4.0 Ultra Scaled")
    print(f"Total Parameters: {total_params:,}")
    print(f"Best Loss: {best_loss:.4f}")
    print(f"Architecture: {HIDDEN_CHANNELS}H + {LSTM_HIDDEN_SIZE}x{LSTM_NUM_LAYERS}L")
    print(f"Sequence: {WINDOW_SIZE}→{FORECAST_HORIZON}")
    print(f"Model saved: ./Out_Data/SavedModels/hybrid_maml_model_v4.pt")
    print(f"Training log: {log_file}")
    print("=" * 80)


if __name__ == "__main__":
    main()
