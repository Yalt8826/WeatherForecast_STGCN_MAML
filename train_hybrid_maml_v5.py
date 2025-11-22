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
from torch.amp import autocast, GradScaler
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
BATCH_SIZE = 4  # Increased with gradient accumulation
INNER_EPOCHS_PER_TASK = 6  # Deeper adaptation
INNER_LR = 0.01  # Higher for faster adaptation
OUTER_LR = 0.001  # Optimized base rate
GRAD_ACCUMULATION_STEPS = 2  # Effective batch size = 8
WINDOW_SIZE = 24  # Full temporal context
FORECAST_HORIZON = 8  # Extended predictions
HIDDEN_CHANNELS = 256  # Ultra maximum GCN capacity
LSTM_HIDDEN_SIZE = 128  # Ultra large LSTM
LSTM_NUM_LAYERS = 4  # Deep multi-layer temporal processing

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
print("🚀 HYBRID STGCN+LSTM MAML MODEL 5.0 - ULTRA SCALED")
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

    # Rich training data for Model 4.0 - increased samples
    max_samples = min(600, len(dataset))  # More data for better performance
    support_size = int(0.75 * max_samples)  # More query data for evaluation

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

    # Deep inner training for Model 4.0 with mixed precision
    for epoch in range(INNER_EPOCHS_PER_TASK):
        for batch_idx, batch in enumerate(support_loader):
            if batch_idx >= 15:  # More batches for thorough adaptation
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


def meta_update_v4(hybrid_model, koppen_embed, tasks, device, meta_optimizer):
    criterion = nn.MSELoss()
    meta_loss = 0.0

    # Zero gradients at the start
    meta_optimizer.zero_grad()

    for i, (support_ds, query_ds, stats) in enumerate(tasks):
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
        query_loss = criterion(query_out, query_batch.y) / GRAD_ACCUMULATION_STEPS

        query_loss.backward()
        meta_loss += query_loss.item()

        # Update every accumulation steps
        if (i + 1) % GRAD_ACCUMULATION_STEPS == 0 or i == len(tasks) - 1:
            torch.nn.utils.clip_grad_norm_(
                list(hybrid_model.parameters()) + list(koppen_embed.parameters()),
                max_norm=1.0,
            )
            meta_optimizer.step()
            meta_optimizer.zero_grad()

        del adapted_model, adapted_koppen
        torch.cuda.empty_cache()

    return meta_loss


def main():
    print("Creating Model 5.0 Ultra Scaled Hybrid...")

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
    print(f"✅ Model 5.0 created: {total_params:,} parameters")
    print(
        f"   Architecture: {HIDDEN_CHANNELS}H + {LSTM_HIDDEN_SIZE}x{LSTM_NUM_LAYERS}L"
    )
    print(f"   Sequence: {WINDOW_SIZE}→{FORECAST_HORIZON}")

    # Load Model 5.0 tasks
    print("\nLoading Model 5.0 tasks...")
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

    print(f"\n✅ Loaded {len(all_tasks)} Model 5.0 tasks")

    # Model 5.0 training with advanced optimizations
    print("\n" + "=" * 80)
    print("🚀 STARTING MODEL 5.0 ULTRA TRAINING WITH OPTIMIZATIONS")
    print("=" * 80)

    # Initialize optimizers and schedulers
    meta_optimizer = torch.optim.AdamW(
        list(hybrid_model.parameters()) + list(koppen_embed.parameters()),
        lr=OUTER_LR,
        weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        meta_optimizer, T_0=10, T_mult=2, eta_min=1e-6
    )

    best_loss = float("inf")
    task_losses = []
    log_file = "./Out_Data/hybrid_maml_v5_log.csv"

    with open(log_file, "w") as f:
        f.write("epoch,meta_loss,learning_rate\n")

    for epoch in range(NUM_EPOCHS):
        start_time = time.time()

        # Adaptive task sampling based on difficulty
        if len(all_tasks) > BATCH_SIZE and task_losses:
            # Sample harder tasks more frequently
            probs = (
                np.array(task_losses) / sum(task_losses)
                if sum(task_losses) > 0
                else None
            )
            indices = np.random.choice(
                len(all_tasks), BATCH_SIZE, replace=False, p=probs
            )
            batch_tasks = [all_tasks[i] for i in indices]
        elif len(all_tasks) > BATCH_SIZE:
            indices = np.random.choice(len(all_tasks), BATCH_SIZE, replace=False)
            batch_tasks = [all_tasks[i] for i in indices]
        else:
            batch_tasks = all_tasks

        loss = meta_update_v4(
            hybrid_model, koppen_embed, batch_tasks, DEVICE, meta_optimizer
        )

        # Update task difficulties
        if len(task_losses) < len(all_tasks):
            task_losses.extend([loss] * (len(all_tasks) - len(task_losses)))
        else:
            # Update with exponential moving average
            for i in range(len(task_losses)):
                task_losses[i] = 0.9 * task_losses[i] + 0.1 * loss

        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        epoch_time = time.time() - start_time
        print(
            f"Epoch {epoch+1}/{NUM_EPOCHS} - Loss: {loss:.4f} - LR: {current_lr:.6f} - Time: {epoch_time:.1f}s"
        )

        # Log progress
        with open(log_file, "a") as f:
            f.write(f"{epoch+1},{loss},{current_lr}\n")

        # Save best model
        if loss < best_loss:
            best_loss = loss
            best_save_path = "./Out_Data/SavedModels/hybrid_maml_model_v5_best.pt"
            os.makedirs(os.path.dirname(best_save_path), exist_ok=True)
            torch.save(
                {
                    "hybrid_model_state_dict": hybrid_model.state_dict(),
                    "koppen_embed_state_dict": koppen_embed.state_dict(),
                    "meta_optimizer_state_dict": meta_optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "epoch": epoch,
                    "best_loss": best_loss,
                    "model_version": "5.0",
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
                best_save_path,
            )
            print(f"  💾 New best model saved! Loss: {best_loss:.4f}")

        # Memory cleanup every 5 epochs
        if epoch % 5 == 0:
            torch.cuda.empty_cache()

    # Save final model after all training
    save_path = "./Out_Data/SavedModels/hybrid_maml_model_v5_final.pt"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(
        {
            "hybrid_model_state_dict": hybrid_model.state_dict(),
            "koppen_embed_state_dict": koppen_embed.state_dict(),
            "meta_optimizer_state_dict": meta_optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "epoch": NUM_EPOCHS,
            "final_loss": loss,
            "best_loss": best_loss,
            "model_version": "5.0",
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

    print("\n" + "=" * 80)
    print("✅ MODEL 5.0 ULTRA TRAINING COMPLETE!")
    print("=" * 80)
    print(f"Model Version: 5.0 Ultra Scaled")
    print(f"Total Parameters: {total_params:,}")
    print(f"Best Loss: {best_loss:.4f}")
    print(f"Architecture: {HIDDEN_CHANNELS}H + {LSTM_HIDDEN_SIZE}x{LSTM_NUM_LAYERS}L")
    print(f"Sequence: {WINDOW_SIZE}→{FORECAST_HORIZON}")
    print(f"Best model saved: ./Out_Data/SavedModels/hybrid_maml_model_v5_best.pt")
    print(f"Final model saved: ./Out_Data/SavedModels/hybrid_maml_model_v5_final.pt")
    print(f"Training log: {log_file}")
    print("=" * 80)


if __name__ == "__main__":
    main()
