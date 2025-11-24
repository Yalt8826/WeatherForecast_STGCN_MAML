import torch
import torch.nn as nn
import os
import xarray as xr
from torch_geometric.loader import DataLoader
from torch.utils.data import Subset

from embed_utils import add_time_embeddings, KoppenEmbedding
from graphBuilder import build_spatial_graph
from featurePreprocessor import prepare_model_input
from dataset import WeatherGraphDataset
from model import STGCN
from hybrid_model import HybridSTGCN_LSTM
from adaptive_scheduler import create_climate_optimizer, ClimateAwareLRScheduler

# Configuration
MODEL_PATH = "./Out_Data/SavedModels/hybrid_maml_model_v5_best.pt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Configuration for multi-year data loading
YEAR = ["2023", "2024"]
DATASET_ROOT = "E:/Study/5th Sem Mini Project/Datasets"
QUARTERS = ["Jan2Mar", "Apr2Jun", "Jul2Sept", "Oct2Dec"]
NC_FILENAMES = [
    "data_stream-oper_stepType-accum.nc",
    "data_stream-oper_stepType-instant.nc",
]


def load_adaptation_data(region_coords):
    """Load all available years and quarters for adaptation"""
    lat_min, lat_max, lon_min, lon_max = region_coords

    def slice_dim(ds, dim, start, stop):
        coords = ds[dim].values
        if coords[0] > coords[-1]:
            return ds.sel({dim: slice(stop, start)})
        else:
            return ds.sel({dim: slice(start, stop)})

    all_datasets = []
    for year in YEAR:
        for quarter in QUARTERS:
            file_datasets = []
            for fname in NC_FILENAMES:
                fpath = os.path.join(DATASET_ROOT, year, quarter, fname)
                if os.path.exists(fpath):
                    print(f"Loading {year}/{quarter} - {fname}")
                    ds = xr.open_dataset(fpath)
                    ds_sel = ds.pipe(slice_dim, "latitude", lat_min, lat_max)
                    ds_sel = ds_sel.pipe(slice_dim, "longitude", lon_min, lon_max)
                    ds_sel = ds_sel.drop_vars("expver", errors="ignore")
                    file_datasets.append(ds_sel)

            if file_datasets:
                quarter_combined = xr.merge(file_datasets, compat="override")
                all_datasets.append(quarter_combined)

    combined_ds = xr.concat(all_datasets, dim="valid_time").sortby("valid_time")
    print(f"Combined multi-year data: {dict(combined_ds.sizes)}")

    return combined_ds


def adaptModel(region_coords, region_name):
    """Adapt Model V5 to a specific region

    Args:
        region_coords: Tuple of (lat_min, lat_max, lon_min, lon_max)
        region_name: String name for the region

    Returns:
        str: Path to saved adapted model
    """
    print("=" * 80)
    print(f"🏙️ MODEL 5.0 REGIONAL ADAPTATION: {region_name}")
    print("=" * 80)
    print(f"Device: {DEVICE}")
    print(f"Region: {region_coords}")
    print(f"Base Model: {MODEL_PATH}")
    print("=" * 80)
    # Load Model 5.0 hybrid model
    print("Loading Model 5.0 hybrid model...")
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
    config = checkpoint["config"]
    hybrid_config = checkpoint["hybrid_config"]

    print(f"✅ Model 5.0 Config loaded:")
    print(f"   Version: {checkpoint.get('model_version', '5.0')}")
    print(f"   Parameters: {checkpoint.get('total_params', 'N/A'):,}")
    print(f"   Window size: {config['window_size']}")
    print(f"   Forecast horizon: {config['forecast_horizon']}")
    print(f"   Hidden channels: {config['hidden_channels']}")
    print(
        f"   LSTM: {hybrid_config['lstm_hidden_size']}x{hybrid_config['lstm_num_layers']}"
    )

    # Recreate base STGCN
    base_stgcn = STGCN(
        in_channels=config["input_channels"],
        hidden_channels=config["hidden_channels"],
        out_channels=config["output_channels"],
        window_size=config["window_size"],
        forecast_horizon=config["forecast_horizon"],
        dropout_rate=0.2,
    ).to(DEVICE)

    # Recreate Model 5.0 hybrid model
    hybrid_model = HybridSTGCN_LSTM(
        base_stgcn=base_stgcn,
        lstm_hidden_size=hybrid_config["lstm_hidden_size"],
        lstm_num_layers=hybrid_config["lstm_num_layers"],
        lstm_dropout=hybrid_config["lstm_dropout"],
        out_channels=config["output_channels"],
        forecast_horizon=config["forecast_horizon"],
        freeze_base=False,  # Allow both STGCN and LSTM to be fine-tuned
    ).to(DEVICE)

    # Load pre-trained weights
    hybrid_model.load_state_dict(checkpoint["hybrid_model_state_dict"])

    koppen_embed = KoppenEmbedding(embedding_dim=8).to(DEVICE)
    koppen_embed.load_state_dict(checkpoint["koppen_embed_state_dict"])

    hybrid_model.lstm.flatten_parameters()

    total_params = sum(p.numel() for p in hybrid_model.parameters())
    print(f"✅ Model 5.0 hybrid loaded: {total_params:,} parameters")

    # Load adaptation data
    print(f"\nLoading {region_name} adaptation data...")
    ds = load_adaptation_data(region_coords)
    print(f"Adaptation data shape: {dict(ds.sizes)}")

    if "day_of_year_sin" not in ds:
        ds = add_time_embeddings(ds)

    # Prepare data
    edge_index, num_nodes, _ = build_spatial_graph(ds, k_neighbors=4)
    features, stats = prepare_model_input(ds, 0, koppen_embed, normalize=True)

    dataset = WeatherGraphDataset(
        features,
        edge_index,
        window_size=config["window_size"],
        forecast_horizon=config["forecast_horizon"],
    )

    print(f"Adaptation dataset size: {len(dataset)}")

    # Create train/val split for adaptation - more samples with multi-year data
    max_samples = min(1200, len(dataset))  # Increased for multi-year dataset
    train_size = int(0.8 * max_samples)

    train_indices = list(range(0, train_size))
    val_indices = list(range(train_size, max_samples))

    train_ds = Subset(dataset, train_indices)
    val_ds = Subset(dataset, val_indices)

    print(f"Adaptation training samples: {len(train_ds)}")
    print(f"Adaptation validation samples: {len(val_ds)}")

    # Fine-tuning setup for new region
    print(f"\n🏙️ FINE-TUNING MODEL 5.0 FOR NEW REGION: {region_name}")
    print("=" * 80)

    hybrid_model.train()

    # Climate-aware optimizer for ALL parameters (STGCN + LSTM)
    optimizer, initial_lr = create_climate_optimizer(
        hybrid_model.parameters(), region_name
    )
    
    # Climate-aware learning rate scheduler
    lr_scheduler = ClimateAwareLRScheduler(optimizer, region_name, initial_lr)
    
    print(f"   🎯 Climate-aware optimizer initialized for {region_name}")
    print(f"   Initial LR: {initial_lr:.6f}")
    criterion = nn.MSELoss()

    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=0)

    # Training loop - optimized for multi-year adaptation
    epochs = 15  # Reduced epochs for faster processing
    for epoch in range(epochs):
        epoch_losses = []

        for batch_idx, batch in enumerate(train_loader):
            if batch_idx % 25 == 0:  # Memory management
                torch.cuda.empty_cache()

            batch = batch.to(DEVICE)
            optimizer.zero_grad()

            out = hybrid_model(batch.x, batch.edge_index)
            loss = criterion(out, batch.y)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(hybrid_model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_losses.append(loss.item())

        avg_loss = sum(epoch_losses) / len(epoch_losses)
        
        # Update learning rate based on performance
        current_lr = lr_scheduler.step(avg_loss)
        
        print(f"Epoch {epoch+1}/{epochs}: Loss = {avg_loss:.6f}, LR = {current_lr:.6f}")

    # Validation
    print(f"\n🏙️ ADAPTATION VALIDATION ON {region_name}")
    print("=" * 50)

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
    print(f"Adaptation Validation MSE: {avg_val_loss:.6f}")

    # Save adapted Model 5.0
    save_dir = "./Out_Data/AdaptedModels"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(
        save_dir, f"hybrid_v5_adapted_{region_name}_{region_coords}.pt"
    )

    torch.save(
        {
            "hybrid_model_state_dict": hybrid_model.state_dict(),
            "koppen_embed_state_dict": koppen_embed.state_dict(),
            "region": region_coords,
            "region_name": region_name,
            "climate_type": "Adapted_Region",
            "stats": stats,
            "config": config,
            "hybrid_config": hybrid_config,
            "model_version": "5.0",
            "adaptation_type": "v5_regional_adaptation_adaptive",
            "val_loss": avg_val_loss,
            "base_model_loss": checkpoint.get("meta_loss", "N/A"),
            "total_params": total_params,
        },
        save_path,
    )

    print("\n" + "=" * 80)
    print("✅ MODEL 5.0 REGIONAL ADAPTATION COMPLETE!")
    print("=" * 80)
    print(f"Region: {region_name}")
    print(f"Coordinates: {region_coords}")
    print(f"Model Version: 5.0 (Adaptive)")
    print(f"Model size: {total_params:,} parameters")
    print(f"Final validation loss: {avg_val_loss:.6f}")
    print(f"Model saved: {save_path}")
    print("=" * 80)

    torch.cuda.empty_cache()
    return save_path
