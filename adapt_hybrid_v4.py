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

# Configuration
MODEL_PATH = "./Out_Data/SavedModels/hybrid_maml_model_v4.pt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_adaptation_data(region_coords):
    """Load 2023 and 2024 data for adaptation"""
    datasets = []
    
    # Load 2023 Q1 data
    base_path_2023 = "E:/Study/5th Sem Mini Project/Datasets/2023/Jan2Mar"
    accum_file_2023 = os.path.join(base_path_2023, "data_stream-oper_stepType-accum.nc")
    instant_file_2023 = os.path.join(base_path_2023, "data_stream-oper_stepType-instant.nc")
    
    print("Loading 2023 Q1 data for adaptation...")
    ds_accum_2023 = xr.open_dataset(accum_file_2023)
    ds_instant_2023 = xr.open_dataset(instant_file_2023)
    ds_2023 = xr.merge([ds_accum_2023, ds_instant_2023])
    
    # Load 2024 Q1 data
    base_path_2024 = "E:/Study/5th Sem Mini Project/Datasets/2024/Jan2Mar"
    accum_file_2024 = os.path.join(base_path_2024, "data_stream-oper_stepType-accum.nc")
    instant_file_2024 = os.path.join(base_path_2024, "data_stream-oper_stepType-instant.nc")
    
    print("Loading 2024 Q1 data for adaptation...")
    ds_accum_2024 = xr.open_dataset(accum_file_2024)
    ds_instant_2024 = xr.open_dataset(instant_file_2024)
    ds_2024 = xr.merge([ds_accum_2024, ds_instant_2024])
    
    # Extract region for both years
    lat_min, lat_max, lon_min, lon_max = region_coords
    
    ds_2023_reg = ds_2023.sel(
        latitude=slice(lat_max, lat_min), longitude=slice(lon_min, lon_max)
    )
    ds_2024_reg = ds_2024.sel(
        latitude=slice(lat_max, lat_min), longitude=slice(lon_min, lon_max)
    )
    
    # Combine both years
    combined_ds = xr.concat([ds_2023_reg, ds_2024_reg], dim="valid_time")
    print(f"Combined 2023+2024 data: {dict(combined_ds.sizes)}")
    
    return combined_ds


def adaptModel(region_coords, region_name):
    """Adapt Model V4 to a specific region

    Args:
        region_coords: Tuple of (lat_min, lat_max, lon_min, lon_max)
        region_name: String name for the region

    Returns:
        str: Path to saved adapted model
    """
    print("=" * 80)
    print(f"🏙️ MODEL 4.0 REGIONAL ADAPTATION: {region_name}")
    print("=" * 80)
    print(f"Device: {DEVICE}")
    print(f"Region: {region_coords}")
    print(f"Base Model: {MODEL_PATH}")
    print("=" * 80)
    # Load Model 4.0 hybrid model
    print("Loading Model 4.0 hybrid model...")
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
    config = checkpoint["config"]
    hybrid_config = checkpoint["hybrid_config"]

    print(f"✅ Model 4.0 Config loaded:")
    print(f"   Version: {checkpoint.get('model_version', '4.0')}")
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

    # Recreate Model 4.0 hybrid model
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
    print(f"✅ Model 4.0 hybrid loaded: {total_params:,} parameters")

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

    # Create train/val split for adaptation - more samples with 2 years of data
    max_samples = min(800, len(dataset))  # Increased for 2-year dataset
    train_size = int(0.8 * max_samples)

    train_indices = list(range(0, train_size))
    val_indices = list(range(train_size, max_samples))

    train_ds = Subset(dataset, train_indices)
    val_ds = Subset(dataset, val_indices)

    print(f"Adaptation training samples: {len(train_ds)}")
    print(f"Adaptation validation samples: {len(val_ds)}")

    # Fine-tuning setup for new region
    print(f"\n🏙️ FINE-TUNING MODEL 4.0 FOR NEW REGION: {region_name}")
    print("=" * 80)

    hybrid_model.train()

    # Optimizer for ALL parameters (STGCN + LSTM) - standard LR for temperate adaptation
    optimizer = torch.optim.Adam(
        hybrid_model.parameters(), lr=0.0006
    )  # Standard LR for temperate patterns
    criterion = nn.MSELoss()

    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=0)

    # Training loop - extended for 2-year adaptation
    epochs = 25  # More epochs for richer 2-year dataset
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
        print(f"Epoch {epoch+1}/{epochs}: Loss = {avg_loss:.6f}")

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

    # Save adapted Model 4.0
    save_dir = "./Out_Data/AdaptedModels"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(
        save_dir, f"hybrid_v4_adapted_{region_name}_{region_coords}.pt"
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
            "model_version": "4.0",
            "adaptation_type": "v4_regional_adaptation",
            "val_loss": avg_val_loss,
            "base_model_loss": checkpoint.get("meta_loss", "N/A"),
            "total_params": total_params,
        },
        save_path,
    )

    print("\n" + "=" * 80)
    print("✅ MODEL 4.0 REGIONAL ADAPTATION COMPLETE!")
    print("=" * 80)
    print(f"Region: {region_name}")
    print(f"Coordinates: {region_coords}")
    print(f"Model Version: 4.0")
    print(f"Model size: {total_params:,} parameters")
    print(f"Final validation loss: {avg_val_loss:.6f}")
    print(f"Model saved: {save_path}")
    print("=" * 80)

    torch.cuda.empty_cache()
    return save_path


def main():
    """Example usage"""
    # Example: Adapt to New York
    region_coords = (40, 45, 285, 290)
    region_name = "NewYork2024"

    model_path = adaptModel(region_coords, region_name)
    print(f"\n✅ Adapted model saved to: {model_path}")


if __name__ == "__main__":
    main()
