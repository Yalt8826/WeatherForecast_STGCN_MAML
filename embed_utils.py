# embed_utils.py

import torch
import torch.nn as nn
import xarray as xr
import numpy as np
import pandas as pd


def add_time_embeddings(ds: xr.Dataset) -> xr.Dataset:
    time_dim = "time" if "time" in ds.dims else "valid_time"
    times = pd.to_datetime(ds[time_dim].values)
    day_of_year = times.dayofyear.values
    time_of_day = (
        times.hour.values + times.minute.values / 60 + times.second.values / 3600
    )

    year_progress = 2 * np.pi * day_of_year / 365.25
    day_progress = 2 * np.pi * time_of_day / 24.0

    ds = ds.assign(
        year_progress_sin=(time_dim, np.sin(year_progress)),
        year_progress_cos=(time_dim, np.cos(year_progress)),
        day_progress_sin=(time_dim, np.sin(day_progress)),
        day_progress_cos=(time_dim, np.cos(day_progress)),
    )
    return ds


class KoppenEmbedding(nn.Module):
    def __init__(self, embedding_dim=8):
        super(KoppenEmbedding, self).__init__()
        self.num_classes = 31  # indices 0-30, where 0 is unused/padding
        self.embedding_dim = embedding_dim
        self.embedding = nn.Embedding(self.num_classes, embedding_dim)

    def forward(self, koppen_codes):
        return self.embedding(koppen_codes)
