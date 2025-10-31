import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data


class WeatherGraphDataset(Dataset):
    """
    Multi-variable forecasting dataset for multi-step prediction.
    Predicts all weather variables for forecast_horizon steps ahead.
    """

    def __init__(
        self,
        features,
        edge_index,
        window_size=6,
        forecast_horizon=1,
    ):
        self.features = features
        self.edge_index = edge_index
        self.window_size = window_size
        self.forecast_horizon = forecast_horizon
        self.num_weather_vars = 12
        self.num_nodes = features.shape[1]
        self.valid_indices = range(window_size, len(features) - forecast_horizon)

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        actual_idx = self.valid_indices[idx]

        # Input window: Past window_size timesteps
        start = actual_idx - self.window_size
        end = actual_idx
        window = self.features[start:end]  # [window_size, num_nodes, 24]
        x = window.reshape(self.window_size * self.num_nodes, -1)

        # Multi-step targets
        targets = []
        for h in range(1, self.forecast_horizon + 1):
            future_idx = actual_idx + h
            target = self.features[future_idx, :, : self.num_weather_vars]
            targets.append(target)
        targets = torch.stack(targets, dim=0)  # [forecast_horizon, num_nodes, 12]
        y = targets.reshape(
            self.forecast_horizon * self.num_nodes, self.num_weather_vars
        )

        return Data(
            x=x.clone().detach(),
            edge_index=self.edge_index,
            y=y.clone().detach(),
        )
