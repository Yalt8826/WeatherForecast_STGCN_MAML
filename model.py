import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class STGCN(nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_channels,
        out_channels=12,
        window_size=6,
        forecast_horizon=1,
        dropout_rate=0.3,
    ):
        super(STGCN, self).__init__()
        self.window_size = window_size
        self.out_channels = out_channels
        self.forecast_horizon = forecast_horizon
        self.dropout_rate = dropout_rate

        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.conv4 = GCNConv(hidden_channels, hidden_channels)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.output_layer = nn.Linear(hidden_channels, out_channels * forecast_horizon)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.conv3(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.conv4(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)

        # Output for last window for all nodes
        num_nodes = (
            x.shape[0] // self.window_size
        )  # <-- Double check this matches WeatherGraphDataset!
        x = x[-num_nodes:]  # [num_nodes, hidden_channels]
        x = self.output_layer(x)
        x = x.view(num_nodes, self.forecast_horizon, self.out_channels)
        x = x.reshape(-1, self.out_channels)
        return x
