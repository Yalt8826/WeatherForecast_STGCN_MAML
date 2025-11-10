import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class TemporalSTGCN(nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_channels,
        out_channels=12,
        window_size=6,
        forecast_horizon=1,
        dropout_rate=0.3,
    ):
        super(TemporalSTGCN, self).__init__()
        self.window_size = window_size
        self.out_channels = out_channels
        self.forecast_horizon = forecast_horizon
        self.dropout_rate = dropout_rate

        # Spatial layers
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        
        # Temporal layers - LSTM for sequence modeling
        self.lstm = nn.LSTM(
            input_size=hidden_channels,
            hidden_size=hidden_channels,
            num_layers=2,
            batch_first=True,
            dropout=dropout_rate
        )
        
        self.dropout = nn.Dropout(p=dropout_rate)
        
        # Output layers for multi-step prediction
        self.output_layers = nn.ModuleList([
            nn.Linear(hidden_channels, out_channels) 
            for _ in range(forecast_horizon)
        ])

    def forward(self, x, edge_index):
        num_nodes = x.shape[0] // self.window_size
        
        # Reshape to process each timestep
        x = x.view(self.window_size, num_nodes, -1)
        
        # Apply GCN to each timestep
        temporal_features = []
        for t in range(self.window_size):
            # Spatial processing
            h = self.conv1(x[t], edge_index)
            h = F.relu(h)
            h = self.dropout(h)
            h = self.conv2(h, edge_index)
            h = F.relu(h)
            h = self.dropout(h)
            temporal_features.append(h)
        
        # Stack temporal features: [window_size, num_nodes, hidden_channels]
        temporal_features = torch.stack(temporal_features, dim=0)
        
        # Process each node's temporal sequence
        outputs = []
        for node in range(num_nodes):
            node_sequence = temporal_features[:, node, :]  # [window_size, hidden_channels]
            node_sequence = node_sequence.unsqueeze(0)  # [1, window_size, hidden_channels]
            
            # LSTM processing
            lstm_out, _ = self.lstm(node_sequence)
            last_hidden = lstm_out[0, -1, :]  # [hidden_channels]
            
            # Multi-step prediction
            node_predictions = []
            for step in range(self.forecast_horizon):
                pred = self.output_layers[step](last_hidden)
                node_predictions.append(pred)
            
            outputs.append(torch.stack(node_predictions, dim=0))
        
        # Combine all nodes: [num_nodes, forecast_horizon, out_channels]
        outputs = torch.stack(outputs, dim=0)
        
        # Reshape to match expected output format
        outputs = outputs.view(-1, self.out_channels)
        
        return outputs