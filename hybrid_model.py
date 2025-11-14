import torch
import torch.nn as nn
import torch.nn.functional as F
from model import STGCN

class HybridSTGCN_LSTM(nn.Module):
    """
    Hybrid model combining frozen STGCN base with trainable LSTM for regional adaptation.
    
    Architecture:
    1. Frozen STGCN extracts spatial-temporal features
    2. Trainable LSTM refines temporal patterns for specific region
    3. Output layer produces final predictions
    """
    
    def __init__(
        self,
        base_stgcn,
        lstm_hidden_size=64,
        lstm_num_layers=2,
        lstm_dropout=0.2,
        out_channels=12,
        forecast_horizon=8,
        freeze_base=True
    ):
        super(HybridSTGCN_LSTM, self).__init__()
        
        self.forecast_horizon = forecast_horizon
        self.out_channels = out_channels
        self.lstm_hidden_size = lstm_hidden_size
        
        # Frozen base STGCN model
        self.base_stgcn = base_stgcn
        if freeze_base:
            for param in self.base_stgcn.parameters():
                param.requires_grad = False
        
        # Get base model's hidden channels for LSTM input
        base_hidden_channels = self.base_stgcn.conv1.out_channels
        
        # Trainable LSTM for temporal refinement
        self.lstm = nn.LSTM(
            input_size=base_hidden_channels,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            batch_first=True,
            dropout=lstm_dropout if lstm_num_layers > 1 else 0.0,
            bidirectional=False
        )
        
        # Trainable output layer
        self.output_layer = nn.Linear(
            lstm_hidden_size, 
            out_channels * forecast_horizon
        )
        
        # Dropout for regularization
        self.dropout = nn.Dropout(lstm_dropout)
        
    def extract_base_features(self, x, edge_index):
        """Extract features from frozen STGCN base model"""
        # Get intermediate features from base STGCN (before final output layer)
        with torch.no_grad():
            # Forward through spatial layers
            h = self.base_stgcn.conv1(x, edge_index)
            h = F.relu(h)
            h = self.base_stgcn.dropout(h)
            h = self.base_stgcn.conv2(h, edge_index)
            h = F.relu(h)
            h = self.base_stgcn.dropout(h)
            h = self.base_stgcn.conv3(h, edge_index)
            h = F.relu(h)
            h = self.base_stgcn.dropout(h)
            h = self.base_stgcn.conv4(h, edge_index)
            h = F.relu(h)
            # Don't apply final dropout - we want raw features
        
        return h
    
    def forward(self, x, edge_index):
        # Extract base features from frozen STGCN
        base_features = self.extract_base_features(x, edge_index)
        
        # Reshape for temporal processing
        num_nodes = base_features.shape[0] // self.base_stgcn.window_size
        window_size = self.base_stgcn.window_size
        
        # Reshape: [window_size * num_nodes, features] -> [num_nodes, window_size, features]
        base_features = base_features.view(window_size, num_nodes, -1)
        base_features = base_features.permute(1, 0, 2)  # [num_nodes, window_size, features]
        
        # Process each node's temporal sequence through LSTM
        lstm_outputs = []
        for node_idx in range(num_nodes):
            node_sequence = base_features[node_idx:node_idx+1]  # [1, window_size, features]
            
            # LSTM forward pass
            lstm_out, (hidden, cell) = self.lstm(node_sequence)
            
            # Use last hidden state for prediction
            last_hidden = lstm_out[0, -1, :]  # [lstm_hidden_size]
            lstm_outputs.append(last_hidden)
        
        # Stack all node outputs: [num_nodes, lstm_hidden_size]
        lstm_features = torch.stack(lstm_outputs, dim=0)
        
        # Apply dropout
        lstm_features = self.dropout(lstm_features)
        
        # Generate final predictions
        predictions = self.output_layer(lstm_features)  # [num_nodes, out_channels * forecast_horizon]
        
        # Reshape to match expected output format
        predictions = predictions.view(num_nodes, self.forecast_horizon, self.out_channels)
        predictions = predictions.reshape(-1, self.out_channels)  # [num_nodes * forecast_horizon, out_channels]
        
        return predictions
    
    def get_trainable_parameters(self):
        """Get only the trainable parameters (LSTM + output layer)"""
        trainable_params = []
        trainable_params.extend(self.lstm.parameters())
        trainable_params.extend(self.output_layer.parameters())
        return trainable_params
    
    def freeze_base_model(self):
        """Freeze the base STGCN model"""
        for param in self.base_stgcn.parameters():
            param.requires_grad = False
    
    def unfreeze_base_model(self):
        """Unfreeze the base STGCN model"""
        for param in self.base_stgcn.parameters():
            param.requires_grad = True