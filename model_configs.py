# Model Configuration Experiments

configs = {
    "v4_current": {
        "hidden_channels": 128,
        "lstm_hidden": 64,
        "lstm_layers": 2,
        "window_size": 24,
        "forecast_horizon": 8,
        "parameters": 154304
    },
    
    "v4_medium": {
        "hidden_channels": 192,
        "lstm_hidden": 96,
        "lstm_layers": 2,
        "window_size": 36,
        "forecast_horizon": 12,
        "parameters": ~300000
    },
    
    "v4_large": {
        "hidden_channels": 256,
        "lstm_hidden": 128,
        "lstm_layers": 3,
        "window_size": 48,
        "forecast_horizon": 16,
        "parameters": ~500000
    }
}

# Test priority: v4_medium first, then v4_large if results are promising