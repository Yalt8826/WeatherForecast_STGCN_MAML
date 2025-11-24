"""
Adaptive Learning Rate Scheduler for Climate-Aware Training
"""
import torch
import numpy as np

class ClimateAwareLRScheduler:
    """Learning rate scheduler that adapts based on climate zone"""
    
    def __init__(self, optimizer, region_name, base_lr=0.0006):
        self.optimizer = optimizer
        self.region_name = region_name
        self.base_lr = base_lr
        self.current_epoch = 0
        
        # Climate-specific LR multipliers (conservative)
        self.climate_multipliers = {
            "tropical": 0.9,    # Slightly lower LR for stable climates
            "temperate": 1.0,   # Standard LR
            "cold": 1.1         # Slightly higher LR for challenging climates
        }
        
        # Get climate zone
        self.climate_zone = self._get_climate_zone()
        self.lr_multiplier = self.climate_multipliers.get(self.climate_zone, 1.0)
        
    def _get_climate_zone(self):
        """Determine climate zone for region"""
        tropical_regions = ["Indonesia", "Thailand", "QueensAustralia"]
        cold_regions = ["Moscow", "NorthSiberia", "Afghanistan"]
        
        if self.region_name in tropical_regions:
            return "tropical"
        elif self.region_name in cold_regions:
            return "cold"
        else:
            return "temperate"
    
    def step(self, epoch_loss=None):
        """Update learning rate based on epoch and performance"""
        self.current_epoch += 1
        
        # Base schedule: cosine annealing with restarts
        cycle_length = 5
        cycle_progress = (self.current_epoch - 1) % cycle_length / cycle_length
        cosine_factor = 0.5 * (1 + np.cos(np.pi * cycle_progress))
        
        # Climate-aware adjustment
        climate_lr = self.base_lr * self.lr_multiplier * cosine_factor
        
        # Conservative performance-based adjustment
        if epoch_loss is not None and self.current_epoch > 3:
            if epoch_loss > 1.0:  # Very high loss
                climate_lr *= 1.1
            elif epoch_loss < 0.2:  # Very low loss
                climate_lr *= 0.95
        
        # Apply to optimizer
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = climate_lr
            
        return climate_lr
    
    def get_last_lr(self):
        """Get current learning rate"""
        return [param_group['lr'] for param_group in self.optimizer.param_groups]

def create_climate_optimizer(model_params, region_name, base_lr=0.0006):
    """Create optimizer with climate-aware settings"""
    
    # Conservative climate-specific optimizer settings
    climate_configs = {
        "tropical": {"lr": base_lr * 0.9, "weight_decay": 1e-5},
        "temperate": {"lr": base_lr, "weight_decay": 1e-4},
        "cold": {"lr": base_lr * 1.1, "weight_decay": 5e-5}
    }
    
    # Determine climate zone
    tropical_regions = ["Indonesia", "Thailand", "QueensAustralia"]
    cold_regions = ["Moscow", "NorthSiberia", "Afghanistan"]
    
    if region_name in tropical_regions:
        config = climate_configs["tropical"]
    elif region_name in cold_regions:
        config = climate_configs["cold"]
    else:
        config = climate_configs["temperate"]
    
    optimizer = torch.optim.Adam(
        model_params,
        lr=config["lr"],
        weight_decay=config["weight_decay"]
    )
    
    return optimizer, config["lr"]