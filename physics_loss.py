import torch
import torch.nn as nn

class PhysicsAwareLoss(nn.Module):
    def __init__(self, base_loss_weight=1.0, variation_weight=0.1, continuity_weight=0.05):
        super(PhysicsAwareLoss, self).__init__()
        self.base_loss = nn.L1Loss()
        self.base_loss_weight = base_loss_weight
        self.variation_weight = variation_weight
        self.continuity_weight = continuity_weight
    
    def forward(self, predictions, targets, num_nodes, forecast_horizon):
        # Reshape predictions and targets
        pred_reshaped = predictions.view(num_nodes, forecast_horizon, -1)
        target_reshaped = targets.view(num_nodes, forecast_horizon, -1)
        
        # Base prediction loss
        base_loss = self.base_loss(predictions, targets)
        
        # Variation penalty - encourage temporal variation
        pred_variation = torch.var(pred_reshaped[:, :, 2], dim=1)  # Temperature variation over time
        target_variation = torch.var(target_reshaped[:, :, 2], dim=1)
        variation_loss = self.base_loss(pred_variation, target_variation)
        
        # Temporal continuity - smooth transitions
        if forecast_horizon > 1:
            pred_diff = pred_reshaped[:, 1:, 2] - pred_reshaped[:, :-1, 2]
            target_diff = target_reshaped[:, 1:, 2] - target_reshaped[:, :-1, 2]
            continuity_loss = self.base_loss(pred_diff, target_diff)
        else:
            continuity_loss = torch.tensor(0.0, device=predictions.device)
        
        # Combined loss
        total_loss = (self.base_loss_weight * base_loss + 
                     self.variation_weight * variation_loss + 
                     self.continuity_weight * continuity_loss)
        
        return total_loss