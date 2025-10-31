"""
Model-Agnostic Meta-Learning (MAML) implementation.
"""
import torch
import torch.nn as nn
import torch.optim as optim

try:
    import higher
    HIGHER_AVAILABLE = True
except ImportError:
    HIGHER_AVAILABLE = False
    print("Warning: 'higher' library not found. Install with: pip install higher")


class MAMLTrainer:
    """
    MAML trainer for weather forecasting.
    """
    
    def __init__(self, model, inner_lr=0.01, outer_lr=0.001, inner_steps=5):
        """
        Args:
            model: ST-GCN model
            inner_lr: Learning rate for inner loop (task adaptation)
            outer_lr: Learning rate for outer loop (meta-update)
            inner_steps: Number of gradient steps in inner loop
        """
        self.model = model
        self.inner_lr = inner_lr
        self.outer_lr = outer_lr
        self.inner_steps = inner_steps
        
        self.meta_optimizer = optim.Adam(model.parameters(), lr=outer_lr)
        self.criterion = nn.MSELoss()
    
    def train_step(self, tasks):
        """
        Single meta-training step across all tasks.
        
        Args:
            tasks: List of WeatherTask objects
        
        Returns:
            meta_loss: Average loss across tasks
        """
        if not HIGHER_AVAILABLE:
            raise RuntimeError("'higher' library required. Install with: pip install higher")
        
        self.meta_optimizer.zero_grad()
        meta_loss = 0.0
        
        for task in tasks:
            # Create differentiable optimizer for inner loop
            with higher.innerloop_ctx(
                self.model, 
                optim.SGD(self.model.parameters(), lr=self.inner_lr),
                copy_initial_weights=False
            ) as (fmodel, diffopt):
                
                # Inner loop: adapt on support set
                for step in range(self.inner_steps):
                    for batch in task.support_loader:
                        support_loss = self.criterion(
                            fmodel(batch.x, batch.edge_index).squeeze(), 
                            batch.y
                        )
                        diffopt.step(support_loss)
                        break  # One batch per step
                
                # Outer loop: evaluate on query set
                query_loss = 0.0
                num_batches = 0
                
                for batch in task.query_loader:
                    logits = fmodel(batch.x, batch.edge_index).squeeze()
                    loss = self.criterion(logits, batch.y)
                    query_loss += loss
                    num_batches += 1
                
                query_loss = query_loss / num_batches
                meta_loss += query_loss
        
        # Average and backprop meta-loss
        meta_loss = meta_loss / len(tasks)
        meta_loss.backward()
        self.meta_optimizer.step()
        
        return meta_loss.item()
