import torch.nn as nn
import torch.nn.functional as F
import torch 

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.5, gamma=2.0, size_average=True, ignore_index=255):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.size_average = size_average

    def forward(self, inputs, targets):
        """
        Focal Loss for multi-label classification
        Args:
            inputs (torch.Tensor): model's predicted logits, shape (batch_size, num_classes, H, W)
            targets (torch.Tensor): ground truth labels, shape (batch_size, num_classes, H, W)
        Returns:
            torch.Tensor: computed focal loss value
        """
        # Apply sigmoid to get probabilities for each class (for multi-label classification)
        inputs = torch.sigmoid(inputs)

        # Clamp inputs to avoid numerical issues with log operations
        eps = 1e-3
        inputs = torch.clamp(inputs, min=eps, max=1.0 - eps)

        # Calculate BCE loss
        ce_loss = -targets * torch.log(inputs) - (1 - targets) * torch.log(1 - inputs)

        # Check for NaN or Inf in ce_loss
        if torch.isnan(ce_loss).any() or torch.isinf(ce_loss).any():
            print(f"NaN or Inf detected in ce_loss: min={ce_loss.min().item()}, max={ce_loss.max().item()}")
            print(f"After clamping Inputs: min={inputs.min().item()}, max={inputs.max().item()}")
            print(f"Targets: min={targets.min().item()}, max={targets.max().item()}")
            ce_loss = torch.nan_to_num(ce_loss, nan=0.0, posinf=1.0, neginf=0.0)


        # Compute the focal loss
        pt = inputs  # Probability predictions
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        # Check for NaN or Inf in focal_loss
        if torch.isnan(focal_loss).any() or torch.isinf(focal_loss).any():
            print(f"NaN or Inf detected in focal_loss: {focal_loss}")
            focal_loss = torch.nan_to_num(focal_loss, nan=0.0, posinf=1.0, neginf=0.0)

        # If we want to reduce the loss
        if self.size_average:
            return focal_loss.mean()
        else:
            return focal_loss.sum()

