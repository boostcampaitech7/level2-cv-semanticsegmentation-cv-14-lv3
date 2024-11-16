import torch
import torch.nn as nn
import torch.nn.functional as F

def binary_iou_loss(pred, target):
    """
    Calculate the binary IoU loss for a single class.
    Args:
        pred (torch.Tensor): Predicted probabilities for a single class, shape (batch_size, H, W).
        target (torch.Tensor): Ground truth binary labels for a single class, shape (batch_size, H, W).
    Returns:
        torch.Tensor: Binary IoU loss for the given class.
    """
    Iand = torch.sum(pred * target, dim=(1, 2))  # Intersection
    Ior = torch.sum(pred, dim=(1, 2)) + torch.sum(target, dim=(1, 2)) - Iand  # Union
    IoU = Iand / (Ior + 1e-6)  # Add epsilon to avoid division by zero
    return 1 - IoU  # IoU loss


class IoULoss(nn.Module):
    """
    Multi-label IoU loss for segmentation tasks.
    """
    def __init__(self, process_input=True):
        """
        Args:
            process_input (bool): Whether to apply sigmoid to the predictions.
        """
        super().__init__()
        self.process_input = process_input

    def forward(self, pred, target):
        """
        Forward pass for IoU loss.
        Args:
            pred (torch.Tensor): Predicted logits, shape (batch_size, num_classes, H, W).
            target (torch.Tensor): Ground truth binary labels, shape (batch_size, num_classes, H, W).
        Returns:
            torch.Tensor: Average IoU loss across all classes.
        """
        if self.process_input:
            pred = torch.sigmoid(pred)  # Apply sigmoid for multi-label predictions

        total_loss = 0
        num_classes = pred.shape[1]

        for i in range(num_classes):
            loss = binary_iou_loss(pred[:, i], target[:, i])
            total_loss += loss.mean()  # Average loss for the current class

        return total_loss / num_classes  # Average across all classes


if __name__ == '__main__':
    # Example usage
    pred = torch.randn((6, 3, 50, 50))  # Predicted logits for 3 classes
    target = torch.randint(0, 2, (6, 3, 50, 50))  # Binary ground truth labels
    iou = IoULoss()
    rst = iou(pred, target)
    print(rst)