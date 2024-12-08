import torch
import torch.nn.functional as F
from typing import Dict, Tuple
# pip install pytorch-msssim
from pytorch_msssim import ms_ssim

def dice_loss(pred: torch.Tensor, target: torch.Tensor, smooth: float = 1.0) -> torch.Tensor:
    pred = F.sigmoid(pred)
    pred = pred.view(pred.size(0), pred.size(1), -1)
    target = target.view(target.size(0), target.size(1), -1)
    
    intersection = (pred * target).sum(dim=2)
    dice = (2. * intersection + smooth) / (pred.sum(dim=2) + target.sum(dim=2) + smooth)
    return 1 - dice.mean()

def focal_loss(pred: torch.Tensor, target: torch.Tensor, alpha: float = 0.25, gamma: float = 2.0) -> torch.Tensor:
    # 클래스 불균형 문제를 해결하기 위한 Focal Loss
    pred = F.sigmoid(pred)
    bce = F.binary_cross_entropy(pred, target, reduction='none')
    pt = torch.where(target == 1, pred, 1-pred)
    focal = alpha * (1 - pt) ** gamma * bce
    return focal.mean() 

def dice_topk_loss(pred: torch.Tensor, target: torch.Tensor, k: int = 10) -> torch.Tensor:
    # 가장 성능이 낮은 k개(10) 클래스에 집중하는 Dice Loss
    pred = F.sigmoid(pred)
    pred = pred.view(pred.size(0), pred.size(1), -1)
    target = target.view(target.size(0), target.size(1), -1)
    
    intersection = (pred * target).sum(dim=2)
    dice_scores = 2. * intersection / (pred.sum(dim=2) + target.sum(dim=2) + 1.0)
    worst_scores, _ = torch.topk(dice_scores, k=min(k, pred.size(1)), largest=False, dim=1)
    return 1 - worst_scores.mean()

def tversky_loss(pred: torch.Tensor, target: torch.Tensor, alpha: float = 0.3, beta: float = 0.7) -> torch.Tensor:
    # FP와 FN에 다른 가중치를 주는 Tversky Loss
    pred = F.sigmoid(pred)
    pred = pred.view(pred.size(0), pred.size(1), -1)
    target = target.view(target.size(0), target.size(1), -1)
    
    tp = (pred * target).sum(dim=2)
    fp = (pred * (1-target)).sum(dim=2)
    fn = ((1-pred) * target).sum(dim=2)
    
    tversky = (tp + 1.0) / (tp + alpha*fp + beta*fn + 1.0)
    return (1 - tversky).mean()

def msssim_loss(pred: torch.Tensor, target: torch.Tensor, data_range: float = 1.0) -> torch.Tensor:
   # 다중 스케일에서 구조적 유사성을 측정하는 MS-SSIM Loss
    pred = F.sigmoid(pred)
    msssim_score = ms_ssim(pred, target, data_range=data_range, size_average=True)
    return 1 - msssim_score

def combined_loss(pred: torch.Tensor, target: torch.Tensor, weights: Dict[str, float]) -> torch.Tensor: 
    # 여러 Loss를 조합하여 사용
    total_loss = 0.0
    
    loss_fns = {
        'dice': dice_loss,
        'focal': focal_loss,
        'dice_topk': dice_topk_loss,
        'tversky': tversky_loss,
        'bce': lambda p, t: F.binary_cross_entropy_with_logits(p, t),
        'msssim': msssim_loss  
    }
    
    for name, weight in weights.items():
        if name in loss_fns:
            loss = loss_fns[name](pred, target)
            total_loss += weight * loss
    
    return total_loss  

def get_loss(loss_name: str, **kwargs):
    loss_registry = {
        'dice': dice_loss,
        'focal': focal_loss,
        'dice_topk': dice_topk_loss,
        'tversky': tversky_loss,
        'combined': combined_loss,
        'msssim': msssim_loss  
    }
    
    if loss_name not in loss_registry:
        raise ValueError(f"지원하지 않는 Loss 함수입니다: {loss_name}")
    
    if loss_name == 'combined':
        weights = kwargs.get('weights', {
            'dice_topk': 0.25, 
            'tversky': 0.25,  
            'dice': 0.2,       
            'focal': 0.2,     
            'msssim': 0.1       
        })
        return lambda pred, target: combined_loss(pred, target, weights)
    
    return loss_registry[loss_name]