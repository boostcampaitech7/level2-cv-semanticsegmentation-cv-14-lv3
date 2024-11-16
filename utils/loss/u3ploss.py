import torch
import torch.nn as nn
import torch.nn.functional as F
from .focalloss import FocalLoss
from .ms_ssimloss import MS_SSIMLoss, SSIMLoss
from .piqa_ssim import SSIM
from .iouloss import IoULoss

class U3PLoss(nn.Module):
    def __init__(self, loss_type='focal', aux_weight=0.4, process_input=True):
        super(U3PLoss, self).__init__()
        self.aux_weight = aux_weight
        self.focal_loss = FocalLoss(ignore_index=255, size_average=True)  # Binary focal loss for multi-label
        
        if loss_type == 'u3p':
            self.iou_loss = IoULoss(process_input=not process_input)
            self.ms_ssim_loss = MS_SSIMLoss(process_input=not process_input)
        elif loss_type != 'focal':
            raise ValueError(f'Unknown loss type: {loss_type}')
        
        self.loss_type = loss_type
        self.process_input = process_input

    def forward(self, preds, targets):
        if not isinstance(preds, dict):
            preds = {'final_pred': preds}
        if self.loss_type == 'focal':
            return self._forward_focal(preds, targets)
        elif self.loss_type == 'u3p':
            return self._forward_u3p(preds, targets)

    def _forward_focal(self, preds, targets):
        loss_dict = {}
        loss = self.focal_loss(preds['final_pred'], targets)
        loss_dict['head_focal_loss'] = loss.detach().item()  # for logging
        
        num_aux, aux_loss = 0, 0.
        for key in preds:
            if 'aux' in key:
                num_aux += 1
                aux_loss += self.focal_loss(preds[key], targets)
        
        if num_aux > 0:
            aux_loss = aux_loss / num_aux * self.aux_weight
            loss_dict['aux_focal_loss'] = aux_loss.detach().item()
            loss += aux_loss
            loss_dict['total_loss'] = loss.detach().item()
        
        return loss, loss_dict

    def onehot_sigmoid(self, pred, target: torch.Tensor, process_target=True):
        """
        Apply sigmoid to the predictions and process target for multi-label.
        """
        _, num_classes, h, w = pred.shape
        pred = torch.sigmoid(pred)  # Sigmoid for multi-label classification
        
        if process_target:
            target = torch.clamp(target, 0, 1)  # Ensure target values are binary
            target = target.float()  # No need for one-hot encoding as it's already binary
        return pred, target

    def _forward_u3p(self, preds, targets):
        
        # Process input predictions and targets for IoU and MS-SSIM
        if self.process_input:
            final_pred, targets = self.onehot_sigmoid(preds['final_pred'], targets)
        
        # Focal loss for final prediction
        loss, loss_dict = self._forward_focal(preds, targets)
        
        # IoU loss and MS-SSIM loss for final prediction
        iou_loss = self.iou_loss(final_pred, targets)
        msssim_loss = self.ms_ssim_loss(final_pred, targets)
        loss += iou_loss + msssim_loss
        loss_dict['head_iou_loss'] = iou_loss.detach().item()
        loss_dict['head_msssim_loss'] = msssim_loss.detach().item()

        # Handle auxiliary predictions (if any)
        num_aux, aux_iou_loss, aux_msssim_loss = 0, 0., 0.
        for key in preds:
            if 'aux' in key:
                num_aux += 1
                if self.process_input:
                    preds[key], targets = self.onehot_sigmoid(preds[key], targets, process_target=False)
                aux_iou_loss += self.iou_loss(preds[key], targets)
                aux_msssim_loss += self.ms_ssim_loss(preds[key], targets)
        
        if num_aux > 0:
            aux_iou_loss /= num_aux
            aux_msssim_loss /= num_aux
            loss_dict['aux_iou_loss'] = aux_iou_loss.detach().item()
            loss_dict['aux_msssim_loss'] = aux_msssim_loss.detach().item()
            loss += (aux_iou_loss + aux_msssim_loss) * self.aux_weight
            loss_dict['total_loss'] = loss.detach().item()
        
        return loss, loss_dict

# U3P Loss 객체 생성
def build_u3p_loss(loss_type='focal', aux_weight=0.4) -> U3PLoss:
    return U3PLoss(loss_type, aux_weight)
