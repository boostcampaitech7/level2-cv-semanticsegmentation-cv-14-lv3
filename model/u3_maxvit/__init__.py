import torch
import torch.nn as nn
from torchvision.models.feature_extraction import create_feature_extractor


from utils.weight_init import weight_init
from .unet3plus import UNet3Plus
from timm import create_model


# MaxVit 백본 추가
maxvit_backbones = ['maxvit_base_tf_512']

class U3PMaxVitEncoder(nn.Module):
    '''
    MaxVit encoder wrapper
    '''
    def __init__(self, backbone='maxvit_base_tf_512', pretrained=False) -> None:
        super().__init__()

        maxvit = create_model(backbone, pretrained=pretrained, features_only=True, out_indices=(0, 1, 2, 3, 4))
        self.channels = [3] + [layer['num_chs'] for layer in maxvit.feature_info]

        self.backbone = maxvit

    def forward(self, x):
        out = self.backbone(x)
        return out


# UNet3+ 모델 생성 함수
def build_unet3plus(num_classes, encoder='default', skip_ch=64, aux_losses=2, use_cgm=False, pretrained=False, dropout=0.3) -> UNet3Plus:
    if encoder in maxvit_backbones:
        encoder = U3PMaxVitEncoder(backbone=encoder, pretrained=pretrained)
        transpose_final = True
        fast_up = True
    else:
        raise ValueError(f'Unsupported backbone : {encoder}')

    model = UNet3Plus(num_classes, skip_ch, aux_losses, encoder, use_cgm=use_cgm, dropout=dropout, transpose_final=transpose_final, fast_up=fast_up)
    return model
