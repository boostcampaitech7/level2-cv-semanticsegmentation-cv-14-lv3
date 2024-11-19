import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0, efficientnet_b5, efficientnet_b6
from torchvision.models.feature_extraction import create_feature_extractor, get_graph_node_names

from utils.weight_init import weight_init
from .unet3plus import UNet3Plus

efficientnets = ['efficientnet-b0', 'efficientnet-b5', 'efficientnet-b6']
efficientnet_cfg = {
    'return_nodes': {
        'relu': 'layer0',
        'features.2': 'layer1',
        'features.3': 'layer2',
        'features.4': 'layer3',
        'features.5': 'layer4',
    },
    'efficientnet-b0': {
        'fe_channels': [32, 24, 40, 112, 1280],
        'channels': [16, 24, 40, 80, 128],
    },
    'efficientnet-b5': {
        'fe_channels': [48, 40, 64, 176, 2048],
        'channels': [24, 40, 64, 112, 160],
    },
    'efficientnet-b6': {
        'fe_channels': [56, 40, 72, 200, 2304],
        'channels': [32, 40, 72, 144, 192],
    }
}

##### DEVIDING LINE #####
class U3PEfficientNetEncoder(nn.Module):
    '''
    EfficientNet encoder wrapper
    '''
    def __init__(self, backbone='efficientnet-b0', pretrained=False) -> None:
        super().__init__()

        # Step 1. Select the encoder
        if backbone == 'efficientnet-b0':
            efficientnet = efficientnet_b0(pretrained=pretrained)
            cfg = efficientnet_cfg['efficientnet-b0']
        elif backbone == 'efficientnet-b5':
            efficientnet = efficientnet_b5(pretrained=pretrained)
            cfg = efficientnet_cfg['efficientnet-b5']
        elif backbone == 'efficientnet-b6':
            efficientnet = efficientnet_b6(pretrained=pretrained)
            cfg = efficientnet_cfg['efficientnet-b6']
        else:
            raise ValueError(f'Unsupported backbone : {backbone}')

        # Step 2. Check if pretrained
        if not pretrained:
            efficientnet.apply(weight_init)
        self.backbone = create_feature_extractor(efficientnet, return_nodes=efficientnet_cfg['return_nodes'])

        # Step 3. Layer to compress features to match channel sizes
        self.compress_convs = nn.ModuleList()
        for fe_ch, ch in zip(cfg['fe_channels'], cfg['channels']):
            if fe_ch != ch:
                self.compress_convs.append(nn.Conv2d(fe_ch, ch, 1, bias=False))
            else:
                self.compress_convs.append(nn.Identity())
        self.channels = [3] + cfg['channels']

    def forward(self, x):
        out = self.backbone(x)
        for ii, compress in enumerate(self.compress_convs):
            out[f'layer{ii}'] = compress(out[f'layer{ii}'])
        out = [v for _, v in out.items()]
        return out


def build_unet3plus(num_classes, encoder='default', skip_ch=64, aux_losses=2, use_cgm=False, pretrained=False, dropout=0.3) -> UNet3Plus:
    if encoder == 'default':
        encoder = None
        aux_losses = 4
        dropout = 0.0
        transpose_final = False
        fast_up = False
    elif encoder in efficientnets:
        encoder = U3PEfficientNetEncoder(backbone=encoder, pretrained=pretrained)
        transpose_final = True
        fast_up = True
    else:
        raise ValueError(f'Unsupported backbone : {encoder}')

    model = UNet3Plus(num_classes, skip_ch, aux_losses, encoder, use_cgm=use_cgm, dropout=dropout, transpose_final=transpose_final, fast_up=fast_up)
    return model
