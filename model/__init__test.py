import torch
import torch.nn as nn
from torchvision.models import efficientnet_b5
# from efficientnet_pytorch import EfficientNet

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

