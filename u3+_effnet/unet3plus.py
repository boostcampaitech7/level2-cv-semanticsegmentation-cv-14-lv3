import numpy as np
import torch
import torch.nn as nn
from typing import List
import torch.nn.functional as F
import sys, os
import os.path as osp
sys.path.append(osp.dirname(osp.dirname(__file__)))

from utils.weight_init import weight_init

def autopad(k, p=None):  # kernel, padding
    # Pad to 'same'
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


def u3pblock(in_ch, out_ch, num_block=2, kernel_size=3, padding=1, down_sample=False):
    m = []
    if down_sample:
        m.append(nn.MaxPool2d(kernel_size=2))
    for _ in range(num_block):
        m += [nn.Conv2d(in_ch, out_ch, kernel_size, bias=False, padding=padding),
              nn.BatchNorm2d(out_ch),
              nn.ReLU(inplace=True)]
        in_ch = out_ch
    return nn.Sequential(*m)

def en2dec_layer(in_ch, out_ch, scale):
    m = [nn.Identity()] if scale == 1 else [nn.MaxPool2d(scale, scale, ceil_mode=True)]
    m.append(u3pblock(in_ch, out_ch, num_block=1))
    return nn.Sequential(*m)

def dec2dec_layer(in_ch, out_ch, scale, fast_up=True):
    up = [nn.Upsample(scale_factor=scale, mode='bilinear', align_corners=True) if scale != 1 else nn.Identity()]
    m = [u3pblock(in_ch, out_ch, num_block=1)]
    if fast_up:
        m = m + up
    else:
        m = up + m  # used in paper
    return nn.Sequential(*m)


class FullScaleSkipConnect(nn.Module):
    def __init__(self,
                 en_channels,   # encoder out channels, high to low
                 en_scales,
                 num_dec,       # number of decoder out
                 skip_ch=64,
                 dec_scales=None,
                 bottom_dec_ch=1024,
                 dropout=0.3,
                 fast_up=True,):

        super().__init__()
        concat_ch = skip_ch * (len(en_channels) + num_dec)

        # encoder maps to decoder maps connections
        self.en2dec_layers = nn.ModuleList()
        # print(en_scales)
        for ch, scale in zip(en_channels, en_scales):
            self.en2dec_layers.append(en2dec_layer(ch, skip_ch, scale))

        # decoder maps to decoder maps connections
        self.dec2dec_layers = nn.ModuleList()
        if dec_scales is None:
            dec_scales = []
            for ii in reversed(range(num_dec)):
                dec_scales.append(2 ** (ii + 1))
        for ii, scale in enumerate(dec_scales):
            dec_ch = bottom_dec_ch if ii == 0 else concat_ch
            self.dec2dec_layers.append(dec2dec_layer(dec_ch, skip_ch, scale, fast_up=fast_up))

        self.droupout = nn.Dropout(dropout)
        self.fuse_layer = u3pblock(concat_ch, concat_ch, 1)

    def forward(self, en_maps, dec_maps=None):
        out = []
        for en_map, layer in zip(en_maps, self.en2dec_layers):
            out.append(layer(en_map))
        if dec_maps is not None and len(dec_maps) > 0:
            for dec_map, layer in zip(dec_maps, self.dec2dec_layers):
                out.append(layer(dec_map))
        return self.fuse_layer(self.droupout(torch.cat(out, 1)))





class UNet3Plus(nn.Module):
    def __init__(self,
                 num_classes=1,
                 skip_ch=64,
                 aux_losses=2,
                 encoder: U3PEncoderDefault = None,
                 channels=[3, 64, 128, 256, 512, 1024],
                 dropout=0.3,
                 transpose_final=False,
                 use_cgm=True,
                 fast_up=True):
        super().__init__()


if __name__ == '__main__':
    input = torch.randn((2, 3, 320, 320))
    model = UNet3Plus(num_classes=7)
    out = model(input)
