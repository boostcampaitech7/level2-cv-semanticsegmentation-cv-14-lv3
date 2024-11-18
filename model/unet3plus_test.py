import numpy as np
import torch
import torch.nn as nn
from typing import List
import torch.nn.functional as F
import sys, os
import os.path as osp
sys.path.append(osp.dirname(osp.dirname(__file__)))

from utils.weight_init import weight_init

''' [To-do]
- [~] Encoder : SENet(Squeeze-and-Excitation Network)
- [ ] Decoder : Transposed Convolution -> Upsampling layer
'''
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

##### 구분선 #####
class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class U3PEncoderSE(nn.Module):
    def __init__(self, channels=[3, 64, 128, 256, 512, 1024], num_block=2):
        super().__init__()
        self.layers = nn.ModuleList()
        self.downsample_list = nn.Module()

        for ii, (ch_in, ch_out) in enumerate(zip(channels[:-1], channels[1:])):
            layer = nn.Sequential(
                u3pblock(ch_in, ch_out, num_block, down_sample=ii > 0),
                SEBlock(ch_out)  # SE Block 추가
            )
            self.layers.append(layer)

        self.channels = channels
        self.apply(weight_init)

    def forward(self, x):
        encoder_out = []
        for layer in self.layers:
            x = layer(x)
            encoder_out.append(x)
        return encoder_out
##### 구분선 #####

class UNet3Plus(nn.Module):
    def __init__(self,
                 num_classes=1,
                 skip_ch=64,
                 aux_losses=2,
                 encoder: U3PEncoderSE = None,  # SE Encoder로 변경
                 channels=[3, 64, 128, 256, 512, 1024],
                 dropout=0.3,
                 transpose_final=False,
                 use_cgm=True,
                 fast_up=True):
        super().__init__()

        self.encoder = U3PEncoderSE(channels) if encoder is None else encoder  # SE Encoder 사용
        channels = self.encoder.channels
        num_decoders = len(channels) - 1
        decoder_ch = skip_ch * num_decoders

        # 나머지 코드는 원본과 동일
        self.decoder = U3PDecoder(self.encoder.channels[1:], skip_ch=skip_ch, dropout=dropout, fast_up=fast_up)
        self.decoder.apply(weight_init)

        self.decoder = U3PDecoder(self.encoder.channels[1:], skip_ch=skip_ch, dropout=dropout, fast_up=fast_up)
        self.decoder.apply(weight_init)

        self.cls = nn.Sequential(
                    nn.Dropout(p=0.5),
                    nn.Conv2d(channels[-1], 2, 1),
                    nn.AdaptiveMaxPool2d(1),
                    nn.Sigmoid()
                ) if use_cgm else None

        if transpose_final:
            self.head = nn.Sequential(
                nn.ConvTranspose2d(decoder_ch, num_classes, kernel_size=4, stride = 2, padding=1, bias=False),
            )
        else:
            self.head = nn.Conv2d(decoder_ch, num_classes, 3, padding=1)
        self.head.apply(weight_init)

        if aux_losses > 0:
            self.aux_head = nn.ModuleDict()
            layer_indices = np.arange(num_decoders - aux_losses - 1, num_decoders - 1)
            for ii in layer_indices:
                ch = decoder_ch if ii != 0 else channels[-1]
                self.aux_head.add_module(f'aux_head{ii}', nn.Conv2d(ch, num_classes, 3, padding=1))
            self.aux_head.apply(weight_init)
        else:
            self.aux_head = None

    def forward(self, x):
        _, _, h, w = x.shape
        de_out = self.decoder(self.encoder(x))
        have_obj = 1

        pred = self.resize(self.head(de_out[-1]), h, w)

        pred = {'out': pred}
        if self.aux_head is not None:
            for ii, de in enumerate(de_out[:-1]):
                if ii == 0:
                    if self.cls is not None:
                        pred['cls'] = self.cls(de).squeeze(3).squeeze(2)  # (B,N,1,1)->(B,N)
                        have_obj = torch.argmax(pred['cls'], dim=1)
                head_key = f'aux_head{ii}'
                if head_key in self.aux_head:
                    de: torch.Tensor = de * have_obj
                    # de = self.dotProduct(de,have_obj)
                    pred[f'aux{ii}'] = self.resize(self.aux_head[head_key](de), h, w)

        return pred

    def dotProduct(self, seg, cls):
        B, N, H, W = seg.size()
        seg = seg.view(B, N, H * W)
        final = torch.einsum("ijk,ij->ijk", [seg, cls])
        final = final.view(B, N, H, W)
        return final

    def resize(self, x, h, w) -> torch.Tensor:
        _, _, xh, xw = x.shape
        if xh != h or xw != w:
            x = F.interpolate(x, size=(h, w), mode='bilinear', align_corners=True)
        return x

if __name__ == '__main__':
    input = torch.randn((2, 3, 320, 320))
    model = UNet3Plus(num_classes=7)
    out = model(input)
