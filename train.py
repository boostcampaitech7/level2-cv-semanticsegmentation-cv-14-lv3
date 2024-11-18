# trainer.py

import os
import albumentations as A
import argparse
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models
import wandb
from dataset import XRayDataset, CLASSES
import torch
import torch.nn.functional as F
from pytorch_msssim import ms_ssim

# U-Net3+
from model import build_unet3plus, UNet3Plus
''' [About gpu_trainer.py]
- gpu_trainer : Validation 연산에 GPU를 이용합니다.
- 이를 사용하기 위해 아래 주석을 해제하고, "gpu_trainer.py" 파일을 사용해주세요.
- gpu_trainer는 memory를 사용하므로, OOM(Out-of-Memory) 에러가 발생할 수 있습니다.
'''
from trainer import train, set_seed
# from gpu_trainer import train, set_seed

def parse_args():
    parser = argparse.ArgumentParser(description='Human Bone Image Segmentation Train')

    parser.add_argument('--image_dir', type=str, default='/data/ephemeral/home/data/train/DCM',
                        help='Train image가 있는 디렉토리 경로')
    parser.add_argument('--label_dir', type=str, default='/data/ephemeral/home/data/train/outputs_json',
                        help='Train label json 파일이 있는 디렉토리 경로')
    parser.add_argument('--image_size', type=int, default=512,
                        help='이미지 Resize')
    parser.add_argument('--save_dir', type=str, default='./checkpoints',
                        help='모델 저장 경로')
    parser.add_argument('--batch_size', type=int, default=2,
                        help='배치 크기')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='학습률')
    parser.add_argument('--max_epochs', type=int, default=50,
                        help='총 에폭 수')
    parser.add_argument('--val_interval', type=int, default=1,
                        help='검증 주기')
    parser.add_argument('--wandb_name', type=str, default='unet3p_1_test',
                        help='wandb에 표시될 실험 이름')

    return parser.parse_args()

def dice_loss(pred, target, smooth=1.):
    pred = pred.contiguous()
    target = target.contiguous()   
    intersection = (pred * target).sum(dim=2).sum(dim=2)
    loss = (1 - ((2. * intersection + smooth) / (pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) + smooth)))
    return loss.mean()

def IOU_loss(inputs, targets, smooth=1) :    
    inputs = inputs.reshape(-1)
    targets = targets.reshape(-1)
    intersection = (inputs * targets).sum()
    total = (inputs + targets).sum()
    union = total - intersection 
    IoU = (intersection + smooth)/(union + smooth)
    return 1 - IoU

def focal_loss(inputs, targets, alpha=0.25, gamma=2):
    bce_loss = F.binary_cross_entropy(inputs, targets, reduction='none')  # (B, C, H, W)
    pt = torch.where(targets == 1, inputs, 1 - inputs)  # (B, C, H, W)

    focal = alpha * (1 - pt) ** gamma * bce_loss

    return focal.mean()

def msssim_loss(inputs, targets, data_range=1.0):
    msssim_score = ms_ssim(inputs, targets, data_range=data_range, size_average=True)
    
    msssim_loss = 1 - msssim_score
    return msssim_loss

def calc_loss(pred, target, focal_weight=0.33, iou_weight=0.33, msssim_weight = 0.33):
    
    # focal Loss 계산
    pred = F.sigmoid(pred)
    focal = focal_loss(pred, target, alpha=0.25, gamma=2)
    
    # IoU Loss 계산
    iou = IOU_loss(pred, target)

    # msssim loss 계산
    msssim = msssim_loss(pred, target)
    
    # 가중치 기반 결합
    loss = focal_weight * focal + iou_weight * iou + msssim_weight * msssim
    return loss

def main():
    args = parse_args()

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # 시드 고정
    set_seed()

    # 데이터셋 및 데이터로더 설정
    train_transform = A.Compose([A.Resize(args.image_size, args.image_size),
                                A.ElasticTransform(
                                    alpha=10.0,
                                    sigma=10.0,
                                    alpha_affine=0.1,
                                    p=0.5),
                                    A.GridDistortion(p=0.5),
                                    A.HorizontalFlip(p=0.5)])

    train_dataset = XRayDataset(args.image_dir, args.label_dir, is_train=True, transforms=train_transform)
    valid_dataset = XRayDataset(args.image_dir, args.label_dir, is_train=False, transforms=train_transform)

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        drop_last=True
    )

    valid_loader = DataLoader(
        dataset=valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        drop_last=False
    )

    # 모델 설정
    # model = models.segmentation.fcn_resnet50(pretrained=True)
    model = build_unet3plus(num_classes=29, encoder = 'resnet50', pretrained=True)
    # model.classifier[4] = nn.Conv2d(512, len(CLASSES), kernel_size=1)
    model = model.cuda()

    # 손실 함수 및 옵티마이저 설정
    criterion = calc_loss
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-6)

    # Wandb 초기화
    wandb.init(
        project="hand_bone_segmentation",
        name=args.wandb_name,
        # wandb 초기화
        config = {
            "learning_rate": args.lr,
            "epochs": args.max_epochs,
            "batch_size": args.batch_size,
            "image_size": args.image_size,
        })

    # 학습 시작
    train(model, train_loader, valid_loader, criterion, optimizer, args.max_epochs, args.val_interval, args.save_dir)

if __name__ == '__main__':
    main()
