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
from trainer_new import train, set_seed


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
    parser.add_argument('--batch_size', type=int, default=8,
                        help='배치 크기')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='학습률')
    parser.add_argument('--max_epochs', type=int, default=20,
                        help='총 에폭 수')
    parser.add_argument('--val_interval', type=int, default=1,
                        help='검증 주기')
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Wandb 초기화
    wandb.init(
      project="bone_segmentation", 
      config = {
        "learning_rate": args.lr,
        "batch_size": args.batch_size,
        "max_epochs": args.max_epochs,
        "image_size": args.image_size,
      })
    
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    
    # 시드 고정
    set_seed()
    
    # 데이터셋 및 데이터로더 설정
    train_transform = A.Compose([A.Resize(args.image_size, args.image_size)] )
    
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
    model = models.segmentation.fcn_resnet50(pretrained=True)
    model.classifier[4] = nn.Conv2d(512, len(CLASSES), kernel_size=1)
    model = model.cuda()
    
    # 손실 함수 및 옵티마이저 설정
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-6)
    
    # 학습 시작
    train(model, train_loader, valid_loader, criterion, optimizer, args.max_epochs, args.val_interval, args.save_dir)

if __name__ == '__main__':
    main()
