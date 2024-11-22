import os
import albumentations as A
import argparse
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models
import wandb
from dataset import XRayDataset, CLASSES
# pip install segmentation-models-pytorch
import segmentation_models_pytorch as smp
from trainer import train, set_seed
import torch
import torch.nn.functional as F
from loss import get_loss
import numpy as np

''' [About U-Net3+ Model]
- Duck-Net, U3+(backbone=ResNet), U3+(backbone=EfficientNet) 중에서 선택해 훈련할 수 있습니다.
- 아래 주석에서 학습하고 싶은 부분의 주석을 해체하고 훈련시켜 주세요.
- 주석을 해체하지 않는다면, 모델을 불러올 수 없어 ModuleError가 발생합니다.
'''
# from model.duck_net import build_unet3plus, build_ducknet
# from model.u3_resnet import UNet3Plus, build_unet3plus
from model.u3_effnet import UNet3Plus, build_unet3plus

from trainer import train, set_seed
# from gpu_trainer import train, set_seed

def parse_args():
    parser = argparse.ArgumentParser(description='Human Bone Image Segmentation Train')

    parser.add_argument('--image_dir', type=str, default='/data/ephemeral/home/data/train/DCM',
                        help='Train image가 있는 디렉토리 경로')
    parser.add_argument('--label_dir', type=str, default='/data/ephemeral/home/data/train/outputs_json',
                        help='Train label json 파일이 있는 디렉토리 경로')
    parser.add_argument('--image_size', type=int, default=1024,
                        help='이미지 Resize')
    parser.add_argument('--save_dir', type=str, default='./checkpoints',
                        help='모델 저장 경로')
    parser.add_argument('--batch_size', type=int, default=2,
                        help='배치 크기')
    parser.add_argument('--max_epochs', type=int, default=100,
                        help='총 에폭 수')
    parser.add_argument('--val_interval', type=int, default=1,
                        help='검증 주기')
    parser.add_argument('--wandb_name', type=str, default='sweep',
                        help='wandb에 표시될 실험 이름')
    
    # Sweep
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='학습률')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='가중치 감쇠')
    parser.add_argument('--scheduler', type=str, default='cosine_annealing',
                        choices=['cosine_annealing'],
                        help='스케줄러')
    parser.add_argument('--bce_weight', type=float, default=0.5,
                        help='BCE loss 가중치 ')
    parser.add_argument('--num_epochs', type=int, default=100,
                        help='학습 에폭수')

    # K-fold Cross Validation
    parser.add_argument('--use_cv', action='store_true',
                        help='전체 fold 학습 여부')
    parser.add_argument('--n_splits', type=int, default=5, 
                        help='K-fold에서 분할할 fold 개수')
    parser.add_argument('--fold', type=int, default=0, 
                        help='특정 fold 학습시 fold 번호')

    return parser.parse_args()

def train_fold(args, fold):
    """단일 fold 학습 함수"""
    print(f"\nStarting training for fold {fold}/{args.n_splits}")

     # 폴드별 저장 디렉토리 생성
    fold_save_dir = os.path.join(args.save_dir, f'fold{fold}')
    os.makedirs(fold_save_dir, exist_ok=True)
    
    # Transform 설정
    train_transform = A.Compose([A.Resize(args.image_size, args.image_size),
                                 A.HorizontalFlip(p=0.5),
                                 A.GridDistortion(p=0.5),
                                 A.ElasticTransform(alpha=10.0, sigma=10.0, p=0.5),
                                 A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.5), # hrnet(x)
                                 A.Rotate(limit=45, p=0.5)]) # hrnet(x)
							                  
    val_transform = A.Compose([A.Resize(args.image_size, args.image_size)])
    
    # 데이터셋 준비
    train_dataset = XRayDataset(args.image_dir, args.label_dir, is_train=True, 
                                transforms=train_transform, n_splits=args.n_splits, fold=fold)
    
    valid_dataset = XRayDataset(args.image_dir, args.label_dir, is_train=False, 
                                transforms=val_transform, n_splits=args.n_splits, fold=fold)
    
    # 데이터로더 설정
    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2, drop_last=True)
    
    valid_loader = DataLoader(dataset=valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2, drop_last=False)

    # wandb 초기화 
    run = wandb.init(
        project="sweep",
        name=f"{args.wandb_name}_fold{fold}",
        config=args.__dict__  
    )

   # 모델 설정
    model = smp.UnetPlusPlus(
        encoder_name='tu-hrnet_w64',
        encoder_weights='imagenet',
        in_channels=3,
        classes=29
    ).cuda()
    
    # Loss function 설정
    criterion = get_loss('combined', weights={
        'bce': args.bce_weight,
        'dice': 1 - args.bce_weight
    })

    # optimizer 설정
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )

    # scheduler 설정
    if args.scheduler == 'cosine_annealing':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.num_epochs
        )

    # 학습 시작
    best_dice = train(
        model=model,
        data_loader=train_loader,
        val_loader=valid_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        num_epochs=args.num_epochs,
        val_interval=args.val_interval,
        save_dir=fold_save_dir
    )
    wandb.finish()

    return best_dice

def main():
    args = parse_args()

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # 시드 고정
    set_seed()

    # Cross Validation 실행
    cv_scores = []

    if args.use_cv:  # 전체 fold 학습
        print(f"Starting {args.n_splits}-fold cross validation...")
        for fold in range(args.n_splits):
            best_dice = train_fold(args, fold)
            cv_scores.append(best_dice)

        # Cross Validation 결과 출력
        mean_dice = np.mean(cv_scores)
        std_dice = np.std(cv_scores)
        print("\nCross Validation Results:")
        print(f"Fold Scores: {cv_scores}")
        print(f"Mean Dice: {mean_dice:.4f} ± {std_dice:.4f}")

    else:  # 특정 fold만 학습
        print(f"Training fold {args.fold} of {args.n_splits}")
        train_fold(args, args.fold)

if __name__ == '__main__':
    main()