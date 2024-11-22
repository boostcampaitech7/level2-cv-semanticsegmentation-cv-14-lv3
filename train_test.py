import os
import albumentations as A
import argparse
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models
import wandb
from dataset import XRayDataset, CLASSES
import segmentation_models_pytorch as smp
from trainer import train, set_seed
import torch
import torch.nn.functional as F
from loss import get_loss
import numpy as np
import logging
import heapq
from mmcv.runner import Hook
from mmcv.runner import HOOKS
''' [About U-Net3+ Model]
- Duck-Net, U3+(backbone=ResNet), U3+(backbone=EfficientNet) 중에서 선택해 훈련할 수 있습니다.
- 아래 주석에서 학습하고 싶은 부분의 주석을 해체하고 훈련시켜 주세요.
- 주석을 해체하지 않는다면, 모델을 불러올 수 없어 ModuleError가 발생합니다.
'''
# from model.duck_net import build_unet3plus, build_ducknet
# from model.u3_resnet import UNet3Plus, build_unet3plus
# from model.u3_effnet import UNet3Plus, build_unet3plus

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
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='학습률')
    parser.add_argument('--max_epochs', type=int, default=100,
                        help='총 에폭 수')
    parser.add_argument('--val_interval', type=int, default=1,
                        help='검증 주기')
    parser.add_argument('--wandb_name', type=str, required=True,
                        help='wandb에 표시될 실험 이름')

    # K-fold Cross Validation
    parser.add_argument('--use_cv', action='store_true',
                        help='전체 fold 학습 여부')
    parser.add_argument('--n_splits', type=int, default=5,
                        help='K-fold에서 분할할 fold 개수')
    parser.add_argument('--fold', type=int, default=0,
                        help='특정 fold 학습시 fold 번호')

    return parser.parse_args()

@HOOKS.register_module()
class SaveBestHook(Hook):
    """Save top-3 models based on accuracy and dice metrics"""
    def __init__(self, work_dir):
        self.top3_val_dice = []  # dice metric top 3
        self.top3_val_loss = []  # loss top 3
        self.work_dir = work_dir

        # Setup logger
        log_path = os.path.join(work_dir, 'train.log')
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_path),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger()

    def save_top_model(self, metric_list, value, epoch, filename, mode='max'):
        """Save and manage top 3 models"""
        if mode == 'max':
            # Track top 3 dice scores
            heapq.heappush(metric_list, (value, epoch, filename))
            if len(metric_list) > 3:
                old_model = heapq.heappop(metric_list)
                if os.path.exists(old_model[2]):
                    os.remove(old_model[2])
        else:
            # Track top 3 losses (lower is better)
            heapq.heappush(metric_list, (-value, epoch, filename))
            if len(metric_list) > 3:
                old_model = heapq.heappop(metric_list)
                if os.path.exists(old_model[2]):
                    os.remove(old_model[2])

    def after_val_epoch(self, runner):
        # Get validation metrics
        val_dice = runner.outputs['metrics'].get('dice', 0)
        val_loss = runner.outputs['metrics'].get('loss', 0)
        epoch = runner.epoch + 1

        # Create model filenames
        dice_filename = os.path.join(self.work_dir, f"best_dice_model.pth")
        loss_filename = os.path.join(self.work_dir, f"best_loss_model.pth")

        # Save models based on dice score
        self.save_top_model(self.top3_val_dice, val_dice, epoch, dice_filename, mode='max')

        # Save models based on loss
        self.save_top_model(self.top3_val_loss, val_loss, epoch, loss_filename, mode='min')

        # Log training and validation results
        train_loss = runner.outputs.get('loss', 0)
        self.logger.info(
            f"Epoch {epoch}/{runner.max_epochs}, "
            f"Train Loss: {train_loss:.4f}, "
            f"Validation Loss: {val_loss:.4f}, "
            f"Validation Dice: {val_dice:.4f}"
        )

    def after_run(self, runner):
        self.logger.info("Training completed.")


def train_fold(args, fold):
    """단일 fold 학습 함수"""
    print(f"\nStarting training for fold {fold}/{args.n_splits}")

    # 폴드별 저장 디렉토리 생성
    fold_save_dir = os.path.join(args.save_dir, f'fold{fold}')
    os.makedirs(fold_save_dir, exist_ok=True)

    # Transform 설정
    train_transform = A.Compose([A.Resize(args.image_size, args.image_size)])

    # 데이터셋 준비
    train_dataset = XRayDataset(args.image_dir, args.label_dir, is_train=True,
                               transforms=train_transform, n_splits=args.n_splits, fold=fold)
    valid_dataset = XRayDataset(args.image_dir, args.label_dir, is_train=False,
                               transforms=train_transform, n_splits=args.n_splits, fold=fold)

    # 데이터로더 설정
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )

    valid_loader = DataLoader(
        dataset=valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    # 모델 설정
    model = build_unet3plus(num_classes=29, encoder='efficientnet-b5', pretrained=True)
    model = model.cuda()

    # 손실 함수 및 옵티마이저 설정
    criterion = get_loss('combined', weights={'bce': 0.5, 'dice': 0.5})
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-6)

    # Create runner
    runner = EpochBasedRunner(
        model=model,
        optimizer=optimizer,
        work_dir=fold_save_dir,
        max_epochs=args.max_epochs
    )

    # Add custom attributes
    runner.criterion = criterion
    runner.batch_size = args.batch_size

    # Configure hooks
    runner.register_hook(ValidationHook(valid_loader, interval=args.val_interval))
    runner.register_hook(SaveBestHook(work_dir=fold_save_dir))

    # Configure logging hooks
    log_config = dict(
        interval=50,
        hooks=[
            dict(type='TextLoggerHook'),
            dict(type='TensorboardLoggerHook')
        ]
    )
    runner.register_hook_from_cfg(log_config)

    # Start training
    runner.run([train_loader], workflow=[('train', 1)])

    # Return best dice score
    return max([x[0] for x in self.top3_val_dice]) if self.top3_val_dice else 0.0


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
