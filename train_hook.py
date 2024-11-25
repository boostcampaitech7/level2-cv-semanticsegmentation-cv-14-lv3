import os
import logging
import heapq
import wandb
import numpy as np
import argparse
import albumentations as A

# Library about torch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torchvision import models
import segmentation_models_pytorch as smp

from dataset import XRayDataset, CLASSES
from loss import get_loss
from trainer_hook import set_seed

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
    parser.add_argument('--wandb_name', type=str, default='sweep-HOOK',
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
                        help='BCE loss 가중치')
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

class SaveTopModelsHook:
    """Hook to save and track top performing models based on dice score"""
    def __init__(self, save_dir, n_top_models=2):
        self.save_dir = save_dir
        self.n_top_models = n_top_models
        self.top_models = []  # dice score 기준 상위 모델을 저장할 리스트

        # Logger 설정
        log_path = os.path.join(save_dir, 'train.log')
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_path),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger()

    def save_top_model(self, dice, epoch, model_state):
        """상위 모델을 저장하고 관리하는 함수"""
        filename = os.path.join(
            self.save_dir,
            f'model_dice_{dice:.4f}_epoch_{epoch}.pth'
        )

        # dice score가 높은 순으로 저장 (음수로 변환하여 최소 힙을 최대 힙처럼 사용)
        heapq.heappush(self.top_models, (-dice, epoch, filename, model_state))

        # 지정된 개수 이상의 모델이 저장되면 가장 낮은 성능의 모델 삭제
        if len(self.top_models) > self.n_top_models:
            _, _, old_filename, _ = heapq.heappop(self.top_models)
            if os.path.exists(old_filename):
                os.remove(old_filename)

        # 새 모델 저장
        torch.save(model_state, filename)
        self.logger.info(f"Saved model with dice score: {dice:.4f} at epoch {epoch}")


def train_fold(args, fold):
    """Single fold training function with enhanced model selection"""
    print(f"\nStarting training for fold {fold}/{args.n_splits}")

    # Create fold-specific save directory
    fold_save_dir = os.path.join(args.save_dir, f'fold{fold}')
    os.makedirs(fold_save_dir, exist_ok=True)

    # Initialize hook
    hook = SaveTopModelsHook(fold_save_dir, n_top_models=2)

    # Transform settings
    train_transform = A.Compose([
        A.Resize(args.image_size, args.image_size),
        A.HorizontalFlip(p=0.5),
        A.GridDistortion(p=0.5),
        A.ElasticTransform(alpha=10.0, sigma=10.0, p=0.5),
        A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.5),
        A.Rotate(limit=45, p=0.5)
    ])
    val_transform = A.Compose([
        A.Resize(args.image_size, args.image_size)
    ])

    # Dataset and DataLoader setup
    train_dataset = XRayDataset(
        args.image_dir, args.label_dir,
        is_train=True, transforms=train_transform,
        n_splits=args.n_splits, fold=fold
    )
    valid_dataset = XRayDataset(
        args.image_dir, args.label_dir,
        is_train=False, transforms=val_transform,
        n_splits=args.n_splits, fold=fold
    )

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

    # wandb initialization
    run = wandb.init(
        project="sweep",
        name=f"{args.wandb_name}_fold{fold}",
        config=args.__dict__
    )

    # Model setup
    model = smp.UnetPlusPlus(
        encoder_name='tu-hrnet_w64',
        encoder_weights='imagenet',
        in_channels=3,
        classes=29
    ).cuda()

    # Loss, optimizer and scheduler setup
    criterion = get_loss('combined', weights={
        'bce': args.bce_weight,
        'dice': 1 - args.bce_weight
    })

    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )

    if args.scheduler == 'cosine_annealing':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.num_epochs
        )

    # Training
    for epoch in range(args.num_epochs):
        model.train()
        train_loss = 0
        train_dice = 0

        for batch_idx, (images, masks) in enumerate(train_loader):
            images = images.cuda()
            masks = masks.cuda()

            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, masks)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        # Validation
        if (epoch + 1) % args.val_interval == 0:
            model.eval()
            val_loss = 0
            val_dice = 0

            with torch.no_grad():
                for images, masks in valid_loader:
                    images = images.cuda()
                    masks = masks.cuda()

                    outputs = model(images)
                    loss = criterion(outputs, masks)

                    val_loss += loss.item()
                    # Calculate dice score
                    pred_masks = (outputs > 0.5).float()
                    dice = (2 * (pred_masks * masks).sum()) / (pred_masks.sum() + masks.sum() + 1e-8)
                    val_dice += dice.item()

            val_loss /= len(valid_loader)
            val_dice /= len(valid_loader)

            # Log metrics
            metrics = {
                'train_loss': train_loss / len(train_loader),
                'val_loss': val_loss,
                'val_dice': val_dice,
                'epoch': epoch + 1
            }
            wandb.log(metrics)

            # Save top models based on dice score
            hook.save_top_model(
                val_dice,
                epoch + 1,
                model.state_dict()
            )

            print(f"Epoch [{epoch+1}/{args.num_epochs}] - "
                  f"Train Loss: {train_loss/len(train_loader):.4f}, "
                  f"Val Loss: {val_loss:.4f}, "
                  f"Val Dice: {val_dice:.4f}")

        scheduler.step()

    wandb.finish()
    return hook.top_models


def main():
    args = parse_args()

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    set_seed()

    if args.use_cv:
        cv_scores = []
        all_top_models = []

        for fold in range(args.n_splits):
            top_models = train_fold(args, fold)
            cv_scores.extend([-m[0] for m in top_models])  # 음수를 다시 양수로 변환
            all_top_models.extend(top_models)

        # Print final results
        mean_dice = np.mean(cv_scores)
        std_dice = np.std(cv_scores)
        print(f"\nCross Validation Results:")
        print(f"Mean Dice: {mean_dice:.4f} ± {std_dice:.4f}")

        # Save absolute best models
        sorted_models = sorted(all_top_models)[:2]  # 상위 2개 모델

        for i, (neg_dice, epoch, filename, state_dict) in enumerate(sorted_models):
            save_path = os.path.join(
                args.save_dir,
                f'best_dice_model_{i+1}.pth'
            )
            torch.save(state_dict, save_path)
            print(f"Saved top dice model {i+1} with score: {-neg_dice:.4f}")

    else:
        train_fold(args, args.fold)

if __name__ == '__main__':
    main()
