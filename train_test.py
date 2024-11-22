import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
import wandb
import albumentations as A
from mmcv.runner import HOOKS, Hook

from dataset import XRayDataset
from model.u3_effnet import UNet3Plus, build_unet3plus
from trainer import train, set_seed
from loss import get_loss

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
class TopModelSelectionHook(Hook):
    """Hook to track and select top 2 models based on performance"""
    def __init__(self, n_top_models=2):
        self.n_top_models = n_top_models
        self.top_models = []
        self.scores = []

    def after_train_epoch(self, runner):
        """Track model performance after each training"""
        # Assuming you have a validation metric available in runner
        current_metric = runner.log_buffer.output.get('val_dice', 0)

        # If we haven't collected enough top models yet
        if len(self.top_models) < self.n_top_models:
            self.top_models.append(runner.model.state_dict())
            self.scores.append(current_metric)
        else:
            # Replace the worst performing model if current is better
            min_score_idx = np.argmin(self.scores)
            if current_metric > self.scores[min_score_idx]:
                self.top_models[min_score_idx] = runner.model.state_dict()
                self.scores[min_score_idx] = current_metric


def train_fold(args, fold, hook=None):
    """Single fold training function with top model selection"""
    print(f"\nStarting training for fold {fold}/{args.n_splits}")

    # Create fold-specific save directory
    fold_save_dir = os.path.join(args.save_dir, f'fold{fold}')
    os.makedirs(fold_save_dir, exist_ok=True)

    # Transform settings
    train_transform = A.Compose([A.Resize(args.image_size, args.image_size)])

    # Dataset preparation
    train_dataset = XRayDataset(args.image_dir, args.label_dir, is_train=True,
                                transforms=train_transform, n_splits=args.n_splits, fold=fold)

    valid_dataset = XRayDataset(args.image_dir, args.label_dir, is_train=False,
                                transforms=train_transform, n_splits=args.n_splits, fold=fold)

    # DataLoader setup
    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=2, drop_last=True)

    valid_loader = DataLoader(dataset=valid_dataset, batch_size=args.batch_size,
                              shuffle=False, num_workers=2, drop_last=False)

    # Model setup
    model = build_unet3plus(num_classes=29, encoder='efficientnet-b5', pretrained=True)
    model = model.cuda()

    # Loss and optimizer
    criterion = get_loss('combined', weights={'bce': 0.5, 'dice': 0.5})
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-6)

    # Wandb initialization
    wandb.init(
        project="hand_bone_segmentation",
        name=f"{args.wandb_name}_fold{fold}",
        config={
            "learning_rate": args.lr,
            "epochs": args.max_epochs,
            "batch_size": args.batch_size,
            "image_size": args.image_size,
            "fold": fold,
            "n_splits": args.n_splits
        }
    )

    # Create top model selection hook if not provided
    if hook is None:
        hook = TopModelSelectionHook(n_top_models=2)

    # Training
    best_dice = train(
        model=model,
        data_loader=train_loader,
        val_loader=valid_loader,
        criterion=criterion,
        optimizer=optimizer,
        num_epochs=args.max_epochs,
        val_interval=args.val_interval,
        save_dir=os.path.join(args.save_dir, f'fold{fold}'),
        hook=hook  # Pass the hook to the training function
    )

    wandb.finish()

    return best_dice, hook


def main():
    args = parse_args()

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # Fix seed
    set_seed()

    # Top model selection hook
    top_model_hook = TopModelSelectionHook(n_top_models=2)

    # Cross Validation
    cv_scores = []

    if args.use_cv:  # Full fold training
        print(f"Starting {args.n_splits}-fold cross validation...")
        for fold in range(args.n_splits):
            best_dice, top_model_hook = train_fold(args, fold, hook=top_model_hook)
            cv_scores.append(best_dice)

        # Cross Validation results
        mean_dice = np.mean(cv_scores)
        std_dice = np.std(cv_scores)
        print("\nCross Validation Results:")
        print(f"Fold Scores: {cv_scores}")
        print(f"Mean Dice: {mean_dice:.4f} ± {std_dice:.4f}")

        # Save top 2 models
        for i, (model_state, score) in enumerate(zip(top_model_hook.top_models, top_model_hook.scores)):
            torch.save(model_state, os.path.join(args.save_dir, f'top_model_{i+1}_dice_{score:.4f}.pth'))
            print(f"Saved top model {i+1} with Dice score: {score:.4f}")

    else:  # Train specific fold
        print(f"Training fold {args.fold} of {args.n_splits}")
        train_fold(args, args.fold)

if __name__ == '__main__':
    main()
