import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import wandb
from wandb.integration.ultralytics import add_wandb_callback
# pip install ultralytics (8.0.238 version) 
from ultralytics import YOLO
from convert_dataset import prepare_yolo_dataset, create_dataset_yaml
import numpy as np
from trainer import set_seed

def parse_args():
    parser = argparse.ArgumentParser(description='YOLO Training for Hand Bone Segmentation')
    
    parser.add_argument('--image_dir', type=str, 
                       default='/data/ephemeral/home/data/train/DCM')
    parser.add_argument('--label_dir', type=str, 
                       default='/data/ephemeral/home/data/train/outputs_json')
    parser.add_argument('--model', type=str, 
                       default='yolov8x-seg')
    parser.add_argument('--image_size', type=int, default=1024)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=200)
    
    # K-fold Cross Validation
    parser.add_argument('--use_cv', action='store_true',
                        help='전체 fold 학습 여부')
    parser.add_argument('--n_splits', type=int, default=5)
    parser.add_argument('--fold', type=int, default=0)
    
    parser.add_argument('--project', type=str, default='Ultralytics')
    parser.add_argument('--name', type=str, default='yolov8x-seg')
    
    return parser.parse_args()

def train_fold(args, fold=None):
    if fold is not None:
        print(f"\nStarting training for fold {fold}/{args.n_splits}")
    
    # 데이터셋 준비
    dataset_path = prepare_yolo_dataset(args, fold)
    yaml_path = create_dataset_yaml(dataset_path)
    
    # 모델 생성
    model = YOLO(args.model)

    # 학습 설정
    training_args = dict(
        data=yaml_path,  
        epochs=args.epochs,
        imgsz=args.image_size,
        batch=args.batch_size,
        project=args.project,
        name=f"{args.name}_fold{fold if fold is not None else ''}",
        fliplr=0.5,
        amp=True,
        device=0,
        save=True,
        save_period=30,
        degrees=45.0
    )

    # YOLO 학습 
    results = model.train(**training_args)
    
    wandb.finish()
    return results.results_dict.get('metrics/mAP50-95(M)', 0)

def main():
    args = parse_args()
    
    # 시드 설정 
    set_seed()
    
    if args.use_cv:
        # CV 모드
        cv_scores = []
        for fold in range(args.n_splits):
            with wandb.init(project=args.project, name=f"{args.name}_fold{fold}", config=vars(args)):
                score = train_fold(args, fold)
                cv_scores.append(score)
            
        print("\nCross Validation Results:")
        print(f"Fold Scores: {cv_scores}")
        print(f"Mean mAP: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")
    else:
        # 단일 fold 모드
        run_name = f"{args.name}_fold{args.fold}" if args.fold is not None else args.name
        with wandb.init(project=args.project, name=run_name, config=vars(args)):
            train_fold(args, args.fold)

if __name__ == '__main__':
    main()