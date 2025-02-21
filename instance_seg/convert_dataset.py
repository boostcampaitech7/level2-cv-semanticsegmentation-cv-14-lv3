import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm
import yaml
from dataset import CLASSES, XRayDataset

def convert_json_to_yolo(json_path, image_shape):
    """JSON 형식의 라벨을 YOLO segmentation 형식으로 변환"""
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    height, width = image_shape
    yolo_annotations = []
    
    for ann in data['annotations']:
        class_name = ann['label']
        try:
            class_id = CLASSES.index(class_name)
        except ValueError:
            continue
        
        points = np.array(ann['points'], dtype=np.float32)
        
        # polygon 형식으로 변환
        points[:, 0] = points[:, 0] / width
        points[:, 1] = points[:, 1] / height
        yolo_line = f"{class_id} " + " ".join([f"{x:.6f} {y:.6f}" for x, y in points])
        
        yolo_annotations.append(yolo_line)
    
    return yolo_annotations

def prepare_yolo_dataset(args, fold=None):
    """YOLO 형식의 데이터셋 준비"""
    # 기본 dir 설정
    dataset_root = Path(os.getcwd()) / 'yolo_datasets'
    if fold is not None:
        dataset_root = dataset_root / f'fold{fold}'
    
    print(f"Dataset directory: {dataset_root}")
    
    # 이미 디렉토리가 존재하는 경우 확인
    if dataset_root.exists() and (dataset_root/'train'/'images').exists() and (dataset_root/'val'/'images').exists():
        response = input(f"'{dataset_root}' already exists. Do you want to reuse it? (y/n): ")
        if response.lower() == 'n':
            print("Removing existing dataset...")
            import shutil
            shutil.rmtree(dataset_root)
        else:
            print("Reusing existing dataset...")
            return {
                'path': str(dataset_root.absolute()),
                'train': str(dataset_root / 'train' / 'images'),
                'val': str(dataset_root / 'val' / 'images')
            }
    
    # 새로운 데이터셋 생성
    print("Creating new dataset...")
    for split in ['train', 'val']:
        (dataset_root / split / 'images').mkdir(parents=True, exist_ok=True)
        (dataset_root / split / 'labels').mkdir(parents=True, exist_ok=True)
    
    # 데이터셋 준비 (train)
    train_dataset = XRayDataset(args.image_dir, args.label_dir, 
                               is_train=True,
                               use_cv=(fold is not None), 
                               n_splits=args.n_splits, 
                               fold=fold if fold is not None else args.fold)
    
    from sklearn.model_selection import GroupKFold
    _filenames = np.array(train_dataset.pngs)
    groups = [os.path.dirname(fname) for fname in _filenames]
    
    if fold is not None:
        gkf = GroupKFold(n_splits=args.n_splits)
        folds = list(gkf.split(_filenames, [0]*len(_filenames), groups))
        train_indices = []
        val_indices = []
        
        for i, (_, fold_indices) in enumerate(folds):
            if i == fold:
                val_indices = fold_indices  # 현재 fold -> validation
            else:
                train_indices.extend(fold_indices)  # 나머지 -> train
    else:
        train_indices = list(range(len(_filenames)))
    
    print("Preparing dataset...")
    for idx in tqdm(range(len(train_dataset.pngs))):
        image_name = train_dataset.pngs[idx]
        label_name = train_dataset.jsons[idx]
        
        # train/val 분할
        is_train = idx in train_indices
        split = 'train' if is_train else 'val'

        image_path = Path(args.image_dir) / image_name
        img = cv2.imread(str(image_path))
        if img is None:
            continue
            
        label_path = Path(args.label_dir) / label_name
        yolo_annotations = convert_json_to_yolo(label_path, img.shape[:2])
        
        new_img_path = dataset_root / split / 'images' / Path(image_name).name
        new_label_path = dataset_root / split / 'labels' / Path(image_name).with_suffix('.txt').name
        
        cv2.imwrite(str(new_img_path), img)
        with open(new_label_path, 'w') as f:
            f.write('\n'.join(yolo_annotations))
    
    return {
        'path': str(dataset_root.absolute()),
        'train': str(dataset_root / 'train' / 'images'),
        'val': str(dataset_root / 'val' / 'images')
    }

def create_dataset_yaml(dataset_paths, yaml_path='dataset.yaml'):
    """데이터셋 설정 yaml 파일 생성"""
    data = {
        'path': dataset_paths['path'],
        'train': 'train/images',
        'val': 'val/images',
        'names': {i: name for i, name in enumerate(CLASSES)},
        'nc': len(CLASSES)
    }
    
    with open(yaml_path, 'w') as f:
        yaml.dump(data, f)
    
    return yaml_path 