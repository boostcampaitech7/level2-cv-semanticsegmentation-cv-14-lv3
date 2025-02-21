import os
import pandas as pd
import cv2
from tqdm import tqdm
import albumentations as A
import argparse

import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from torchvision.models.segmentation import fcn_resnet50
from functions import encode_mask_to_rle
from ultralytics import YOLO
from dataset import IND2CLASS, XRayInferenceDataset, RoiXRayInferenceDataset

def parse_args():
    parser = argparse.ArgumentParser(description='Human Bone Image Segmentation Inference')

    parser.add_argument('--image_dir', type=str, default='./data/test/DCM',
                        help='테스트 이미지가 있는 디렉토리 경로')
    parser.add_argument('--model_path', type=str, default='./checkpoints/best_model.pt',
                        help='학습된 모델 파일 경로')
    parser.add_argument('--batch_size', type=int, default=2,
                        help='배치 크기')
    parser.add_argument('--threshold', type=float, default=0.4,
                        help='세그멘테이션 임계값')
    parser.add_argument('--output_path', type=str, default='output.csv',
                        help='결과 저장할 CSV 파일 경로')
    parser.add_argument('--img_size', type=int, default=1024,
                        help='입력 이미지 크기')
    parser.add_argument('--num_classes', type=int, default=len(IND2CLASS),
                        help='클래스 개수')
    
    # ROI영역 추론
    parser.add_argument('--use_roi', action='store_true',
                        help='ROI영역 추론 진행 여부')
    parser.add_argument('--roi_csv_path', type=str, default='roi_test.csv',
                        help='추론할 데이터셋의 bbox정보가 담긴 roi.csv 파일 경로')
    # YOLO 추론
    parser.add_argument('--use_yolo', action='store_true',
                        help='YOLO 모델 사용 여부')

    return parser.parse_args()

def load_bbox_data(csv_file):
        """CSV 파일에서 bbox 데이터를 로드"""
        bbox_df = pd.read_csv(csv_file)
        bbox_dict = {}
        for idx, row in bbox_df.iterrows():
            image_name = row['image_name']
            bbox = eval(row['bbox'])  # bbox는 문자열 형태이므로 파싱
            bbox_dict[image_name] = bbox
        return bbox_dict

def test_roi(model, data_loader, csv_file, thr=0.5):
    model = model.cuda()
    model.eval()

    rles = []
    filename_and_class = []

    bbox_data = load_bbox_data(csv_file)

    with torch.no_grad():
        for step, (images, image_names) in tqdm(enumerate(data_loader), total=len(data_loader)):
            images = images.cuda()
            # outputs = model(images)['out']
            outputs = model(images)
            image_name = image_names[0]

            # bbox 크기 확인
            if image_name in bbox_data:
                bbox = bbox_data[image_name]  # CSV에서 가져온 bbox
                x_min, y_min, x_max, y_max = bbox
                width = x_max - x_min
                height = y_max - y_min

            outputs = F.interpolate(outputs, size=(height, width), mode="bilinear")
            outputs = torch.sigmoid(outputs)
            outputs = (outputs > thr).detach().cpu().numpy()

            for output, image_name in zip(outputs, image_names):
                for c, segm in enumerate(output):
                    final_output = np.zeros((2048, 2048), dtype=float)
                    final_output[y_min:y_max, x_min:x_max] = segm

                    rle = encode_mask_to_rle(final_output)
                    rles.append(rle)
                    filename_and_class.append(f"{IND2CLASS[c]}_{image_name}")

    return rles, filename_and_class

def test_yolo(model, data_loader, thr=0.4):
    model.to('cuda')
    
    rles = []
    filename_and_class = []
    
    for images, image_names in tqdm(data_loader):
        results = model.predict(source=images, device='cuda')
        
        for result, image_name in zip(results, image_names):
            masks = result.masks  # 예측된 마스크 정보
            boxes = result.boxes # 예측된 바운딩 박스 정보
            
            if masks is None or boxes is None:
                for class_id in range(len(IND2CLASS)):
                    rles.append('')
                    filename_and_class.append(f"{IND2CLASS[class_id]}_{image_name}")
                continue
            
            class_masks = np.zeros((len(IND2CLASS), 2048, 2048), dtype=np.uint8)
            
            for mask, cls in zip(masks.data, boxes.cls):
                mask = cv2.resize(
                    mask.cpu().numpy(),
                    (2048, 2048),
                    interpolation=cv2.INTER_LINEAR
                )
                mask = (mask > thr).astype(np.uint8)
                class_id = int(cls.item())
                class_masks[class_id] = np.logical_or(class_masks[class_id], mask)
            
            for class_id, mask in enumerate(class_masks):
                rle = encode_mask_to_rle(mask)
                rles.append(rle)
                filename_and_class.append(f"{IND2CLASS[class_id]}_{image_name}")
                    
    return rles, filename_and_class

def test(model, data_loader, thr=0.5):
    model = model.cuda()
    model.eval()

    rles = []
    filename_and_class = []

    with torch.no_grad():
        for step, (images, image_names) in tqdm(enumerate(data_loader), total=len(data_loader)):
            images = images.cuda()
            outputs = model(images)['out']

            outputs = F.interpolate(outputs, size=(2048, 2048), mode="bilinear")
            outputs = torch.sigmoid(outputs)
            outputs = (outputs > thr).detach().cpu().numpy()

            for output, image_name in zip(outputs, image_names):
                for c, segm in enumerate(output):
                    rle = encode_mask_to_rle(segm)
                    rles.append(rle)
                    filename_and_class.append(f"{IND2CLASS[c]}_{image_name}")

    return rles, filename_and_class

def load_model(model_path, num_classes=29):
    # 모델 아키텍처 초기화
    model = fcn_resnet50(weights=None)
    model.classifier[4] = torch.nn.Conv2d(512, num_classes, kernel_size=(1, 1), stride=(1, 1))

    # 저장된 모델 로드
    print(f"Loading model from {model_path}")
    checkpoint = torch.load(model_path)

    # 체크포인트 구조 확인
    if isinstance(checkpoint, dict):
        print("Checkpoint keys:", checkpoint.keys())
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint

    # state_dict 키 확인
    print("\nFirst few state_dict keys:", list(state_dict.keys())[:5])
    print("\nModel state_dict keys:", list(model.state_dict().keys())[:5])

    # 키 불일치 확인
    model_keys = set(model.state_dict().keys())
    state_dict_keys = set(state_dict.keys())
    missing_keys = model_keys - state_dict_keys
    unexpected_keys = state_dict_keys - model_keys

    if missing_keys:
        print("\nMissing keys:", missing_keys)
    if unexpected_keys:
        print("\nUnexpected keys:", unexpected_keys)

    # state_dict 키 수정이 필요한 경우
    if any(key.startswith('module.') for key in state_dict.keys()):
        print("\nRemoving 'module.' prefix from state_dict keys")
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

    # strict=False로 로드 시도
    model.load_state_dict(state_dict, strict=False)
    print("\nModel loaded successfully")

    return model

def main():
    args = parse_args()

    # YOLO 모델 사용 시
    if args.use_yolo:
        model = YOLO(args.model_path)
    else:
        model = load_model(args.model_path, args.num_classes)

    # 데이터셋 및 데이터로더 설정
    tf = A.Compose([
        A.Resize(args.img_size, args.img_size),
    ])
    
    if args.use_roi:
        test_dataset = RoiXRayInferenceDataset(args.image_dir, args.roi_csv_path, transforms=tf)
        test_loader = DataLoader(
            dataset=test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=2,
            drop_last=False
        )
        # 추론 수행
        rles, filename_and_class = test_roi(model, test_loader, args.roi_csv_path, thr=args.threshold)
    elif args.use_yolo:
        test_dataset = XRayInferenceDataset(args.image_dir, transforms=tf)
        test_loader = DataLoader(
            dataset=test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=2,
            drop_last=False
        )
        # yolo 추론 수행
        rles, filename_and_class = test_yolo(model, test_loader, thr=args.threshold)
    else:
        test_dataset = XRayInferenceDataset(args.image_dir, args.roi_csv_path, transforms=tf)
        test_loader = DataLoader(
            dataset=test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=2,
            drop_last=False
        )
        # 추론 수행
        rles, filename_and_class = test(model, test_loader, thr=args.threshold)

    # submission 파일 생성
    classes, filename = zip(*[x.split("_") for x in filename_and_class])
    image_name = [os.path.basename(f) for f in filename]

    df = pd.DataFrame({
        "image_name": image_name,
        "class": classes,
        "rle": rles,
    })

    df.to_csv(args.output_path, index=False)

if __name__ == '__main__':
    main()
