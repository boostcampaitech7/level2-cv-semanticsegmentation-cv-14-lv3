import os
import pandas as pd
from tqdm import tqdm
import albumentations as A
import argparse
import numpy as np

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from functions import encode_mask_to_rle
from dataset import IND2CLASS, XRayInferenceDataset
from inference import load_model

def parse_args():
    parser = argparse.ArgumentParser(description='Human Bone Image Segmentation Inference')

    parser.add_argument('--image_dir', type=str, default='/data/ephemeral/home/data/test/DCM',
                        help='테스트 이미지가 있는 디렉토리 경로')
    parser.add_argument('--model_path', type=str, default='./checkpoints/fcn_resnet50_best_dice_0.0477.pt',
                        help='학습된 모델 파일 경로')
    parser.add_argument('--batch_size', type=int, default=2,
                        help='배치 크기')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='세그멘테이션 임계값')
    parser.add_argument('--output_path', type=str, default='output.csv',
                        help='결과 저장할 CSV 파일 경로')
    parser.add_argument('--img_size', type=int, default=512,
                        help='입력 이미지 크기')

    return parser.parse_args()

def create_transforms(img_size):
    """ Augmentation used in train dataset """
    return A.Compose([
        A.Resize(img_size, img_size),
        A.HorizontalFlip(p=0.5),
        A.GridDistortion(p=0.5),
        A.ElasticTransform(alpha=10.0, sigma=10.0, p=0.5),
        A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.5),
        A.Rotate(limit=45, p=0.5),
    ])

def create_tta_transforms(img_size):
    """ TTA를 위한 transform 리스트 생성 """
    tta_transforms = [
        # Original
        A.Compose([
            A.Resize(img_size, img_size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]),
        # Horizontal Flip
        A.Compose([
            A.Resize(img_size, img_size),
            A.HorizontalFlip(p=1.0),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]),
    ]
    return tta_transforms

def inverse_transform(outputs, tta_idx):
    """ Horizontal Flip에 대한 역변환 함수 """
    if tta_idx == 1:  # Horizontal Flip
        return outputs.flip(dims=(-1,))
    return outputs

def test_with_tta(model, data_loader, tta_transforms, thr=0.5):
    """ TTA를 적용한 테스트 함수 """
    model = model.cuda()
    model.eval()

    rles = []
    filename_and_class = []

    with torch.no_grad():
        for step, (images, image_names) in tqdm(enumerate(data_loader), total=len(data_loader)):
            batch_size = images.size(0)
            outputs_list = []

            # 원본과 Horizontal Flip에 대해 추론 수행
            for tta_idx, transform in enumerate(tta_transforms):
                # Transform 적용
                transformed_images = torch.stack([
                    torch.from_numpy(
                        transform(image=img.permute(1,2,0).numpy())['image']
                    ).permute(2,0,1)
                    for img in images
                ])

                # 예측
                transformed_images = transformed_images.cuda()
                outputs = model(transformed_images)['out']
                outputs = F.interpolate(outputs, size=(2048, 2048), mode="bilinear")

                # Horizontal Flip 역변환 적용
                outputs = inverse_transform(outputs, tta_idx)
                outputs_list.append(outputs)

            # 앙상블 (평균)
            outputs = torch.stack(outputs_list).mean(dim=0)
            outputs = torch.sigmoid(outputs)
            outputs = (outputs > thr).detach().cpu().numpy()

            # RLE 인코딩 및 결과 저장
            for output, image_name in zip(outputs, image_names):
                for c, segm in enumerate(output):
                    rle = encode_mask_to_rle(segm)
                    rles.append(rle)
                    filename_and_class.append(f"{IND2CLASS[c]}_{image_name}")

    return rles, filename_and_class

def main():
    args = parse_args()

    # Load model
    model = load_model(args.model_path, 29)

    # Augmentation을 포함한 transform 생성
    train_transform = create_transforms(args.img_size)

    # Dataset 및 DataLoader 설정
    test_dataset = XRayInferenceDataset(args.image_dir, transforms=train_transform)
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        drop_last=False
    )

    # TTA transforms 생성
    tta_transforms = create_tta_transforms(args.img_size)

    # TTA를 적용한 추론 수행
    rles, filename_and_class = test_with_tta(
        model=model,
        data_loader=test_loader,
        tta_transforms=tta_transforms,
        thr=args.threshold
    )

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
