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


''' [README]
- 아직 어떤 Augmentation 기법이 효과적인지 확인되지 않아서, TTA 기법은 최소한으로 정의했습니다.
- 앞으로 실험을 통해서 효과적인 Augmentation이 확인된다면, 내용을 추가할 예정입니다.
'''

def tta_augments():

    return [
        A.Compose([]),  # 원본 이미지
        A.Compose([A.HorizontalFlip(p=1.0)]),  # 좌우 반전
    ]


def apply_tta(model, images, tta_transforms, img_size):
    predictions = []

    # Image Tensor -> Numpy Array
    images_np = images.cpu().numpy()  # [batch_size, channels, height, width]

    for tta in tta_transforms:
        augmented_images = []

        for image in images_np:
            # [channels, height, width] -> [height, width, channels]로 변경
            image = np.transpose(image, (1, 2, 0))
            augmented_image = tta(image=image)["image"]
            # [height, width, channels] -> [channels, height, width]로 변경
            augmented_images.append(np.transpose(augmented_image, (2, 0, 1)))

        # TTA image -> Tensor
        augmented_images = torch.tensor(np.array(augmented_images)).float().cuda()

        # 모델 추론
        outputs = model(augmented_images)
        outputs = F.interpolate(outputs, size=(img_size, img_size), mode="bilinear")
        outputs = torch.sigmoid(outputs)

        # 결과 저장
        predictions.append(outputs.cpu().numpy())

    # 모든 TTA 결과를 평균
    final_output = np.mean(predictions, axis=0)
    return final_output


def test(model, data_loader, thr=0.5, img_size=2048):
    model = model.cuda()
    model.eval()

    rles = []
    filename_and_class = []

    tta_transforms = tta_augments() # TTA transform

    with torch.no_grad():
        for step, (images, image_names) in tqdm(enumerate(data_loader), total=len(data_loader)):
            images = images.cuda()

            # TTA 적용
            outputs = apply_tta(model, images, tta_transforms, img_size)
            # Threshold 적용
            outputs = (outputs > thr)

            for output, image_name in zip(outputs, image_names):
                for c, segm in enumerate(output):
                    rle = encode_mask_to_rle(segm)
                    rles.append(rle)
                    filename_and_class.append(f"{IND2CLASS[c]}_{image_name}")

    return rles, filename_and_class


def parse_args():
    parser = argparse.ArgumentParser(description='Human Bone Image Segmentation Inference')

    parser.add_argument('--image_dir', type=str, default='/data/ephemeral/home/data/test/DCM')
    parser.add_argument('--model_path', type=str, default='./checkpoints/best_dice_0.9715.pt',
                        help='Checkpoint path')
    parser.add_argument('--batch_size', type=int, default=2,
                        help='배치 크기')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Threshold for the segmentation')
    parser.add_argument('--output_path', type=str, default='./output.csv')
    parser.add_argument('--img_size', type=int, default=1024)

    return parser.parse_args()

def main():
    args = parse_args()

    # 모델 로드
    model = torch.load(args.model_path)

    # 데이터셋 및 데이터로더 설정
    tf = A.Compose([
        A.Resize(args.img_size, args.img_size),
    ])

    test_dataset = XRayInferenceDataset(args.image_dir, transforms=tf)
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        drop_last=False
    )

    # 추론 수행
    rles, filename_and_class = test(model, test_loader, thr=args.threshold, img_size=args.img_size)

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
