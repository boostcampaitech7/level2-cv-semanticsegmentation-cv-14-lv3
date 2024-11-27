import os
import pandas as pd
from tqdm import tqdm
import albumentations as A
import argparse

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import ttach as tta

from dataset import CLASSES, IND2CLASS, XRayInferenceDataset
from functions import encode_mask_to_rle, set_seed

def parse_args():
    parser = argparse.ArgumentParser(description='Human Bone Image Segmentation Inference')

    # 두 스크립트의 주요 파라미터 결합
    parser.add_argument('--seed', type=int, default=21, help='Random seed (default: 21)')
    parser.add_argument('--model_dir', type=str, required=True, help='Ckpt path')

    parser.add_argument('--image_dir', type=str, default='/data/ephemeral/home/data/test/DCM')
    parser.add_argument('--output_path', type=str, default='inf_output.csv')
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--threshold', type=float, default=0.5)

    # TTA와 Augmentation 옵션
    parser.add_argument('--use_tta', type=bool, default=False, help='Use Test Time Augmentation')
    parser.add_argument('--use_clahe', type=bool, default=False, help='Use CLAHE augmentation')
    parser.add_argument('--img_size', type=int, default=512)

    return parser.parse_args()

def create_transforms(img_size, use_clahe=False):
    """ Augmentation used in train dataset """
    transform_list = [
        A.Resize(img_size, img_size),
        A.HorizontalFlip(p=0.5),
        A.GridDistortion(p=0.5),
        A.ElasticTransform(alpha=10.0, sigma=10.0, p=0.5),
    ]

    # Optional CLAHE
    if use_clahe:
        transform_list.append(A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.5))

    return A.Compose(transform_list)


def create_tta_transforms():
    """ Create TTA transforms """
    return tta.Compose([
        tta.HorizontalFlip(),
    ])


def inference(args):
    # 시드 고정
    set_seed(args.seed)

    # 모델 로드 (두 번째 코드의 방식 채택)
    model = torch.load(os.path(args.model_dir))

    # 데이터셋 및 데이터로더 생성
    transforms = create_transforms(args.img_size, args.use_clahe)
    test_dataset = XRayInferenceDataset(args.image_dir, transforms=transforms)
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        drop_last=False
    )

    # TTA 설정
    if args.use_tta:
        tta_transforms = create_tta_transforms()
        model = tta.SegmentationTTAWrapper(model, tta_transforms)

    model = model.cuda()
    model.eval()

    rles = []
    filename_and_class = []

    with torch.no_grad():
        for step, (images, image_names) in tqdm(enumerate(test_loader), total=len(test_loader)):
            images = images.cuda()

            # 모델 추론
            outputs = model(images)

            # 원본 이미지 크기로 복원 및 후처리
            outputs = F.interpolate(outputs, size=(2048, 2048), mode="bilinear")
            outputs = torch.sigmoid(outputs)
            outputs = (outputs > args.threshold).detach().cpu().numpy()

            # RLE 인코딩 및 결과 저장
            for output, image_name in zip(outputs, image_names):
                for c, segm in enumerate(output):
                    rle = encode_mask_to_rle(segm)
                    rles.append(rle)
                    filename_and_class.append(f"{IND2CLASS[c]}_{image_name}")

    # CSV 파일로 결과 저장
    classes, filename = zip(*[x.split("_") for x in filename_and_class])
    image_name = [os.path.basename(f) for f in filename]

    df = pd.DataFrame({
        "image_name": image_name,
        "class": classes,
        "rle": rles,
    })
    df.to_csv(args.output_path, index=False)


def main():
    args = parse_args()
    inference(args)

if __name__ == '__main__':
    main()
