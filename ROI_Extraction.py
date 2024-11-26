import os
import pandas as pd
from tqdm import tqdm
import albumentations as A
import argparse

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from functions import encode_mask_to_rle
from dataset import IND2CLASS, XRayInferenceDataset
import numpy as np
import cv2

def parse_args():
    parser = argparse.ArgumentParser(description='Human Bone Image Segmentation Inference')

    parser.add_argument('--image_dir', type=str, default='/data/ephemeral/home/data/test/DCM',
                        help='추출할 이미지가 있는 디렉토리 경로')
    parser.add_argument('--model_path', type=str, default='./checkpoints/best_dice_0.9729_hr.pt',
                        help='학습된 모델 파일 경로')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='배치 크기는 1로 고정해야함')
    parser.add_argument('--threshold', type=float, default=0.4,
                        help='세그멘테이션 임계값')
    parser.add_argument('--output_path', type=str, default='roi_test.csv',
                        help='결과 저장할 CSV 파일 경로')
    parser.add_argument('--img_size', type=int, default=1024,
                        help='입력 이미지 크기')
    parser.add_argument('--x_offset', type=int, default=50,
                        help='추출할 이미지의 offset 크기')
    parser.add_argument('--y_offset', type=int, default=40,
                        help='추출할 이미지의 offset 크기')
    return parser.parse_args()


def roi_extraction(model, data_loader, thr=0.5, input_path = '', x_off = 50, y_off = 40):
    model = model.cuda()
    model.eval()

    filename_and_class = []
    bbox_results = []  # Bounding box 결과 저장

    with torch.no_grad():
        for step, (images, image_names) in tqdm(enumerate(data_loader), total=len(data_loader)):
            images = images.cuda()
            # outputs = model(images)['out']
            outputs = model(images)

            outputs = F.interpolate(outputs, size=(2048, 2048), mode="bilinear")
            outputs = torch.sigmoid(outputs)
            outputs = (outputs > thr).detach().cpu().numpy()

            max_x, max_y = 0., 0.
            min_x, min_y = float("inf"), float("inf")

            for output, image_name in zip(outputs, image_names):
                for c, segm in enumerate(output):
                    if IND2CLASS[c] in ['Trapezium', 'Trapezoid', 'Capitate', 'Hamate', 'Scaphoid', 'Lunate', 'Triquetrum', 'Pisiform']: # 손목 뼈 클래스 bbox 추출
                        ys, xs = np.where(segm)  # 마스크에서 픽셀 좌표 추출
                        if min_x > xs.min():
                            min_x = xs.min()
                        if min_y > ys.min():
                            min_y = ys.min()
                        if max_x < xs.max():
                            max_x = xs.max()
                        if max_y < ys.max():
                            max_y = ys.max()     

                # Bounding Box 좌표 확장
                min_x = max(0, min_x - x_off)  # 이미지 밖으로 나가지 않도록 0보다 작으면 0으로 설정
                min_y = max(0, min_y - y_off)
                max_x = max_x + x_off
                max_y = max_y + y_off

                # Bounding Box 부분 그리기
                # image_path = os.path.join(input_path, image_name)
                # image = cv2.imread(image_path)

                # color = (0, 255, 0)  # 초록색
                # thickness = 2  # 선 두께
                # cv2.rectangle(image, (min_x, min_y), (max_x, max_y), color, thickness)

                # # 이미지 이름 추출 및 저장
                # output_dir = '../data_roi/train/DCM'
                # _, img_name = image_name.split("\\")
                # # 이미지가 저장될 output 디렉토리 생성
                # os.makedirs(os.path.join(output_dir, _), exist_ok=True)
                
                # output_path = os.path.join(output_dir, image_name)
                # # print(output_path)
                # cv2.imwrite(output_path, image)

                bbox_results.append([min_x, min_y, max_x, max_y])
                filename_and_class.append(f"{IND2CLASS[c]}__{image_name}")
                
    return bbox_results, filename_and_class


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

    # 이미지가 저장될 output 디렉토리 생성
    os.makedirs('../data_roi/train/DCM', exist_ok=True)

    # 추론 수행
    bbox_results, filename_and_class = roi_extraction(model, test_loader, thr=args.threshold, input_path=args.image_dir, x_off=args.x_offset, y_off=args.y_offset)

    # submission 파일 생성
    classes, filename = zip(*[x.split("__") for x in filename_and_class])
    # print(filename)
    image_name = [f for f in filename]

    df = pd.DataFrame({
        "image_name": image_name,
        "bbox": bbox_results,
    })

    df.to_csv(args.output_path, index=False)

if __name__ == '__main__':
    main()
