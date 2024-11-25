import os
import pandas as pd
from tqdm import tqdm
import albumentations as A
import argparse

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.models.segmentation import fcn_resnet50
from functions import encode_mask_to_rle
from dataset import IND2CLASS, XRayInferenceDataset

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

    # 모델 로드
    model = load_model(args.model_path, 29)

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
