import os
import sys
import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm
import glob

from functions import encode_mask_to_rle
from dataset import CLASSES

def parse_arguments():
    """
    명령줄 인자를 파싱하는 함수
    """
    parser = argparse.ArgumentParser(description='Multi CSV Mask Fusion Script')
    parser.add_argument('--input_path', type=str, required=True,
                        help='CSV 파일들이 위치한 디렉토리 경로')
    parser.add_argument('--output_path', type=str, default='fusion_result.csv',
                        help='출력 CSV 파일 경로 (기본값: fusion_result.csv)')
    parser.add_argument('--fusion_method', type=str, default='weighted_average',
                        choices=['weighted_average', 'max', 'vote'],
                        help='마스크 융합 방법 선택')
    parser.add_argument('--priority', type=str, nargs='+',
                        help='CSV 파일 우선순위 (파일명 순서대로)')

    return parser.parse_args()

def rle_to_mask(rle, shape=(1024, 1024)):
    """
    RLE 인코딩된 마스크를 디코딩하는 함수
    """
    mask = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    if isinstance(rle, str):
        rle = [int(x) for x in rle.split()]
        for i in range(0, len(rle), 2):
            start = rle[i] - 1
            length = rle[i+1]
            mask[start:start+length] = 1

    return mask.reshape(shape)


def advanced_fusion(masks, fusion_method='weighted_average'):
    """
    여러 마스크를 융합하는 고급 Fusion 메서드

    Args:
        masks (list): 융합할 마스크들의 리스트
        fusion_method (str): 융합 방법 선택

    Returns:
        numpy.ndarray: 융합된 마스크
    """
    if fusion_method == 'weighted_average':
        # 가중 평균: 마스크 개수에 따라 동적으로 가중치 조정
        weights = [1/len(masks)] * len(masks)
        fused_mask = np.zeros_like(masks[0], dtype=np.float32)
        for mask, weight in zip(masks, weights):
            fused_mask += mask * weight
        return (fused_mask > 0.5).astype(np.uint8)

    elif fusion_method == 'max':
        # 최대값 선택
        fused_mask = np.zeros_like(masks[0], dtype=np.uint8)
        for mask in masks:
            fused_mask = np.maximum(fused_mask, mask)
        return fused_mask

    elif fusion_method == 'vote':
        # 투표 방식: 과반수 이상 활성화된 픽셀 선택
        combined_mask = np.sum(masks, axis=0)
        return (combined_mask >= len(masks)/2).astype(np.uint8)


def load_csv_files(input_path, priority=None):
    """
    지정된 경로의 모든 CSV 파일을 로드

    Args:
        input_path (str): CSV 파일들이 있는 디렉토리 경로
        priority (list, optional): CSV 파일 우선순위

    Returns:
        list: DataFrame 리스트
    """
    # CSV 파일 찾기
    csv_files = glob.glob(os.path.join(input_path, '*.csv'))

    # 우선순위가 지정된 경우 정렬
    if priority:
        csv_files = sorted(csv_files, key=lambda x:
            priority.index(os.path.basename(x))
            if os.path.basename(x) in priority
            else len(priority)
        )

    # CSV 파일 로드
    dataframes = [pd.read_csv(file) for file in csv_files]

    print("로드된 CSV 파일들:")
    for file in csv_files:
        print(f"- {file}")

    return dataframes


def process_mask_fusion(dataframes, fusion_method):
    """
    여러 데이터프레임의 마스크를 융합하는 함수

    Args:
        dataframes (list): 데이터프레임 리스트
        fusion_method (str): 융합 방법

    Returns:
        pandas.DataFrame: 최종 융합 결과 데이터프레임
    """
    # 첫 번째 데이터프레임을 기본 결과로 사용
    fusion_df = dataframes[0].copy()

    # 진행 상황 표시
    for i in tqdm(range(len(fusion_df)), desc="마스크 융합 진행"):
        image_name = fusion_df.at[i, 'image_name']
        class_label = fusion_df.at[i, 'class']

        # 동일한 이미지와 클래스의 마스크 수집
        masks = []
        mask_dataframes = []
        for df in dataframes:
            matching_rows = df[(df['image_name'] == image_name) & (df['class'] == class_label)]

            if not matching_rows.empty:
                mask = rle_to_mask(matching_rows.iloc[0]['rle'])
                masks.append(mask)
                mask_dataframes.append(df)

        # 마스크가 2개 이상인 경우 융합
        if len(masks) > 1:
            fused_mask = advanced_fusion(masks, fusion_method=fusion_method)
            fusion_df.at[i, 'rle'] = encode_mask_to_rle(fused_mask)
        # 마스크가 없는 경우 마지막 데이터프레임의 값 사용
        elif len(masks) == 0 and len(dataframes) > 1:
            last_df = dataframes[-1]
            matching_rows = last_df[(last_df['image_name'] == image_name) & (last_df['class'] == class_label)]

            if not matching_rows.empty:
                fusion_df.at[i, 'rle'] = matching_rows.iloc[0]['rle']

    return fusion_df


def main():
    """
    메인 실행 함수
    """
    # 인자 파싱
    args = parse_arguments()

    # CSV 파일 로드
    dataframes = load_csv_files(args.input_path, args.priority)

    # 마스크 융합 처리
    fusion_df = process_mask_fusion(
        dataframes,
        args.fusion_method
    )

    # 결과 저장
    fusion_df.to_csv(args.output_path, index=False)
    print(f"Fusion result saved to {args.output_path}")

if __name__ == "__main__":
    main()
