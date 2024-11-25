import sys
import numpy as np
import pandas as pd
import copy
import argparse
from tqdm import tqdm
from typing import List, Dict, Set, Tuple, Iterator
import warnings
from dataclasses import dataclass
from pathlib import Path

sys.path.append("/data/ephemeral/home/ng-youn")
from functions import encode_mask_to_rle, decode_rle_to_mask
from dataset import CLASSES
'''
- Method : Hoft Voting ensemble
- 가중치 반영 없이, 5개의 csv file에서 3(=threshold)개의 csv file 이상에서 예측된 값을 결과로 저장합니다.
- chunk : Test image를 불러오는 과정에서 OOM 문제를 방지하기 위해서 추가한 기능입니다. bach_size의 역할을 합니다.
- 이 코드를 실행하기 위해서는 아래와 같은 파일 구조가 필요합니다.
    ng-youn (User name)
    ├─ output : inference.py를 사용해 만들어진 output.csv들이 위치한 폴더입니다.
    │  ├─ 9741.csv
    │  ├─ 9733.csv
    │  .
    │  └─ 9744.csv
    ├─ tools
    │   └─ ensemble
    │    ├─ 2class_ensemble.py
    .    ├─ hard_ensemble.py
    .    └─ weight_ensemble.py  
    ├─ functions.py
    └─ dataset.py
'''
def parse_args():
    parser = argparse.ArgumentParser(description='Hard Voting Ensemble')
    parser.add_argument('--input_dir', type=str, default='/data/ephemeral/home/ng-youn/output',
                        help='output.csv들이 위치한 폴더의 경로를 입력해주세요.')
    parser.add_argument('--image_dir', type=str, default='/data/ephemeral/home/data/test/DCM',
                        help='Test image의 경로')
    parser.add_argument('--threshold', type=float, default=0.6)
    parser.add_argument('--chunk_size', type=int, default=10)  # Reduced chunk size
    parser.add_argument('--class_chunk_size', type=int, default=2)  # Process classes in chunks

    return parser.parse_args()

@dataclass
class EnsembleConfig:
    input_dir: str
    image_dir: str
    threshold: float
    height: int = 2048
    width: int = 2048
    chunk_size: int = 10
    class_chunk_size: int = 2

def check_paths(config: EnsembleConfig) -> None:
    """올바른 경로가 입력되었는지 확인합니다."""
    if not Path(config.input_dir).exists():
        raise FileNotFoundError(f"Input directory does not exist: {config.input_dir}")
    if not Path(config.image_dir).exists():
        raise FileNotFoundError(f"Image directory does not exist: {config.image_dir}")

def load_csv_files(input_dir: str) -> List[pd.DataFrame]:
    """CSV 파일들을 로드하되, 필요한 컬럼만 메모리에 유지합니다."""
    outputs = []
    csv_files = list(Path(input_dir).glob("*.csv"))

    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {input_dir}")

    for file_path in tqdm(csv_files, desc="Loading CSV files"):
        try:
            # 필요한 컬럼만 로드
            df = pd.read_csv(file_path, usecols=['image_name', 'class', 'rle'])
            outputs.append(df)
        except Exception as e:
            raise ValueError(f"Error loading {file_path}: {e}")

    return outputs

def get_chunks(lst: List, n: int) -> Iterator[List]:
    """리스트를 n 크기의 청크로 나눕니다."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def process_single_prediction(
    row: pd.Series,
    height: int,
    width: int
) -> np.ndarray:
    """단일 예측을 처리합니다."""
    if pd.isna(row['rle']):
        return np.zeros((height, width), dtype=np.float32)
    try:
        return decode_rle_to_mask(row['rle'], height, width)
    except Exception as e:
        warnings.warn(f"Error processing {row['image_name']}, class {row['class']}: {e}")
        return np.zeros((height, width), dtype=np.float32)

def process_chunk(
    image_names: List[str],
    class_names: List[str],
    dfs: List[pd.DataFrame],
    config: EnsembleConfig
) -> pd.DataFrame:
    """이미지와 클래스의 청크를 처리합니다."""
    predictions = []
    num_models = len(dfs)

    for img_name in tqdm(image_names, desc="Processing images", leave=False):
        for class_name in class_names:
            # 각 모델의 예측을 누적
            ensemble_mask = np.zeros((config.height, config.width), dtype=np.float32)

            for df in dfs:
                mask_row = df[(df['image_name'] == img_name) & (df['class'] == class_name)]
                if not mask_row.empty:
                    mask = process_single_prediction(mask_row.iloc[0], config.height, config.width)
                    ensemble_mask += mask

            # 평균을 구하고 임계값 적용
            ensemble_mask = ensemble_mask / num_models
            binary_mask = (ensemble_mask > config.threshold).astype(np.uint8)

            # RLE 인코딩 및 결과 저장
            rle = encode_mask_to_rle(binary_mask)
            predictions.append({
                "image_name": img_name,
                "class": class_name,
                "rle": rle
            })

            # 메모리 해제
            del ensemble_mask, binary_mask

    return pd.DataFrame(predictions)

def main():
    args = parse_args()
    config = EnsembleConfig(**vars(args))

    try:
        print("\n=== Starting Ensemble Process ===")
        check_paths(config)

        # CSV 파일 로드
        dfs = load_csv_files(config.input_dir)
        print(f"Loaded {len(dfs)} valid model predictions")

        # 유니크한 이미지와 클래스 얻기
        all_images = sorted(set(dfs[0]['image_name']))
        all_classes = sorted(set(dfs[0]['class']))

        print(f"Total images to process: {len(all_images)}")
        print(f"Total classes to process: {len(all_classes)}")

        # 청크 단위로 처리
        all_predictions = []

        for img_chunk in get_chunks(all_images, config.chunk_size):
            for class_chunk in get_chunks(all_classes, config.class_chunk_size):
                chunk_predictions = process_chunk(img_chunk, class_chunk, dfs, config)
                all_predictions.append(chunk_predictions)

                # 메모리 정리
                del chunk_predictions

        # 모든 예측 결합
        final_df = pd.concat(all_predictions, ignore_index=True)

        # 결과 저장
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"ensemble_result_{timestamp}.csv"
        output_path = Path(config.input_dir) / output_filename
        final_df.to_csv(output_path, index=False)

        print(f"\nSuccessfully saved ensemble results to {output_path}")
        print(f"Total predictions: {len(final_df)}")

        return final_df

    except Exception as e:
        print(f"\nCritical error in ensemble process: {e}")
        raise

if __name__ == '__main__':
    main()
