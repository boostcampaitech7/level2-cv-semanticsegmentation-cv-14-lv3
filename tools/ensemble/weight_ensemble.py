import os, sys
import numpy as np
import pandas as pd
import copy
import argparse
from tqdm import tqdm
from typing import List, Dict, Set, Tuple
import warnings
from dataclasses import dataclass
from pathlib import Path

sys.path.append("/data/ephemeral/home/ng-youn")
from functions import encode_mask_to_rle, decode_rle_to_mask
from dataset import CLASSES

'''
- Method : Weight Voting ensemble
- 이 방법을 사용하기 위해서는, 파일 이름이 아래와 같은 형식으로 저장되어야 합니다.
- 9743.csv, 9738.csv, 9745.csv, ... (가중치의 역할을 합니다.)
- (!) 이름 순으로 가장 마지막에 입력받는 파일의 가중치가 2 입니다. (weights = [1, 1, 1, 2])
'''

def parse_args():
    parser = argparse.ArgumentParser(description='Hard & Weight Voting Ensemble')
    parser.add_argument('--input_dir', type=str, default='/data/ephemeral/home/ng-youn/output',
                        help='output.csv들이 위치한 폴더의 경로를 입력해주세요.')
    parser.add_argument('--image_dir', type=str, default='/data/ephemeral/home/data/test/DCM',
                        help='Test image의 경로')
    parser.add_argument('--threshold', type=float, default=0.6)

    return parser.parse_args()

@dataclass
class EnsembleConfig:
    input_dir: str
    image_dir: str
    threshold: float
    height: int = 2048
    width: int = 2048

def check_paths(config: EnsembleConfig) -> None:
    """Check if the provided paths are valid."""
    if not Path(config.output_dir).exists():
        raise FileNotFoundError(f"Output directory does not exist: {config.output_dir}")

    if not Path(config.image_dir).exists():
        raise FileNotFoundError(f"Image directory does not exist: {config.image_dir}")

    if not Path(config.output_path).parent.exists():
        raise FileNotFoundError(f"Output path's parent directory does not exist: {Path(config.output_path).parent}")


def calculate_weights_from_filenames(csv_files: List[Path]) -> List[float]:
    # 점수를 제곱하여 더 큰 차이를 만듭니다
    scores = [float(f.stem) for f in csv_files]
    scores = np.array(scores) ** 2  # 제곱하여 큰 점수에 더 큰 가중치 부여

    # 합계로 정규화
    weights = scores / scores.sum()

    return weights.tolist()


def validate_predictions(dfs: List[pd.DataFrame], weights: List[float]) -> None:
    """Validate consistency across prediction files and weights."""
    if not dfs:
        raise ValueError("No prediction files provided")

    if len(dfs) != len(weights):
        raise ValueError(f"Number of models ({len(dfs)}) does not match number of weights ({len(weights)})")

    base_df = dfs[0]
    required_columns = {'image_name', 'class', 'rle'}

    for idx, df in enumerate(dfs, 1):
        if not set(df.columns).issuperset(required_columns):
            raise ValueError(f"Prediction file {idx} missing required columns: {required_columns}")

        if not set(df['image_name']).issubset(set(base_df['image_name'])):
            raise ValueError(f"Prediction file {idx} has mismatched image names")
        if not set(df['class']).issubset(set(base_df['class'])):
            raise ValueError(f"Prediction file {idx} has mismatched classes")


def load_csv_files(output_dir: str) -> Tuple[List[pd.DataFrame], List[float]]:
    """Load CSV files and calculate weights based on filenames."""
    csv_files = sorted(Path(output_dir).glob("*.csv"))

    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {output_dir}")

    # Calculate weights based on filenames
#    weights = calculate_weights_from_filenames(csv_files)
    weights = [1, 1, 1, 2]

    # Load DataFrame
    outputs = []
    for file_path in tqdm(csv_files, desc="Loading CSV files"):
        try:
            df = pd.read_csv(file_path)
            outputs.append(df)
        except Exception as e:
            raise ValueError(f"Error loading {file_path}: {e}")

    return outputs, weights


def process_images(
    image_names: List[str],
    dfs: List[pd.DataFrame],
    classes: List[str],
    config: EnsembleConfig,
    weights: List[float]
) -> Dict:
    """Process all images with weighted voting."""
    ensemble = {
        img: {bone: np.zeros((config.height, config.width), dtype=np.float32)
              for bone in classes}
        for img in image_names
    }

    for df, weight in zip(dfs, weights):
        for _, row in df.iterrows():
            if pd.isna(row['rle']):
                warnings.warn(f"Missing RLE for {row['image_name']}, class {row['class']}")
                continue

            try:
                mask = decode_rle_to_mask(row['rle'], config.height, config.width)
                ensemble[row['image_name']][row['class']] += mask.astype(np.float32) * weight

            except Exception as e:
                warnings.warn(f"Error processing {row['image_name']}, class {row['class']}: {e}")

    return ensemble


def create_final_predictions(
    ensemble: Dict,
    config: EnsembleConfig,
    classes: List[str],
    weights: List[float]
) -> pd.DataFrame:
    predictions = []

    for img_name, class_preds in tqdm(ensemble.items(), desc="Weight voting ensemble in progress..."):
        for bone in classes:
            # 가중치 평균을 기반으로 동적 threshold 계산
            dynamic_threshold = config.threshold * np.mean(weights)

            binary_mask = class_preds[bone] > dynamic_threshold

            rle = encode_mask_to_rle(binary_mask.astype(np.uint8))
            predictions.append({
                "image_name": img_name,
                "class": bone,
                "rle": rle
            })

    return pd.DataFrame(predictions)


def main():
    args = parse_args()
    config = EnsembleConfig(**vars(args))

    try:
        print("\n=== Starting Score-based Weighted Ensemble Process ===")

        # Check paths validity
        check_paths(config)

        # Load predictions and calculate weights
        dfs, weights = load_csv_files(config.input_dir)

        # Validate predictions and weights
        validate_predictions(dfs, weights)

        num_models = len(dfs)
        print(f"\nLoaded {num_models} valid model predictions")

        # Get unique image names and classes
        all_images = sorted(set(dfs[0]['image_name']))
        classes = sorted(set(dfs[0]['class']))

        # Process all images at once
        ensemble_results = process_images(all_images, dfs, classes, config, weights)

        # Create final predictions
        final_df = create_final_predictions(ensemble_results, config, num_models, classes)

        # Save results

        final_df.to_csv(config.input_dir, index=False)
        print(f"\nSuccessfully saved weighted ensemble results to {config.input_dir}")
        print(f"Total predictions: {len(final_df)}")

        return final_df

    except Exception as e:
        print(f"\nCritical error in ensemble process: {e}")
        raise

if __name__ == '__main__':
    main()
