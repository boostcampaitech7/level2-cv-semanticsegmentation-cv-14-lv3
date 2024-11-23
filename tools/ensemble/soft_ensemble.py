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
- Source Code : https://github.com/boostcampaitech6/level2-cv-semanticsegmentation-cv-12/blob/main/utils/ensemble.ipynb
- Method : Majority Voting ensemble (Hard)
- 가중치 반영 없이, 5개의 csv file에서 3개의 csv file 이상에서 예측된 값을 결과로 저장합니다.
'''

def parse_args():
    parser = argparse.ArgumentParser(description='Improved Human Bone Image Segmentation Ensemble')
    parser.add_argument('--output_dir', type=str, default='/data/ephemeral/home/ng-youn/output',
                        help='output.csv들이 위치한 폴더의 이름을 입력해주세요.')
    parser.add_argument('--output_path', type=str, default='/data/ephemeral/home/ng-youn')
    parser.add_argument('--image_dir', type=str, default='/data/ephemeral/home/data/test/DCM',
                        help='Test image의 경로')
    parser.add_argument('--threshold', type=float, default=0.6)

    return parser.parse_args()

@dataclass
class EnsembleConfig:
    output_dir: str
    output_path: str
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

def validate_predictions(dfs: List[pd.DataFrame]) -> None:
    """Validate consistency across prediction files."""
    if not dfs:
        raise ValueError("No prediction files provided")

    # Check if all DataFrames have the same structure
    base_df = dfs[0]
    required_columns = {'image_name', 'class', 'rle'}

    for idx, df in enumerate(dfs, 1):
        if not set(df.columns).issuperset(required_columns):
            raise ValueError(f"Prediction file {idx} missing required columns: {required_columns}")

        # Check for matching image names and classes
        if not set(df['image_name']).issubset(set(base_df['image_name'])):
            raise ValueError(f"Prediction file {idx} has mismatched image names")
        if not set(df['class']).issubset(set(base_df['class'])):
            raise ValueError(f"Prediction file {idx} has mismatched classes")

def load_csv_files(output_dir: str) -> List[pd.DataFrame]:
    """Load and validate CSV files from the output directory."""
    outputs = []
    csv_files = list(Path(output_dir).glob("*.csv"))

    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {output_dir}")

    for file_path in tqdm(csv_files, desc="Loading CSV files"):
        try:
            df = pd.read_csv(file_path)
            outputs.append(df)
        except Exception as e:
            raise ValueError(f"Error loading {file_path}: {e}")

    validate_predictions(outputs)
    return outputs

def process_images(
    image_names: List[str],
    dfs: List[pd.DataFrame],
    classes: List[str],
    config: EnsembleConfig
) -> Dict:
    """Process all images at once."""
    ensemble = {
        img: {bone: np.zeros((config.height, config.width), dtype=np.float32)
              for bone in classes}
        for img in image_names
    }

    for df in dfs:
        for _, row in df.iterrows():
            if pd.isna(row['rle']):
                warnings.warn(f"Missing RLE for {row['image_name']}, class {row['class']}")
                continue

            try:
                mask = decode_rle_to_mask(row['rle'], config.height, config.width)
                ensemble[row['image_name']][row['class']] += mask.astype(np.float32)
            except Exception as e:
                warnings.warn(f"Error processing {row['image_name']}, class {row['class']}: {e}")

    return ensemble

def create_final_predictions(
    ensemble: Dict,
    config: EnsembleConfig,
    num_models: int,
    classes: List[str]
) -> pd.DataFrame:
    """Create final predictions with proper thresholding."""
    predictions = []

    for img_name, class_preds in tqdm(ensemble.items(), desc="Soft voting ensemble in progress..."):
        for bone in classes:
            # Normalize and threshold probabilities
            pred = class_preds[bone] / num_models
            binary_mask = pred > config.threshold

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
        print("\n=== Starting Ensemble Process ===")

        # Check paths validity
        check_paths(config)

        # Load and validate predictions
        dfs = load_csv_files(config.output_dir)
        num_models = len(dfs)
        print(f"Loaded {num_models} valid model predictions")

        # Get unique image names and classes
        all_images = sorted(set(dfs[0]['image_name']))
        classes = sorted(set(dfs[0]['class']))

        # Process all images at once
        ensemble_results = process_images(all_images, dfs, classes, config)

        # Create final predictions
        final_df = create_final_predictions(ensemble_results, config, num_models, classes)

        # Save results
        output_path = Path(config.output_dir) / config.output_path
        final_df.to_csv(output_path, index=False)
        print(f"\nSuccessfully saved ensemble results to {output_path}")
        print(f"Total predictions: {len(final_df)}")

        return final_df

    except Exception as e:
        print(f"\nCritical error in ensemble process: {e}")
        raise

if __name__ == '__main__':
    main()
