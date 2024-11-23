import sys
import numpy as np
import pandas as pd
import argparse
from tqdm import tqdm
from typing import List, Dict, Set, Tuple
import warnings
from dataclasses import dataclass
from pathlib import Path
import random

sys.path.append("/data/ephemeral/home/ng-youn")
from functions import encode_mask_to_rle, decode_rle_to_mask
from dataset import CLASSES

'''
- Source Code : https://github.com/boostcampaitech6/level2-cv-semanticsegmentation-cv-03/blob/main/utils/validation_ensemble_2class.py
- 구분이 어려운 2개 category('Trapezoid' & 'Pisiform')에 대하여 ensemble 합니다. (다른 값은 수정하지 않습니다.)
- [Bug] 현재 코드는 정상적으로 작동하지만, 앙상블된 결과를 upstage에 제출하면 점수가 '-1'로 기록되는 문제가 있습니다.
'''
@dataclass
class EnsembleConfig:
    output_dir: str
    output_path: str
    image_dir: str
    threshold: float
    ensemble_type: str
    target_classes: List[str] = ('Trapezoid', 'Pisiform')
    height: int = 2048
    width: int = 2048

def parse_args():
    parser = argparse.ArgumentParser(description='Human Bone Image Segmentation Ensemble')
    parser.add_argument('--output_dir', type=str, default='/data/ephemeral/home/ng-youn/output',
                        help='Directory where output.csv files exist')
    parser.add_argument('--output_path', type=str, default='ensemble_result.csv',
                        help='Path to save the ensemble result CSV')
    parser.add_argument('--image_dir', type=str, default='/data/ephemeral/home/data/test/DCM',
                        help='Directory containing test images')
    parser.add_argument('--threshold', type=float, default=0.6,
                        help='Threshold for ensemble agreement')
    parser.add_argument('--ensemble_type', type=str, choices=['hard', 'soft'], default='soft',
                        help='Type of ensemble: hard (majority voting) or soft (probability averaging)')

    return parser.parse_args()

def validate_predictions(dfs: List[pd.DataFrame], target_classes: Tuple[str, str]) -> None:
    """Validate consistency across prediction files for target classes."""
    if not dfs:
        raise ValueError("No prediction files provided")

    required_columns = {'image_name', 'class', 'rle'}
    for idx, df in enumerate(dfs, 1):
        if not set(df.columns).issuperset(required_columns):
            raise ValueError(f"Prediction file {idx} missing required columns")

        if not set(df['class']).issuperset(set(target_classes)):
            raise ValueError(f"Prediction file {idx} missing target classes: {target_classes}")

def load_csv_files(output_dir: str, target_classes: Tuple[str, str]) -> List[pd.DataFrame]:
    """Load and validate CSV files, filtering for target classes."""
    outputs = []
    csv_files = list(Path(output_dir).glob("*.csv"))

    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {output_dir}")

    for file_path in tqdm(csv_files, desc="Loading CSV files"):
        try:
            df = pd.read_csv(file_path)
            # Filter for target classes
            df_filtered = df[df['class'].isin(target_classes)].copy()
            outputs.append(df_filtered)
        except Exception as e:
            raise ValueError(f"Error loading {file_path}: {e}")

    validate_predictions(outputs, target_classes)
    return outputs

def process_predictions(
    dfs: List[pd.DataFrame],
    config: EnsembleConfig
) -> pd.DataFrame:
    """Process predictions using adaptive sampling and weighted ensemble."""
    # Adaptive sampling of input predictions
    num_models = random.randint(max(2, len(dfs) // 2), len(dfs))
    selected_dfs = random.sample(dfs, num_models)
    weights = [1/num_models] * num_models

    # Get unique image names
    image_names = sorted(set(dfs[0]['image_name']))

    final_predictions = []
    for img_name in tqdm(image_names, desc="Processing predictions"):
        for target_class in config.target_classes:
            weighted_masks = []

            # Collect masks from all selected models
            for df, weight in zip(selected_dfs, weights):
                mask_row = df[(df['image_name'] == img_name) & (df['class'] == target_class)]
                if not mask_row.empty and not pd.isna(mask_row.iloc[0]['rle']):
                    try:
                        mask = decode_rle_to_mask(mask_row.iloc[0]['rle'], config.height, config.width)
                        weighted_masks.append(mask.astype(np.float32) * weight)
                    except Exception as e:
                        warnings.warn(f"Error processing mask for {img_name}, {target_class}: {e}")
                        weighted_masks.append(np.zeros((config.height, config.width), dtype=np.float32))
                else:
                    weighted_masks.append(np.zeros((config.height, config.width), dtype=np.float32))

            # Combine masks
            if config.ensemble_type == 'soft':
                combined_mask = sum(weighted_masks)
                final_mask = (combined_mask > config.threshold).astype(np.uint8)
            else:  # hard voting
                combined_mask = sum([mask > 0 for mask in weighted_masks])
                final_mask = (combined_mask >= (num_models * config.threshold)).astype(np.uint8)

            # Encode result
            rle = encode_mask_to_rle(final_mask)
            final_predictions.append({
                'image_name': img_name,
                'class': target_class,
                'rle': rle
            })

    return pd.DataFrame(final_predictions)

def main():
    args = parse_args()
    config = EnsembleConfig(**vars(args))

    try:
        print("\n=== Starting 2-Class Ensemble Process ===")
        print(f"Target classes: {config.target_classes}")

        # Load and validate predictions
        dfs = load_csv_files(config.output_dir, config.target_classes)
        print(f"Loaded {len(dfs)} valid model predictions")

        # Process predictions
        final_df = process_predictions(dfs, config)

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
