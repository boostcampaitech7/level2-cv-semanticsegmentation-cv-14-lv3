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
- 이 코드를 실행하기 위해서는, 파일 이름이 아래와 같은 형식으로 저장되어야 합니다.
- 9743.csv, 9738.csv, 9745.csv, ... (가중치의 역할을 합니다.)
    ng-youn (User name)
    ├─ output : 9743.csv, 9738.csv, 9745.csv, ...
    ├─ tools
    │  ├─ ensemble
    │  │  ├─ 2class_ensemble.py
    │  │  ├─ soft_ensemble.py
    │  │  └─ weight_ensemble.py  # This is where the main script resides
    │  └─ streamlit
    ├─ functions.py               # Required for encode_mask_to_rle and decode_rle_to_mask
    ├─ dataset.py
'''

def parse_args():
    parser = argparse.ArgumentParser(description='Weight Voting Ensemble')
    parser.add_argument('--input_dir', type=str, default='/data/ephemeral/home/ng-youn/output',
                        help='output.csv들이 위치한 폴더의 이름을 입력해주세요.')
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
    if not Path(config.input_dir).exists():
        raise FileNotFoundError(f"Input directory does not exist: {config.input_dir}")

    if not Path(config.image_dir).exists():
        raise FileNotFoundError(f"Image directory does not exist: {config.image_dir}")

def calculate_weights_from_filenames(csv_files: List[Path]) -> List[float]:
    """
    Calculate weights based on the scores in filenames.
    Higher scores get higher weights.
    """
    # Extract scores from filenames
    scores = [float(f.stem) for f in csv_files]

    # Convert to numpy array for easier manipulation
    scores = np.array(scores)

    # Normalize scores to create weights that sum to 1
    weights = scores / scores.sum()

    # Print weight information for verification
    for file, weight in zip(csv_files, weights):
        print(f"File: {file.name}, Weight: {weight:.4f}")

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

def load_csv_files(input_dir: str) -> Tuple[List[pd.DataFrame], List[float]]:
    """Load CSV files and calculate weights based on filenames."""
    csv_files = sorted(Path(input_dir).glob("*.csv"))

    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {input_dir}")

    # Calculate weights based on filenames
    weights = calculate_weights_from_filenames(csv_files)

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
    num_models: int,
    classes: List[str]
) -> pd.DataFrame:
    """Create final predictions with weighted thresholding."""
    predictions = []

    for img_name, class_preds in tqdm(ensemble.items(), desc="Weight voting ensemble in progress..."):
        for bone in classes:
            # For weighted ensemble, we already have weighted sum
            binary_mask = class_preds[bone] > config.threshold

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

        # Save results to input_dir with ensemble suffix
        output_path = Path(config.input_dir) / f"weight_ensemble_result.csv"
        final_df.to_csv(output_path, index=False)
        print(f"\nSuccessfully saved weighted ensemble results to {output_path}")
        print(f"Total predictions: {len(final_df)}")

        return final_df

    except Exception as e:
        print(f"\nCritical error in ensemble process: {e}")
        raise

if __name__ == '__main__':
    main()
