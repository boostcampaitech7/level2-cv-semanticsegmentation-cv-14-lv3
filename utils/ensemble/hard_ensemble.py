import os
import numpy as np
import pandas as pd
import copy
import argparse
from tqdm import tqdm
from functions import encode_mask_to_rle, decode_rle_to_mask
from dataset import CLASSES

'''
- Source Code : https://github.com/boostcampaitech6/level2-cv-semanticsegmentation-cv-12/blob/main/utils/ensemble.ipynb
- Method : Majority Voting ensemble
'''

def parse_args():
    parser = argparse.ArgumentParser(description='Human Bone Image Segmentation Ensemble')

    parser.add_argument('--output_dir', type=str, default='/data/ephemeral/home/ng-youn/output',
                        help='Directory where output.csv files exist')
    parser.add_argument('--output_path', type=str, default='ensemble_result.csv',
                        help='Path to save the ensemble result CSV')
    parser.add_argument('--image_dir', type=str, default='/data/ephemeral/home/data/test/DCM',
                        help='Directory containing test images')
    parser.add_argument('--threshold', type=float, default=3,
                        help='Minimum number of models that must agree for a positive prediction')
    parser.add_argument('--ensemble_type', type=str, choices=['hard', 'soft'], default='hard',
                        help='Type of ensemble: hard (majority voting) or soft (probability averaging)')

    return parser.parse_args()

def load_csv_files(output_dir: str) -> list:
    """Load all CSV files from the output directory with progress bar."""
    outputs = []
    csv_files = [f for f in os.listdir(output_dir) if f.endswith(".csv")]

    for output in tqdm(csv_files, desc="Loading CSV files"):
        file_path = os.path.join(output_dir, output)
        try:
            df = pd.read_csv(file_path)
            outputs.append(df)
        except Exception as e:
            print(f"Error loading {output}: {e}")
    return outputs

def load_images(image_dir: str) -> set:
    """Load all PNG image files from the image directory."""
    if not os.path.exists(image_dir):
        raise FileNotFoundError(f"Image directory not found: {image_dir}")

    pngs = {
        os.path.relpath(os.path.join(root, fname), start=image_dir)
        for root, _dirs, files in os.walk(image_dir)
        for fname in files
        if os.path.splitext(fname)[1].lower() == ".png"
    }
    return pngs

def initialize_ensemble_dict(pngs: set, classes: list, height: int, width: int) -> dict:
    """Initialize the ensemble dictionary with zeros for each image and class."""
    class_dict = {
        bone: np.zeros((height, width), dtype=np.uint8)
        for bone in tqdm(classes, desc="Initializing classes")
    }

    return {
        png[6:]: copy.deepcopy(class_dict)
        for png in tqdm(pngs, desc="Initializing images")
    }

def process_predictions_hard(ensemble: dict, dfs: list, height: int, width: int) -> dict:
    """Process predictions using hard voting (majority voting)."""
    total_rows = sum(len(df) for df in dfs)
    pbar = tqdm(total=total_rows, desc="Processing predictions (Hard Ensemble)")

    for fold, df in enumerate(dfs):
        for index, row in df.iterrows():
            if pd.isna(row['rle']):
                print(f'Warning: Missing RLE in fold {fold}, index {index}')
                pbar.update(1)
                continue

            try:
                mask_img = decode_rle_to_mask(row['rle'], height, width)
                ensemble[row['image_name']][row['class']] += mask_img
            except Exception as e:
                print(f'Error processing fold {fold}, index {index}: {e}')
                print(row)
            pbar.update(1)

    pbar.close()
    return ensemble

def process_predictions_soft(ensemble: dict, dfs: list, height: int, width: int) -> dict:
    """Process predictions using soft voting (averaging probabilities)."""
    num_models = len(dfs)
    total_rows = sum(len(df) for df in dfs)
    pbar = tqdm(total=total_rows, desc="Processing predictions (Soft Ensemble)")

    for fold, df in enumerate(dfs):
        for index, row in df.iterrows():
            if pd.isna(row['rle']):
                print(f'Warning: Missing RLE in fold {fold}, index {index}')
                pbar.update(1)
                continue

            try:
                mask_img = decode_rle_to_mask(row['rle'], height, width)
                ensemble[row['image_name']][row['class']] += mask_img.astype(float)
            except Exception as e:
                print(f'Error processing fold {fold}, index {index}: {e}')
                print(row)
            pbar.update(1)

    pbar.close()

    # Convert accumulated sum to probability
    for img_name in tqdm(ensemble, desc="Computing probabilities"):
        for class_name in ensemble[img_name]:
            ensemble[img_name][class_name] /= num_models

    return ensemble

def create_final_predictions(ensemble: dict, threshold: float, pngs: set, classes: list) -> pd.DataFrame:
    """Create final predictions by applying threshold."""
    predictions = []

    for png in tqdm(pngs, desc="Creating final predictions"):
        image_name = png[6:]
        for bone in classes:
            binary_arr = np.where(ensemble[image_name][bone] > threshold, 1, 0)
            rle = encode_mask_to_rle(binary_arr)
            predictions.append({
                "image_name": image_name,
                "class": bone,
                "rle": rle
            })

    return pd.DataFrame(predictions)

def main():
    args = parse_args()
    height, width = 2048, 2048

    try:
        print("\n=== Starting Ensemble Process ===")
        pngs = load_images(args.image_dir)
        print(f"Found {len(pngs)} images")

        dfs = load_csv_files(args.output_dir)
        if not dfs:
            raise ValueError("No CSV files found in output directory")
        print(f"Loaded {len(dfs)} model predictions")

        # Initialize ensemble dictionary
        ensemble = initialize_ensemble_dict(pngs, CLASSES, height, width)

        # Process predictions based on ensemble type
        if args.ensemble_type == 'soft':
            ensemble = process_predictions_soft(ensemble, dfs, height, width)
            # For soft ensemble, threshold should typically be around 0.5
            threshold = args.threshold if args.threshold <= 1 else 0.5
        else:  # hard ensemble
            ensemble = process_predictions_hard(ensemble, dfs, height, width)
            threshold = args.threshold

        # Create final predictions
        df = create_final_predictions(ensemble, threshold, pngs, CLASSES)

        # Save results
        output_path = os.path.join(args.output_dir, args.output_path)
        df.to_csv(output_path, index=False)
        print(f"\nSaved ensemble results to {output_path}")
        print(f"Total predictions: {len(df)}")

        return df

    except Exception as e:
        print(f"\nError in ensemble process: {e}")
        raise

if __name__ == '__main__':
    main()
