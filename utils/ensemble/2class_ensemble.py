import os, sys
import cv2
import json
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import List, Tuple

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from functions import encode_mask_to_rle, decode_rle_to_mask
from dataset import CLASSES

'''
- Source Code : https://github.com/boostcampaitech6/level2-cv-semanticsegmentation-cv-03/blob/main/utils/validation_ensemble_2class.py
- ['Trapezoid', 'Pisiform'] 구분이 어려운 2개 category를 ensemble 합니다.
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


def ensemble_masks(csv_files: List[str],
                   height: int = 2048,
                   width: int = 2048,
                   threshold: float = 0.6) -> pd.DataFrame:
    """
    Ensemble masks from multiple CSV predictions with adaptive sampling.

    Args:
        csv_files (List[str]): List of CSV file paths
        height (int): Image height
        width (int): Image width
        threshold (float): Ensemble threshold

    Returns:
        pd.DataFrame: Ensembled predictions
    """
    # Adaptive random sampling of input files
    num_random_elements = random.randint(1, len(csv_files))
    rand_csv_files = random.sample(csv_files, num_random_elements)
    weights = [1/len(rand_csv_files)] * len(rand_csv_files)

    # Load and filter DataFrames
    combined_df = [pd.read_csv(f) for f in rand_csv_files]
    filtered_rows = [df.loc[df['class'].isin(['Trapezoid', 'Pisiform'])] for df in combined_df]

    # Extract class names
    class_names = [name for name in combined_df[0]['class'].tolist()
                   if name in ['Trapezoid', 'Pisiform']]

    image_class = []
    rles = []

    for i in tqdm(range(len(class_names)), desc="Ensembling Masks"):
        weighted_masks = []
        for df, w in zip(filtered_rows, weights):
            rle = df.iloc[i]['rle']
            weighted_masks.append(decode_rle_to_mask(rle, height, width) * w if not pd.isna(rle) else np.zeros((height, width)))

        # Weighted mask combination and thresholding
        combined_mask = sum(weighted_masks)
        combined_mask = (combined_mask > threshold).astype(np.uint8)

        rles.append(encode_mask_to_rle(combined_mask))
        image_class.append(f"{filtered_rows[0].iloc[i]['image_name']}_{filtered_rows[0].iloc[i]['class']}")

    # Create submission DataFrame
    filename, classes = zip(*[x.split("_") for x in image_class])
    image_name = [os.path.basename(f) for f in filename]

    return pd.DataFrame({
        "image_name": image_name,
        "class": classes,
        "rle": rles,
    })


def calculate_dice_scores(submission: pd.DataFrame,
                           label_root: str,
                           class_names: List[str] = ['Trapezoid', 'Pisiform']) -> Tuple[List[str], float]:
    """
    Calculate Dice scores for ensembled predictions.

    Args:
        submission (pd.DataFrame): Submission DataFrame
        label_root (str): Root directory for ground truth labels
        class_names (List[str]): Classes to evaluate

    Returns:
        Tuple[List[str], float]: Dice scores per class and average
    """
    eps = 1e-4
    CLASS2IND = {v: i for i, v in enumerate(class_names)}
    image_names = submission['image_name'].unique()

    dices = []
    for image_name in tqdm(image_names, desc="Calculating Dice Scores"):
        json_name = image_name.replace('-','_').replace('.png', '.json')
        label_path = os.path.join(label_root, json_name)

        # Prepare ground truth masks
        gt_masks = np.zeros((29, 2048, 2048), dtype=np.uint8)
        with open(label_path, "r") as f:
            annotations = json.load(f)

        for ann in annotations["annotations"]:
            c = ann["label"]
            if c not in class_names:
                continue

            class_ind = CLASS2IND[c]
            points = np.array(ann["points"])

            # Fill polygon mask
            class_label = np.zeros((2048, 2048), dtype=np.uint8)
            cv2.fillPoly(class_label, [points], 1)
            gt_masks[class_ind] = class_label

        gt_masks = gt_masks.reshape(29, -1)

        # Prepare prediction masks
        val_masks = np.zeros((29, 2048, 2048), dtype=np.uint8)
        for idx, class_name in enumerate(class_names):
            rle = submission[(submission['image_name'] == image_name) &
                             (submission['class'] == class_name)].iloc[0]['rle']
            val_masks[idx] = decode_rle_to_mask(rle, 2048, 2048)
        val_masks = val_masks.reshape(29, -1)

        # Calculate Dice scores
        intersection = np.sum(gt_masks * val_masks, -1)
        dice_score = (2. * intersection + eps) / (np.sum(gt_masks, -1) + np.sum(val_masks, -1) + eps)
        dices.append(dice_score)

    # Calculate and report Dice scores
    dices_per_class = np.mean(dices, axis=0)
    dice_str = [f"{c}: {d:.4f}" for c, d in zip(class_names, dices_per_class) if c in ["Trapezoid", "Pisiform"]]

    return dice_str, np.mean(dices_per_class)


def main():
    args = parse_args()

    label_root = args.image_dir
    save_dir = args.output_path

    os.makedirs(save_dir, exist_ok=True)

    while True:
        # Ensemble masks
        print('='*10)
        submission = ensemble_masks(load_csv_files(args.output_dir))

        # Calculate and print Dice scores
        dice_scores, avg_dice = calculate_dice_scores(submission, label_root)
        print("Dice Scores:", ", ".join(dice_scores))
        print(f"Average Dice Score: {avg_dice:.4f}")

        # Optional: Save submission
        submission.to_csv(os.path.join(save_dir, 'class_ens_th_6.csv'), index=False)

        # Optional: Add a break condition or user interaction
        # to control the continuous execution
        break


if __name__ == '__main__':
    main()
