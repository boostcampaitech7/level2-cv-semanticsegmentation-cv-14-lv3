import os, cv2, json, random, sys
import pandas as pd
import numpy as np
from tqdm import tqdm
import re

from functions import encode_mask_to_rle, decode_rle_to_mask

def rle_to_mask(rle, height, width):
    s = np.array(rle.split(), dtype=int)
    starts, lengths = s[0::2] - 1, s[1::2]
    ends = starts + lengths

    mask = np.zeros(height * width, dtype=np.int32)
    mask[starts] += 1
    mask[ends] -= 1

    mask = np.cumsum(mask)
    return mask.reshape(height, width).astype(np.uint8)

def mask_to_rle(mask):
    pixels = mask.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

def main():
    csv_files = [
        '/data/output/output1.csv',
        '/data/output/output2.csv',
        '/data/output/output3.csv',
        '/data/output/output4.csv'
    ]

    while True:
        num_random_elements = random.randint(1, len(csv_files))
        rand_csv_files = random.sample(csv_files, num_random_elements)
        print('='*10)
        for rand_csv_file in rand_csv_files:
            print('Using :',rand_csv_file)
        weights = [1/len(rand_csv_files)]*len(rand_csv_files)

        combined_df = [pd.read_csv(f) for f in rand_csv_files]
        filtered_rows = [df.loc[df['class'].isin(['Trapezoid', 'Pisiform'])] for df in combined_df]

        class_name = [name for name in combined_df[0]['class'].tolist() if name in ['Trapezoid', 'Pisiform']]

        image_class = []
        rles = []

        height, width = 2048, 2048

        for i in tqdm(range(len(class_name))):
            weighted_masks = []
            for df, w in zip(filtered_rows, weights):
                if type(df.iloc[i]['rle']) == float:
                    weighted_masks.append(np.zeros((height, width)))
                    continue
                weighted_masks.append(rle_to_mask(df.iloc[i]['rle'], height, width) * w)

            combined_mask = sum(weighted_masks)

            combined_mask[combined_mask <= 0.5] = 0
            combined_mask[combined_mask > 0.5] = 1
            combined_mask = combined_mask.astype(np.uint8)

            image = np.zeros((height, width), dtype=np.uint8)
            image += combined_mask

            rles.append(mask_to_rle(image))
            image_class.append(f"{filtered_rows[0].iloc[i]['image_name']}_{filtered_rows[0].iloc[i]['class']}")

        filename, classes = zip(*[x.split("_") for x in image_class])
        image_name = [os.path.basename(f) for f in filename]

        submission = pd.DataFrame({
                        "image_name": image_name,
                        "class": classes,
                        "rle": rles,
                    })

        save_dir = '/data/ephemeral/home/ng-youn/output'
        submission.to_csv(os.path.join(save_dir, 'class_ens_th_5.csv'), index=False)

        class_names = ['Trapezoid', 'Pisiform']
        CLASS2IND = {v: i for i, v in enumerate(class_names)}

        eps = 0.0001
        label_root = '/data/train/outputs_json'
        csv_path = '/data/sub.csv'
        df = pd.read_csv(csv_path)
        df = submission
        image_names = df['image_name'].unique()

        dices = []

        ##### 구분선 #####

        for image_name in tqdm(image_names, total=len(image_names)):
            match = re.search(r'image(\d+)', image_name)
            if not match:
                print(f"Warning: No numeric ID found in image name {image_name}")
                continue

            id_folder = match.group(1)
            json_name = image_name.replace('-', '_').replace('.png', '.json')
            label_path = os.path.join(label_root, id_folder, json_name)

            if not os.path.exists(label_path):
                print(f"Warning: JSON file not found for image {image_name}")
                print(f"Attempted path: {label_path}")
                continue

            gt_masks = np.zeros((29, 2048, 2048), dtype=np.uint8)
            with open(label_path, "r") as f:
                annotations = json.load(f)
            annotations = annotations["annotations"]
            for ann in annotations:
                c = ann["label"]
                if c not in class_names: continue
                class_ind = CLASS2IND[c]
                points = np.array(ann["points"])

                class_label = np.zeros((2048, 2048), dtype=np.uint8)
                cv2.fillPoly(class_label, [points], 1)
                gt_masks[class_ind] = class_label
            gt_masks = gt_masks.reshape(29, -1)

            val_masks = np.zeros((29, 2048, 2048), dtype=np.uint8)
            for idx, class_name in enumerate(class_names):
                rle = df[(df['image_name'] == image_name) & (df['class']== class_name)].iloc[0]['rle']
                if pd.isna(rle):
                    print('nan :', image_name, class_name)
                    val_masks[idx] = np.zeros((2048, 2048), dtype=np.uint8)
                else:
                    val_masks[idx] = decode_rle_to_mask(rle, 2048, 2048)
            val_masks = val_masks.reshape(29, -1)

            intersection = np.sum(gt_masks * val_masks, -1)
            dice_score = (2. * intersection + eps) / (np.sum(gt_masks, -1) + np.sum(val_masks, -1) + eps)
            dices.append(dice_score)

        if len(dices) == 0:
            print("Warning: No valid dice scores calculated. Check input data.")
            dices_per_class = np.zeros(len(class_names))
        else:
            dices_per_class = np.mean(dices, axis=0)

        dice_str = []
        for c, d in zip(class_names, dices_per_class):
            if c in ["Trapezoid", "Pisiform"]:
                dice_str.append(f"{c}: {d:.4f}")

        dice_str = ", ".join(dice_str)
        avg_dice = np.mean(dices_per_class)
        print(dice_str)

if __name__ == '__main__':
    main()
