import os
import cv2
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import albumentations as A
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import ttach as tta

# Config
KFOLD_N = 0
BATCH_SIZE, BATCH_SIZE_VALID = 16, 1
LR = 1e-4
RANDOM_SEED = 42
NUM_EPOCHS = 100
VAL_EVERY = 5
SAVED_DIR = ""
DATA_ROOT = "~/data"
IMAGE_ROOT, LABEL_ROOT = DATA_ROOT + "/train/DCM", DATA_ROOT + "/train/outputs_json"
TEST_IMAGE_ROOT = DATA_ROOT + "/test/DCM"
MODEL_PATH = "../donghwan/checkpoints/best_dice_0.9729.pt"

CLASSES = [
    'finger-1', 'finger-2', 'finger-3', 'finger-4', 'finger-5',
    'finger-6', 'finger-7', 'finger-8', 'finger-9', 'finger-10',
    'finger-11', 'finger-12', 'finger-13', 'finger-14', 'finger-15',
    'finger-16', 'finger-17', 'finger-18', 'finger-19', 'Trapezium',
    'Trapezoid', 'Capitate', 'Hamate', 'Scaphoid', 'Lunate',
    'Triquetrum', 'Pisiform', 'Radius', 'Ulna',
]

# 각 Category의 RGB 값
PALETTE = [
    (220, 20, 60), (119, 11, 32), (0, 0, 142), (0, 0, 230), (106, 0, 228),
    (0, 60, 100), (0, 80, 100), (0, 0, 70), (0, 0, 192), (250, 170, 30),
    (100, 170, 30), (220, 220, 0), (175, 116, 175), (250, 0, 30), (165, 42, 42),
    (255, 77, 255), (0, 226, 252), (182, 182, 255), (0, 82, 0), (120, 166, 157),
    (110, 76, 0), (174, 57, 255), (199, 100, 0), (72, 0, 118), (255, 179, 240),
    (0, 125, 92), (209, 0, 151), (188, 208, 182), (0, 220, 176),
]

CLASS2IND = {v: i for i, v in enumerate(CLASSES)}
IND2CLASS = {v: k for k, v in CLASS2IND.items()}


##### 구분선 #####
def encode_mask_to_rle(mask):
    '''
    mask: numpy array binary mask
    1 - mask
    0 - background
    Returns encoded run length
    '''
    pixels = mask.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

def decode_rle_to_mask(rle, height, width):
    s = rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(height * width, dtype=np.uint8)

    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1

    return img.reshape(height, width)

def label2rgb(label):
    image_size = label.shape[1:] + (3, )
    image = np.zeros(image_size, dtype=np.uint8)

    for i, class_label in enumerate(label):
        image[class_label == 1] = PALETTE[i]

    return image

class XRayInferenceDataset(Dataset):
    def __init__(self, transforms=None):
        _filenames = {
            os.path.relpath(os.path.join(root, fname), start=TEST_IMAGE_ROOT)
            for root, _dirs, files in os.walk(TEST_IMAGE_ROOT)
            for fname in files
            if os.path.splitext(fname)[1].lower() == ".png"
        }
        _filenames = np.array(sorted(_filenames))

        self.filenames = _filenames
        self.transforms = transforms

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, item):
        image_name = self.filenames[item]
        image_path = os.path.join(TEST_IMAGE_ROOT, image_name)

        image = cv2.imread(image_path)
        image = image / 255.

        if self.transforms is not None:
            inputs = {"image": image}
            result = self.transforms(**inputs)
            image = result["image"]

        # to tensor will be done later
        image = image.transpose(2, 0, 1)    # make channel first
        image = torch.from_numpy(image).float()

        return image, image_name

def test(model, data_loader, thr=0.5):
    model = model.cuda()
    model.eval()

    rles = []
    filename_and_class = []
    with torch.no_grad():
        for step, (images, image_names) in tqdm(enumerate(data_loader), total=len(data_loader)):
            images = images.cuda()
            outputs = model(images)

            # restore original size
            outputs = F.interpolate(outputs, size=(2048, 2048), mode="bilinear")
            outputs = torch.sigmoid(outputs)
            outputs = (outputs > thr).detach().cpu().numpy()

            for output, image_name in zip(outputs, image_names):
                for c, segm in enumerate(output):
                    rle = encode_mask_to_rle(segm)
                    rles.append(rle)
                    filename_and_class.append(f"{IND2CLASS[c]}_{image_name}")

    return rles, filename_and_class

def visualize_prediction(image, preds):
    fig, ax = plt.subplots(1, 2, figsize=(24, 12))
    ax[0].imshow(image)    # remove channel dimension
    ax[1].imshow(label2rgb(preds))
    plt.show()

def main():
    # Load model
    model = torch.load(os.path.join(SAVED_DIR, MODEL_PATH))

    # Setup TTA
    transforms = tta.Compose([
        tta.HorizontalFlip(),
        # tta.Scale(scales=[0.9, 1.0, 1.1])
    ])
    tta_model = tta.SegmentationTTAWrapper(model, transforms)

    # Setup dataset and dataloader
    tf = A.Resize(512, 512)
    test_dataset = XRayInferenceDataset(transforms=tf)
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=8,
        shuffle=False,
        num_workers=2,
        drop_last=False
    )

    # Run inference
    rles, filename_and_class = test(tta_model, test_loader)

    # Process results
    classes, filename = zip(*[x.split("_") for x in filename_and_class])
    image_name = [os.path.basename(f) for f in filename]

    # Save results
    df = pd.DataFrame({
        "image_name": image_name,
        "class": classes,
        "rle": rles,
    })
    df.to_csv(os.path.join(SAVED_DIR, "output.csv"), index=False)

if __name__ == "__main__":
    main()
