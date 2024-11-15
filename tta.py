import os
import cv2
import numpy as np
import pandas as pd
import argparse
from tqdm.auto import tqdm
import albumentations as A
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import ttach as tta

def parse_args():
    parser = argparse.ArgumentParser()

    # Model
    parser.add_argument('--model_path', type=str, default='../donghwan/checkpoints/best_dice_0.9729.pt',
                        help='path to saved model')

    # Data
    parser.add_argument('--data_root', type=str, default="../",
                        help='root directory of dataset')
    parser.add_argument('--test_image_dir', type=str, default="test/DCM",
                        help='directory containing test images relative to data_root')

    # Output
    parser.add_argument('--save_dir', type=str, default="./",
                        help='directory to save results')

    # Inference settings
    parser.add_argument('--batch_size', type=int, default=8,
                        help='batch size for inference')
    parser.add_argument('--num_workers', type=int, default=2,
                        help='number of workers for data loading')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='threshold for binary prediction')

    args = parser.parse_args()

    args.test_image_root = os.path.join(args.data_root, args.test_image_dir)

    return args

# Constants
CLASSES = [
    'finger-1', 'finger-2', 'finger-3', 'finger-4', 'finger-5',
    'finger-6', 'finger-7', 'finger-8', 'finger-9', 'finger-10',
    'finger-11', 'finger-12', 'finger-13', 'finger-14', 'finger-15',
    'finger-16', 'finger-17', 'finger-18', 'finger-19', 'Trapezium',
    'Trapezoid', 'Capitate', 'Hamate', 'Scaphoid', 'Lunate',
    'Triquetrum', 'Pisiform', 'Radius', 'Ulna',
]
CLASS2IND = {v: i for i, v in enumerate(CLASSES)}
IND2CLASS = {v: k for k, v in CLASS2IND.items()}

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

class XRayInferenceDataset(Dataset):
    def __init__(self, root_dir, transforms=None):
        _filenames = {
            os.path.relpath(os.path.join(root, fname), start=root_dir)
            for root, _dirs, files in os.walk(root_dir)
            for fname in files
            if os.path.splitext(fname)[1].lower() == ".png"
        }
        _filenames = np.array(sorted(_filenames))

        self.root_dir = root_dir
        self.filenames = _filenames
        self.transforms = transforms

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, item):
        image_name = self.filenames[item]
        image_path = os.path.join(self.root_dir, image_name)

        image = cv2.imread(image_path)
        image = image / 255.

        if self.transforms is not None:
            inputs = {"image": image}
            result = self.transforms(**inputs)
            image = result["image"]

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

def main(args):
    # Load model
    model = torch.load(args.model_path)

    # Setup TTA
    transforms = tta.Compose([
        tta.HorizontalFlip(),
    ])
    tta_model = tta.SegmentationTTAWrapper(model, transforms)

    # Setup dataset and dataloader
    tf = A.Resize(512, 512)
    test_dataset = XRayInferenceDataset(root_dir=args.test_image_root, transforms=tf)
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False
    )

    # Run inference
    rles, filename_and_class = test(tta_model, test_loader, thr=args.threshold)

    # Process results
    classes, filename = zip(*[x.split("_") for x in filename_and_class])
    image_name = [os.path.basename(f) for f in filename]

    # Save results
    df = pd.DataFrame({
        "image_name": image_name,
        "class": classes,
        "rle": rles,
    })
    df.to_csv(os.path("./output.csv", index=False))

if __name__ == "__main__":
    args = parse_args()
    main(args)
