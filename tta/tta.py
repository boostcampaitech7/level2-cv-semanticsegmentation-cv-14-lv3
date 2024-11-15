# python native
import os

# external library
import cv2
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import albumentations as A

# torch
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


import ttach as tta

# import configuration
from config import TEST_IMAGE_ROOT, CLASSES, CLASS2IND, IND2CLASS

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

        # to tenser will be done later
        image = image.transpose(2, 0, 1)    # make channel first
        image = torch.from_numpy(image).float()

        return image, image_name

def test(model, data_loader, thr=0.5):
    model = model.cuda()
    model.eval()

    rles = []
    filename_and_class = []
    inference_times = []

    with torch.no_grad():
        for step, (images, image_names) in tqdm(enumerate(data_loader), total=len(data_loader)):
            start_time = torch.cuda.Event(enable_timing=True)
            end_time = torch.cuda.Event(enable_timing=True)

            start_time.record()
            images = images.cuda()
            outputs = model(images)

            # restore original size
            outputs = F.interpolate(outputs, size=(2048, 2048), mode="bilinear")
            outputs = torch.sigmoid(outputs)
            outputs = (outputs > thr).detach().cpu().numpy()
            end_time.record()

            torch.cuda.synchronize()
            inference_times.append(start_time.elapsed_time(end_time))

            for output, image_name in zip(outputs, image_names):
                for c, segm in enumerate(output):
                    rle = encode_mask_to_rle(segm)
                    rles.append(rle)
                    filename_and_class.append(f"{IND2CLASS[c]}_{image_name}")

    avg_inference_time = np.mean(inference_times)
    print(f"Average inference time per batch: {avg_inference_time:.2f}ms")

    return rles, filename_and_class

if __name__ == "__main__":
    model = torch.load("../../donghwan/checkpoints/best_dice_0.9729.pt")

    # TTA transforms 설정
    transforms = tta.Compose([
        tta.HorizontalFlip(),
        tta.Scale(scales=[0.9, 1.0, 1.1])
    ])

    # TTA 모델 래핑
    tta_model = tta.SegmentationTTAWrapper(model, transforms)

    # 데이터셋 전처리 설정
    tf = A.Compose([
        A.Resize(512, 512)
    ])

    # 데이터셋 및 데이터로더 설정
    test_dataset = XRayInferenceDataset(transforms=tf)
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=8,
        shuffle=False,
        num_workers=min(os.cpu_count(), 4),
        pin_memory=True,
        drop_last=False
    )

    # 추론 실행
    rles, filename_and_class = test(tta_model, test_loader)

    # 결과 저장
    submission = pd.DataFrame({
        'image_name': filename_and_class,
        'rle': rles
    })
    submission.to_csv('output.csv', index=False)
