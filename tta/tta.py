import os
import numpy as np
import torch
from tqdm.auto import tqdm
import albumentations as A
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import ttach as tta
from dataset import XRayInferenceDataset

##### 구분선 #####

os.environ['KMP_DUPLICATE_LIB_OK']='True'
MODEL_PATH = "./best_model.pth"  # checkpoint path
SAVED_DIR = "./checkpoints"
TEST_IMAGE_ROOT = "../../data/test/images"  # test image path

##### 구분선 #####


# (!) Model path
model = torch.load(os.path(MODEL_PATH))

# Set up TTA
transforms = tta.Compose(
    [tta.HorizontalFlip()]
    )
tta_model = tta.SegmentationTTAWrapper(model, transforms)

# Get list of test image paths
pngs = [os.path.relpath(os.path.join(root, fname), start=TEST_IMAGE_ROOT)
       for root, _dirs, files in os.walk(TEST_IMAGE_ROOT)
       for fname in files
       if os.path.splitext(fname)[1].lower() == ".png"]

'''
def _get_image_files(self):
    return sorted({
        os.path.relpath(os.path.join(root, fname), start=self.image_root)
        for root, _dirs, files in os.walk(self.image_root)
        for fname in files
        if os.path.splitext(fname)[1].lower() == ".png"
    })

def _get_label_files(self):
    return sorted({
        os.path.relpath(os.path.join(root, fname), start=self.label_root)
        for root, _dirs, files in os.walk(self.label_root)
        for fname in files
        if os.path.splitext(fname)[1].lower() == ".json"
    })
'''

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


def test(model, data_loader, thr=0.5):
   model = model.cuda()
   model.eval()
   rles, filename_and_class = [], []
   with torch.no_grad():
       for images, image_names in tqdm(data_loader):
           images = images.cuda()
           # restore original size
           outputs = F.interpolate(model(images), size=(2048, 2048), mode="bilinear")
           outputs = torch.sigmoid(outputs)
           outputs = (outputs > thr).detach().cpu().numpy()
           for output, name in zip(outputs, image_names):
               for c, segm in enumerate(output):
                   rles.append(encode_mask_to_rle(segm)) # encode_mask_to_rle 수정 필요
                   filename_and_class.append(f"{IND2CLASS[c]}_{name}")
   return rles, filename_and_class

if __name__ == '__main__':
    tf = A.Resize(512, 512)
    test_dataset = XRayInferenceDataset(transforms=tf)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=2, drop_last=False)
    rles, filename_and_class = test(tta_model, test_loader)
