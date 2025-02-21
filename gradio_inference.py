'''
- Step1. Load the ckpt
- Step2. Masking to the image (predict)
- Step3. Encode (rle -> mask)
- Step4. Return the image (gradio)
'''

import os, cv2, torch
import streamlit
import numpy as np
import gradio as gr
import albumentations as A
import torch.nn.functional as F
from torchvision.models.segmentation import fcn_resnet50

# Import from other files
from dataset import CLASSES, PALETTE, IND2CLASS, XRayInferenceDataset
from functions import encode_mask_to_rle, decode_rle_to_mask
from inference import load_model
from tools.streamlit.vis_test import get_class_color, overlay_masks


def test(model, image, thr=0.5):
    # Preprocess
    transform = A.Compose([
        A.Resize(512, 512),
    ])

    # Convert to RGB if needed
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    transformed = transform(image=image)['image']

    # Normalize and convert to tensor
    input_tensor = torch.from_numpy(transformed).permute(2, 0, 1).float().unsqueeze(0)

    with torch.no_grad():
        outputs = model(input_tensor)['out']
        outputs = F.interpolate(outputs, size=(2048, 2048), mode="bilinear")
        outputs = torch.sigmoid(outputs)
        outputs = (outputs > thr).detach().numpy()

    return outputs[0]

def segmentation_inference(image, model_path='./gradio_sample/checkpoints/sample_checkpoint.pt'):
    """ main script file"""
    # Load model
    model = load_model(model_path)

    # Process segmentation
    masks = test(model, image)

    # Overlay masks
    result = overlay_masks(image, masks)

    return result

def main():
    iface = gr.Interface(
        fn=segmentation_inference,
        inputs=[
            gr.Image(type="numpy", label="Input X-ray Image"),
        ],
        outputs=[
            gr.Image(type="numpy", label="Segmentation Result")
        ],
        title="X-ray Image Segmentation",
        description="Upload an X-ray image for bone segmentation"
    )

    iface.launch(share=True)

if __name__ == "__main__":
    main()
