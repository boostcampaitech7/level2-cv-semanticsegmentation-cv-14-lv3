import os
import gradio as gr
import torch
import cv2
import numpy as np
import pandas as pd
import albumentations as A
from PIL import Image

from torchvision.models.segmentation import fcn_resnet50
from functions import encode_mask_to_rle
from dataset import IND2CLASS

def load_model(model_path, num_classes=29):
    print(f"Loading model from {model_path}")
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'), weights_only=True)

    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
    elif hasattr(checkpoint, 'state_dict'):
        state_dict = checkpoint.state_dict()
    else:
        state_dict = checkpoint

    if not isinstance(state_dict, dict):
        return checkpoint

    try:
        model = fcn_resnet50(weights=None)
        model.classifier[4] = torch.nn.Conv2d(512, num_classes, kernel_size=(1, 1), stride=(1, 1))
        model.load_state_dict(state_dict, strict=False)
    except Exception as e:
        print(f"Failed to load with FCN: {e}")
        try:
            model = checkpoint
        except:
            raise ValueError("Could not load the model")

    print("\nModel loaded successfully")
    return model

def inference_single_image(image, model, threshold=0.4):
    # Preprocessing
    tf = A.Compose([
        A.Resize(1024, 1024),
    ])

    # Convert input to numpy if it's a PIL Image
    if isinstance(image, Image.Image):
        image = np.array(image)

    # Transform image
    transformed = tf(image=image)['image']

    # Convert to tensor
    input_tensor = torch.from_numpy(transformed).permute(2, 0, 1).float().unsqueeze(0)

    # Inference
    model.eval()
    with torch.no_grad():
        outputs = model(input_tensor)['out']
        outputs = torch.sigmoid(outputs)
        outputs = (outputs > threshold).detach().cpu().numpy()

    # Visualization
    class_masks = []
    for c, segm in enumerate(outputs[0]):
        # Resize back to original image size
        resized_mask = cv2.resize(segm.astype(np.uint8) * 255,
                                  (image.shape[1], image.shape[0]),
                                  interpolation=cv2.INTER_NEAREST)
        class_masks.append((IND2CLASS[c], resized_mask))

    return class_masks

def create_segmentation_overlay(original_image, masks):
    # Convert original image to RGB if it's grayscale
    if len(original_image.shape) == 2:
        original_image = cv2.cvtColor(original_image, cv2.COLOR_GRAY2RGB)

    # Create overlay with different colors for each class
    overlay = original_image.copy()
    colors = [
        (255, 0, 0),   # Red
        (0, 255, 0),   # Green
        (0, 0, 255),   # Blue
        (255, 255, 0), # Yellow
        (255, 0, 255), # Magenta
    ]

    for idx, (class_name, mask) in enumerate(masks):
        color = colors[idx % len(colors)]
        overlay[mask > 0] = color

    # Blend original and overlay
    blended = cv2.addWeighted(original_image, 0.5, overlay, 0.5, 0)
    return blended

def gradio_inference(image, threshold):
    # Load model (you may want to load this once globally)
    model_path = './gradio_sample/checkpoints/sample_checkpoint.pt'
    model = load_model(model_path)

    # Perform inference
    masks = inference_single_image(image, model, threshold)

    # Create overlay
    overlay = create_segmentation_overlay(np.array(image), masks)

    # Prepare results for display
    result_images = [overlay]
    result_labels = [', '.join([name for name, _ in masks])]  # Convert list of class names to single string

    # Add individual class masks
    for class_name, mask in masks:
        result_images.append(mask)
        result_labels.append(f'{class_name} Mask')

    return result_images, result_labels

def launch_gradio():
    iface = gr.Interface(
        fn=gradio_inference,
        inputs=[
            gr.Image(type='pil', label='Input X-Ray Image'),
            gr.Slider(minimum=0, maximum=1, value=0.4, label='Segmentation Threshold')
        ],
        outputs=[
            gr.Gallery(label='Segmentation Results'),
            gr.Label(label='Result Descriptions')
        ],
        title='X-Ray Bone Segmentation',
        description='Upload an X-Ray image to perform semantic segmentation.',
        examples=[
            ['./example_xray1.jpg', 0.4],
            ['./example_xray2.jpg', 0.4]
        ]
    )

    return iface

def main():
    iface = launch_gradio()
    iface.launch()

if __name__ == '__main__':
    main()
