import os
import gradio as gr
import torch
import torch.nn.functional as F
import numpy as np
import cv2
import albumentations as A
import pandas as pd
from torchvision.models.segmentation import fcn_resnet50
from ultralytics import YOLO
import matplotlib.pyplot as plt

# Import from your existing script
from dataset import IND2CLASS, XRayInferenceDataset
from functions import encode_mask_to_rle

def load_model(model_path, num_classes=29):
    # Same model loading function as in the original script
    model = fcn_resnet50(weights=None)
    model.classifier[4] = torch.nn.Conv2d(512, num_classes, kernel_size=(1, 1), stride=(1, 1))

    checkpoint = torch.load(model_path)
    if isinstance(checkpoint, dict):
        state_dict = checkpoint.get('model_state_dict', checkpoint.get('state_dict', checkpoint))
    else:
        state_dict = checkpoint

    # Remove 'module.' prefix if present
    if any(key.startswith('module.') for key in state_dict.keys()):
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

    model.load_state_dict(state_dict, strict=False)
    return model

def prepare_image(image, img_size=1024):
    """Prepare image for inference"""
    # Resize and convert to appropriate format
    tf = A.Compose([A.Resize(img_size, img_size)])
    transformed = tf(image=image)
    processed_image = transformed['image']

    # Convert to tensor
    image_tensor = torch.from_numpy(processed_image).permute(2, 0, 1).float() / 255.0
    image_tensor = image_tensor.unsqueeze(0)  # Add batch dimension

    return image_tensor

def inference(model, image_tensor, threshold=0.4):
    """Perform inference on the image"""
    model = model.cuda()
    model.eval()

    with torch.no_grad():
        image_tensor = image_tensor.cuda()
        outputs = model(image_tensor)['out']

        # Resize outputs to original image size
        outputs = F.interpolate(outputs, size=(2048, 2048), mode="bilinear")
        outputs = torch.sigmoid(outputs)
        outputs = (outputs > threshold).detach().cpu().numpy()

    return outputs

def visualize_segmentation(original_image, outputs):
    """Create visualization of segmentation masks"""
    plt.figure(figsize=(15, 5))

    # Original image
    plt.subplot(1, len(outputs[0]) + 1, 1)
    plt.imshow(original_image)
    plt.title('Original Image')
    plt.axis('off')

    # Segmentation masks
    for i, mask in enumerate(outputs[0]):
        plt.subplot(1, len(outputs[0]) + 1, i + 2)
        overlay = original_image.copy()
        overlay[mask] = [255, 0, 0]  # Red mask overlay
        plt.imshow(overlay)
        plt.title(f'Mask: {IND2CLASS[i]}')
        plt.axis('off')

    plt.tight_layout()

    # Save the plot to a temporary file
    output_path = 'segmentation_result.png'
    plt.savefig(output_path)
    plt.close()

    return output_path

def xray_segmentation(input_image, model_path, threshold):
    """Main function to process input image"""
    # Load model
    model = load_model(model_path)

    # Prepare image
    image_tensor = prepare_image(input_image)

    # Perform inference
    outputs = inference(model, image_tensor, threshold)

    # Visualize results
    result_path = visualize_segmentation(input_image, outputs)

    return result_path

def create_gradio_interface():
    """Create Gradio interface for X-ray segmentation"""
    demo = gr.Interface(
        fn=xray_segmentation,
        inputs=[
            gr.Image(type="numpy", label="Upload X-ray Image"),
            gr.Textbox(
                value="./checkpoints/best_model.pt",
                label="Model Path"
            ),
            gr.Slider(
                minimum=0.1,
                maximum=1.0,
                value=0.4,
                step=0.1,
                label="Threshold"
            )
        ],
        outputs=gr.Image(type="filepath", label="Segmentation Result"),
        title="X-ray Image Segmentation",
        description="Upload an X-ray image to perform multi-class segmentation"
    )

    return demo

def main():
    demo = create_gradio_interface()
    demo.launch(server_name="0.0.0.0", server_port=7860)

if __name__ == "__main__":
    main()
