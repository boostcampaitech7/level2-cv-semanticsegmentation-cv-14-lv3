import matplotlib.pyplot as plt
import numpy as np


# 색상 리스트
PALETTE = [
    (220, 20, 60), (119, 11, 32), (0, 0, 142), (0, 0, 230), (106, 0, 228),
    (0, 60, 100), (0, 80, 100), (0, 0, 70), (0, 0, 192), (250, 170, 30),
    (100, 170, 30), (220, 220, 0), (175, 116, 175), (250, 0, 30), (165, 42, 42),
    (255, 77, 255), (0, 226, 252), (182, 182, 255), (0, 82, 0), (120, 166, 157),
    (110, 76, 0), (174, 57, 255), (199, 100, 0), (72, 0, 118), (255, 179, 240),
    (0, 125, 92), (209, 0, 151), (188, 208, 182), (0, 220, 176),
]

def label2rgb(label):
    image_size = label.shape[1:] + (3, )
    image = np.zeros(image_size, dtype=np.uint8)

    for i, class_label in enumerate(label):
        image[class_label == 1] = PALETTE[i]

    return image

def visualize_prediction(image, pred_mask, true_mask=None):
    """
    예측 결과를 시각화하는 함수

    Args:
        image: 원본 이미지 (H, W, 3)
        pred_mask: 예측 마스크 (C, H, W)
        true_mask: 실제 마스크 (C, H, W), 옵션
    """
    if true_mask is None:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 12))
        ax1.imshow(image)
        ax1.set_title('Original Image')
        ax2.imshow(label2rgb(pred_mask))
        ax2.set_title('Prediction')
    else:
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(36, 12))
        ax1.imshow(image)
        ax1.set_title('Original Image')
        ax2.imshow(label2rgb(pred_mask))
        ax2.set_title('Prediction')
        ax3.imshow(label2rgb(true_mask))
        ax3.set_title('Ground Truth')

    plt.tight_layout()
    return fig