import os, sys
import cv2
import numpy as np
import pandas as pd
import streamlit as st
import argparse

# Load CLASSSES and PAELLETTE
from dataset import CLASSES, PALETTE

'''
CLASSES = [
    'Trapezium', 'Trapezoid', 'Triquetrum', 'Pisiform'
]

PALETTE = [
    (120, 166, 157), (110, 76, 0), (0, 125, 92), (209, 0, 151)
]
'''

# 클래스-색상 매핑 딕셔너리 생성
CLASS_COLORS = dict(zip(CLASSES, PALETTE))

def parse_args():
    parser = argparse.ArgumentParser(description='X-ray Image Viewer with RLE Annotation')
    parser.add_argument('--data_dir',
                       type=str,
                       default='/data/ephemeral/home/data/test/DCM',
                       help='Dataset path')
    parser.add_argument('--csv_path',
                       type=str,
                       default='output.csv',
                       help='inference.py를 통해서 생성한 CSV 파일을 입력해주세요.')

    try:
        streamlit_args = ['streamlit', 'run']
        user_args = [arg for arg in os.sys.argv[1:] if not any(s in arg for s in streamlit_args)]
        args = parser.parse_args(user_args)
    except:
        args = parser.parse_args([])

    return args

def load_annotations(csv_path):
    try:
        annotations = pd.read_csv(csv_path)
        if 'rle' not in annotations.columns:
            annotations['rle'] = ''
        return annotations
    except FileNotFoundError:
        st.error(f"CSV file not found: {csv_path}")
        return pd.DataFrame(columns=["image_name", "class", "rle"])

def rle_to_mask(rle, shape):
    mask = np.zeros(shape[0] * shape[1], dtype=np.uint8)

    if pd.isna(rle) or not rle.strip():
        return mask.reshape(shape).T

    try:
        rle_pairs = np.array([int(x) for x in rle.split()]).reshape(-1, 2)
        for start, length in rle_pairs:
            mask[start:start + length] = 1
    except (ValueError, TypeError) as e:
        st.warning(f"Invalid RLE format: {str(e)}")
        return mask.reshape(shape).T

    return mask.reshape(shape).T

def rotate_and_flip_mask(mask, angle=90, flip_horizontal=False):
    if angle == 90:
        mask = np.rot90(mask, k=3)
    elif angle == 270:
        mask = np.rot90(mask, k=1)
    if flip_horizontal:
        mask = np.fliplr(mask)
    return mask

def get_class_color(label):
    # 지정된 클래스에 대해서만 색상 반환
    return CLASS_COLORS.get(label, None)

def overlay_masks(image, image_name, annotations, visualize, opacity):
    if not visualize or annotations.empty:
        return image

    image_annotations = annotations[annotations["image_name"] == image_name]

    # 지정된 클래스만 필터링
    image_annotations = image_annotations[image_annotations["class"].isin(CLASSES)]

    for _, row in image_annotations.iterrows():
        class_name = row["class"]
        rle = row["rle"]

        if pd.isna(rle) or not rle.strip():
            continue

        color = get_class_color(class_name)
        if color is None:  # 지정된 클래스가 아닌 경우 건너뛰기
            continue

        try:
            mask = rle_to_mask(rle, image.shape[:2])
            mask = rotate_and_flip_mask(mask, angle=90, flip_horizontal=True)

            overlay_color = np.array(color) * opacity
            image[mask == 1] = (image[mask == 1] * (1 - opacity) + overlay_color).astype(np.uint8)
        except Exception as e:
            st.warning(f"Error processing mask for {image_name}: {str(e)}")
            continue

    return image

def main():
    args = parse_args()
    st.title("X-ray Image Viewer")

    with st.expander("Current Settings"):
        st.write(f"Data Directory: {args.data_dir}")
        st.write(f"CSV File Path: {args.csv_path}")

    annotations = load_annotations(args.csv_path)

    if not annotations.empty:
        with st.expander("Show CSV Data Info"):
            st.write("Total number of annotations:", len(annotations))
            st.write("Number of visualized classes:", len(CLASSES))
            visualized_annotations = annotations[annotations['class'].isin(CLASSES)]
            st.write("Number of visualized annotations:", len(visualized_annotations))
            st.write("Preview of visualized annotations:")
            st.dataframe(visualized_annotations.head())

    visualize = st.sidebar.checkbox("Show Annotations (Masks)", value=True)
    opacity = st.sidebar.slider("Mask Overlay Opacity", 0.0, 1.0, 0.5)

    # 색상 범례 표시
    if visualize:
        st.sidebar.markdown("### Visualized Classes")
        for class_name, color in CLASS_COLORS.items():
            st.sidebar.markdown(
                f'<div style="background-color: rgb{color}; padding: 5px; margin: 2px; color: white;">{class_name}</div>',
                unsafe_allow_html=True
            )

    try:
        id_folders = sorted([d for d in os.listdir(args.data_dir) if os.path.isdir(os.path.join(args.data_dir, d))])
    except FileNotFoundError:
        st.error(f"Directory not found: {args.data_dir}")
        id_folders = []

    if id_folders:
        if "selected_id_index" not in st.session_state:
            st.session_state.selected_id_index = 0

        selected_id = id_folders[st.session_state.selected_id_index]
        st.selectbox("Select ID Folder", id_folders, index=st.session_state.selected_id_index, key="selectbox_id")

        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button("⬅️ Previous") and st.session_state.selected_id_index > 0:
                st.session_state.selected_id_index -= 1
        with col2:
            if st.button("➡️ Next") and st.session_state.selected_id_index < len(id_folders) - 1:
                st.session_state.selected_id_index += 1

        if selected_id:
            image_dir = os.path.join(args.data_dir, selected_id)
            image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.png') or f.endswith('.jpg')])

            if image_files:
                cols = st.columns(2)
                for idx, image_file in enumerate(image_files[:2]):
                    image_path = os.path.join(image_dir, image_file)
                    try:
                        image = cv2.imread(image_path)
                        if image is not None:
                            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                            image = overlay_masks(image, image_file, annotations, visualize, opacity)
                            cols[idx].image(image, caption=f"Image {idx+1} ({image_file})", use_column_width=True)
                        else:
                            cols[idx].error(f"Failed to load image: {image_file}")
                    except Exception as e:
                        cols[idx].error(f"Error processing image {image_file}: {str(e)}")
            else:
                st.write("No images found in the selected ID folder.")
    else:
        st.error("No ID folders found in the specified directory.")

if __name__ == "__main__":
    main()
