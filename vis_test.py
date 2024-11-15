import os
import cv2
import numpy as np
import pandas as pd
import streamlit as st
from skimage.draw import polygon

# 데이터 경로 설정
test_data_dir = '/data/ephemeral/home/data/test/DCM'
output_csv_path = '/Users/zangzoo/vscode/boostcamp_project/4_boostcamp_seg/output_U++_resnest50d_1024_CA.csv'  # output CSV 파일 경로

# CSV 파일 불러오기
def load_annotations(csv_path):
    annotations = pd.read_csv(csv_path)
    return annotations

# RLE를 이미지에 디코딩
def rle_to_mask(rle, shape):
    mask = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    rle_pairs = np.array(rle.split(), dtype=int).reshape(-1, 2)
    for start, length in rle_pairs:
        mask[start:start + length] = 1
    mask = mask.reshape(shape).T
    return mask

# 마스크 회전 및 반전 함수
def rotate_and_flip_mask(mask, angle=90, flip_horizontal=False):
    if angle == 90:
        mask = np.rot90(mask, k=3)  # 시계방향 90도 회전
    elif angle == 270:
        mask = np.rot90(mask, k=1)  # 반시계방향 90도 회전
    if flip_horizontal:
        mask = np.fliplr(mask)
    return mask

# 각 클래스에 대해 무작위 색상 생성
class_colors = {}
def get_class_color(label):
    if label not in class_colors:
        class_colors[label] = tuple(np.random.randint(0, 255, 3).tolist())
    return class_colors[label]

# Annotation 마스크를 이미지에 오버레이
def overlay_masks(image, image_name, annotations, visualize, opacity):
    if not visualize:
        return image
    image_annotations = annotations[annotations["image_name"] == image_name]
    for _, row in image_annotations.iterrows():
        class_name = row["class"]
        rle = row["rle"]
        color = get_class_color(class_name)
        
        # 마스크 생성
        mask = rle_to_mask(rle, image.shape[:2])
        mask = rotate_and_flip_mask(mask, angle=90, flip_horizontal=True)
        
        # 마스크 오버레이 (투명도 적용)
        overlay_color = np.array(color) * opacity
        image[mask == 1] = (image[mask == 1] * (1 - opacity) + overlay_color).astype(np.uint8)

    return image

# Streamlit 앱 구성
st.title("Test Hand Bone X-ray Visualization")

# Annotation 데이터 불러오기
annotations = load_annotations(output_csv_path)

# 마스크 시각화 on/off 설정
visualize = st.sidebar.checkbox("Show Annotations (Masks)", value=True)

# 마스크 오버레이 투명도 설정 (슬라이더)
opacity = st.sidebar.slider("Mask Overlay Opacity", 0.0, 1.0, 0.5)

# ID별 폴더에서 ID 목록 불러오기 및 오름차순 정렬
id_folders = sorted([d for d in os.listdir(test_data_dir) if os.path.isdir(os.path.join(test_data_dir, d))])

# ID 선택 상태 관리
if "selected_id_index" not in st.session_state:
    st.session_state.selected_id_index = 0

# 이전/다음 버튼으로 ID 선택 조정
col1, col2 = st.columns([1, 1])
with col1:
    if st.button("⬅️ Previous") and st.session_state.selected_id_index > 0:
        st.session_state.selected_id_index -= 1
with col2:
    if st.button("➡️ Next") and st.session_state.selected_id_index < len(id_folders) - 1:
        st.session_state.selected_id_index += 1

# 현재 선택된 ID 업데이트 및 표시
selected_id = id_folders[st.session_state.selected_id_index]
st.selectbox("Select ID Folder", id_folders, index=st.session_state.selected_id_index, key="selectbox_id")

# 선택한 ID 폴더에서 이미지 파일 선택
if selected_id:
    image_dir = os.path.join(test_data_dir, selected_id)
    image_files = [f for f in os.listdir(image_dir) if f.endswith('.png') or f.endswith('.jpg')]
    
    # CSV 파일에 있는 이미지 이름과 일치하는 파일만 필터링
    matched_images = [img for img in image_files if img in annotations['image_name'].values]
    
    # 왼손, 오른손 이미지를 순서대로 표시
    if matched_images:
        cols = st.columns(2)
        
        for idx, image_file in enumerate(matched_images):
            image_path = os.path.join(image_dir, image_file)
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # 마스크 시각화 여부 및 투명도 설정에 따른 처리
            image = overlay_masks(image, image_file, annotations, visualize, opacity)
            
            # 왼손/오른손 구분하여 이미지 표시
            if idx < len(cols):
                cols[idx].image(image, caption=f"{'Left Hand' if idx == 0 else 'Right Hand'} ({image_file})", use_column_width=True)
    else:
        st.write("No matching images found in the selected ID folder.")