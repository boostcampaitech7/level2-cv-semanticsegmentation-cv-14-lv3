import os
import cv2
import json
import numpy as np
import streamlit as st
import pandas as pd
from albumentations import (
    Compose, ElasticTransform, Sharpen, CLAHE, Rotate, Emboss, GridDistortion,
    RandomBrightnessContrast, HorizontalFlip, Normalize
)

# 데이터 경로 설정
data_dir = '/data/train/DCM'
meta_data_path = '/data/meta_data.xlsx'  # meta_data 파일 경로
outputs_json_dir = '/data/train/outputs_json'

# 사람별로 왼손, 오른손 이미지 불러오기
def load_images_by_id(data_dir):
    images = {}
    person_ids = sorted([person_id for person_id in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, person_id))])
    for person_id in person_ids:
        person_dir = os.path.join(data_dir, person_id)
        id_key = int(person_id[2:])  # 'ID001'에서 숫자만 추출하여 정수로 변환
        images[id_key] = []
        for image_file in os.listdir(person_dir):
            if image_file.endswith('.png') or image_file.endswith('.jpg'):
                image_path = os.path.join(person_dir, image_file)
                images[id_key].append(image_path)
    return images

# 실제 존재하는 ID 목록 불러오기
images_by_id = load_images_by_id(data_dir)
available_ids = sorted(images_by_id.keys())

# Meta data 불러오기 및 ID 매핑
meta_data = pd.read_excel(meta_data_path)
meta_data.set_index("ID", inplace=True)

# PALETTE 정의
PALETTE = [
    (220, 20, 60), (119, 11, 32), (0, 0, 142), (0, 0, 230), (106, 0, 228),
    (0, 60, 100), (0, 80, 100), (0, 0, 70), (0, 0, 192), (250, 170, 30),
    (100, 170, 30), (220, 220, 0), (175, 116, 175), (250, 0, 30), (165, 42, 42),
    (255, 77, 255), (0, 226, 252), (182, 182, 255), (0, 82, 0), (120, 166, 157),
    (110, 76, 0), (174, 57, 255), (199, 100, 0), (72, 0, 118), (255, 179, 240),
    (0, 125, 92), (209, 0, 151), (188, 208, 182), (0, 220, 176),
]

# 클래스 라벨 목록 (예시로 숫자 라벨 사용)
label_list = []  # 실제 클래스 라벨 목록으로 대체해야 함

# 클래스 라벨과 색상 매핑
label_to_color = {}

def get_class_color(label):
    if label not in label_to_color:
        index = len(label_to_color) % len(PALETTE)
        label_to_color[label] = PALETTE[index]
    return label_to_color[label]

# JSON 파일에서 폴리곤 그리기 함수
def draw_polygons_on_image(image, json_path):
    with open(json_path, 'r') as f:
        annotations = json.load(f)['annotations']
    for annotation in annotations:
        label = annotation['label']
        points = np.array(annotation['points'], dtype=np.int32)
        color = get_class_color(label)
        
        # 폴리곤 그리기
        cv2.polylines(image, [points], isClosed=True, color=color, thickness=2)
        cv2.fillPoly(image, [points], color=color)

    return image

# 데이터 증강기법 적용 함수
def apply_augmentation(image, params):
    augmentations = []
    
    # 각 증강 기법의 체크박스와 슬라이더 설정
    if params['ElasticTransform_enabled']:
        augmentations.append(ElasticTransform(alpha=params['ElasticTransform_alpha'], sigma=params['ElasticTransform_sigma'], 
                                              alpha_affine=params['ElasticTransform_alpha_affine'], p=1.0))
    if params['Sharpen_enabled']:
        augmentations.append(Sharpen(alpha=params['Sharpen_alpha'], lightness=params['Sharpen_lightness'], p=1.0))
    if params['CLAHE_enabled']:
        augmentations.append(CLAHE(clip_limit=params['CLAHE_clip_limit'], tile_grid_size=(params['CLAHE_tile_grid_size'], params['CLAHE_tile_grid_size']), p=1.0))
    if params['Rotate_enabled']:
        augmentations.append(Rotate(limit=params['Rotate_limit'], p=1.0))
    if params['Emboss_enabled']:
        augmentations.append(Emboss(alpha=params['Emboss_alpha'], strength=params['Emboss_strength'], p=1.0))
    if params['GridDistort_enabled']:
        augmentations.append(GridDistortion(p=1.0))
    if params['RandomBrightnessContrast_enabled']:
        augmentations.append(RandomBrightnessContrast(brightness_limit=params['RandomBrightnessContrast_brightness_limit'], 
                                                      contrast_limit=params['RandomBrightnessContrast_contrast_limit'], p=1.0))
    if params['HorizontalFlip_enabled']:
        augmentations.append(HorizontalFlip(p=1.0))
    
    # Normalize on/off
    if params['Normalize_enabled']:
        augmentations.append(Normalize())
    
    # 모든 증강 기법이 비활성화된 경우 원본 이미지 반환
    if not augmentations:
        return image
    
    aug = Compose(augmentations)
    augmented = aug(image=image)
    return augmented['image']

# Streamlit 앱 구성
st.title("Hand Bone X-ray Visualization with Augmentation and Masking")
st.sidebar.title("Settings")
# 이미지 표시
# 세션 상태 초기화
if 'selected_person_index' not in st.session_state:
    st.session_state.selected_person_index = 0

# 이전, 다음 버튼 생성
col1, col2, col3 = st.sidebar.columns([1, 2, 1])
with col1:
    if st.button("⬅️ 이전"):
        if st.session_state.selected_person_index > 0:
            st.session_state.selected_person_index -= 1
with col3:
    if st.button("➡️ 다음"):
        if st.session_state.selected_person_index < len(available_ids) - 1:
            st.session_state.selected_person_index += 1

# 현재 선택된 Person ID 업데이트
selected_person_id = available_ids[st.session_state.selected_person_index]

# 선택박스 업데이트 (선택된 ID에 맞게)
selected_person_id = st.sidebar.selectbox("Select Person ID", available_ids, index=st.session_state.selected_person_index)
st.header(f"Person ID: {selected_person_id}")
images = images_by_id[selected_person_id]
# 증강 파라미터 설정 슬라이드바와 체크박스 설정
augmentation_params = {
    "ElasticTransform_enabled": st.sidebar.checkbox("Enable ElasticTransform", value=True),
    "ElasticTransform_alpha": st.sidebar.slider("ElasticTransform Alpha", 0.0, 50.0, 1.0),
    "ElasticTransform_sigma": st.sidebar.slider("ElasticTransform Sigma", 0.0, 10.0, 1.0),
    "ElasticTransform_alpha_affine": st.sidebar.slider("ElasticTransform Alpha Affine", 0.0, 1.0, 0.1),
    
    "Sharpen_enabled": st.sidebar.checkbox("Enable Sharpen", value=True),
    "Sharpen_alpha": st.sidebar.slider("Sharpen Alpha", 0.0, 1.0, 0.5),
    "Sharpen_lightness": st.sidebar.slider("Sharpen Lightness", 0.0, 2.0, 1.0),
    
    "CLAHE_enabled": st.sidebar.checkbox("Enable CLAHE", value=True),
    "CLAHE_clip_limit": st.sidebar.slider("CLAHE Clip Limit", 1.0, 10.0, 2.0),
    "CLAHE_tile_grid_size": st.sidebar.slider("CLAHE Tile Grid Size", 1, 8, 4),
    
    "Rotate_enabled": st.sidebar.checkbox("Enable Rotate", value=True),
    "Rotate_limit": st.sidebar.slider("Rotate Limit", -90, 90, 45),
    
    "Emboss_enabled": st.sidebar.checkbox("Enable Emboss", value=True),
    "Emboss_alpha": st.sidebar.slider("Emboss Alpha", 0.0, 1.0, 0.5),
    "Emboss_strength": st.sidebar.slider("Emboss Strength", 0.0, 1.0, 0.5),
    
    "GridDistort_enabled": st.sidebar.checkbox("Enable GridDistort", value=True),
    
    "RandomBrightnessContrast_enabled": st.sidebar.checkbox("Enable RandomBrightnessContrast", value=True),
    "RandomBrightnessContrast_brightness_limit": st.sidebar.slider("Brightness Limit", -1.0, 1.0, 0.2),
    "RandomBrightnessContrast_contrast_limit": st.sidebar.slider("Contrast Limit", -1.0, 1.0, 0.2),
    
    "HorizontalFlip_enabled": st.sidebar.checkbox("Enable HorizontalFlip", value=True),
    
    "Normalize_enabled": st.sidebar.checkbox("Enable Normalize", value=True)
}

# 마스킹 표시 여부 설정
show_polygons = st.sidebar.checkbox("Show Annotations (Polygons)", value=False)

# 이미지를 한 줄에 하나씩 출력
for idx, image_path in enumerate(images):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 원본 이미지 복사
    original_image = image.copy()

    # 폴리곤 표시 여부
    if show_polygons:
        json_path = os.path.join(outputs_json_dir, f'ID{selected_person_id:03}', os.path.basename(image_path).replace('.jpg', '.json').replace('.png', '.json'))
        if os.path.exists(json_path):
            image = draw_polygons_on_image(image, json_path)
            original_image = draw_polygons_on_image(original_image, json_path)

    augmented_image = apply_augmentation(image, augmentation_params)
    
    if augmented_image.dtype != np.uint8:
        augmented_image = np.clip((augmented_image * 255), 0, 255).astype(np.uint8)
    
    # 이미지 출력
    st.image([original_image, augmented_image], caption=["Original Image", "Augmented Image"], width=300)
