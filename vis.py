import os
import cv2
import json
import numpy as np
import streamlit as st
import pandas as pd

# 데이터 경로 설정
data_dir = '/Users/zangzoo/vscode/boostcamp_project/4_boostcamp_seg/data/train/DCM/'
meta_data_path = '/Users/zangzoo/vscode/boostcamp_project/4_boostcamp_seg/data/meta_data.xlsx'  # meta_data 파일 경로
outputs_json_dir = '/Users/zangzoo/vscode/boostcamp_project/4_boostcamp_seg/data/train/outputs_json/'

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

# 각 클래스에 대해 무작위 색상 생성
class_colors = {}
def get_class_color(label):
    if label not in class_colors:
        class_colors[label] = tuple(np.random.randint(0, 255, 3).tolist())
    return class_colors[label]

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

# Streamlit 앱 구성
st.title("Hand Bone X-ray Visualization")
st.sidebar.title("View Options")

# "Show All" 또는 "Apply Filters" 선택
view_option = st.sidebar.radio("Choose View Option", ["Show All", "Apply Filters"])

# 시각화 on/off 버튼
show_polygons = st.sidebar.checkbox("Show Annotations (Polygons)", value=False)

# 전체 이미지를 보고 싶은 경우
if view_option == "Show All":
    if "selected_person_id" not in st.session_state:
        st.session_state.selected_person_id = available_ids[0]

    col1, col2 = st.sidebar.columns([1, 1])
    with col1:
        if st.button("⬅️ Previous"):
            current_index = available_ids.index(st.session_state.selected_person_id)
            if current_index > 0:
                st.session_state.selected_person_id = available_ids[current_index - 1]
    with col2:
        if st.button("➡️ Next"):
            current_index = available_ids.index(st.session_state.selected_person_id)
            if current_index < len(available_ids) - 1:
                st.session_state.selected_person_id = available_ids[current_index + 1]

    selected_person_id = st.sidebar.selectbox("Select Person ID", available_ids, index=available_ids.index(st.session_state.selected_person_id))

    if selected_person_id in images_by_id:
        st.header(f"Person ID: {selected_person_id}")
        images = images_by_id[selected_person_id]
        cols = st.columns(2)
        
        for idx, image_path in enumerate(images):
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # 폴리곤 표시 여부
            if show_polygons:
                json_path = os.path.join(outputs_json_dir, f'ID{selected_person_id:03}', os.path.basename(image_path).replace('.jpg', '.json').replace('.png', '.json'))
                if os.path.exists(json_path):
                    image = draw_polygons_on_image(image, json_path)
            
            if idx < len(cols):
                cols[idx].image(image, caption=f"{'Left Hand' if idx == 0 else 'Right Hand'}", use_column_width=True)

# 필터링을 통해 이미지를 보고 싶은 경우
elif view_option == "Apply Filters":
    # 필터 설정
    age_ranges = ["All", "0-9", "10-19", "20-29", "30-39", "40-49", "50+"]
    selected_age_range = st.sidebar.selectbox("Select Age Range", age_ranges)

    gender_options = ["All", "남", "여"]
    selected_gender = st.sidebar.selectbox("Select Gender", gender_options)

    weight_categories = ["All", "<50kg", "50-59kg", "60-69kg", "70-79kg", "80kg+"]
    selected_weight_category = st.sidebar.selectbox("Select Weight Category", weight_categories)

    height_categories = ["All", "<150cm", "150-159cm", "160-169cm", "170-179cm", "180cm+"]
    selected_height_category = st.sidebar.selectbox("Select Height Category", height_categories)

    # 필터링 조건 생성
    filtered_meta_data = meta_data
    if selected_age_range != "All":
        min_age, max_age = map(int, selected_age_range.split('-')) if '-' in selected_age_range else (50, 100)
        filtered_meta_data = filtered_meta_data[(filtered_meta_data["나이"] >= min_age) & (filtered_meta_data["나이"] <= max_age)]
    if selected_gender != "All":
        filtered_meta_data = filtered_meta_data[filtered_meta_data["성별"] == selected_gender]
    if selected_weight_category != "All":
        min_weight, max_weight = {
            "<50kg": (0, 49), "50-59kg": (50, 59), "60-69kg": (60, 69), "70-79kg": (70, 79), "80kg+": (80, 300)
        }[selected_weight_category]
        filtered_meta_data = filtered_meta_data[(filtered_meta_data["체중(몸무게)"] >= min_weight) & (filtered_meta_data["체중(몸무게)"] <= max_weight)]
    if selected_height_category != "All":
        min_height, max_height = {
            "<150cm": (0, 149), "150-159cm": (150, 159), "160-169cm": (160, 169), "170-179cm": (170, 179), "180cm+": (180, 250)
        }[selected_height_category]
        filtered_meta_data = filtered_meta_data[(filtered_meta_data["키(신장)"] >= min_height) & (filtered_meta_data["키(신장)"] <= max_height)]

    filtered_ids = filtered_meta_data.index.tolist()

    # 필터링된 ID 목록 표시
    if filtered_ids:
        st.subheader(f"Images for Filtered Group")
        
        for person_id in filtered_ids:
            if person_id in images_by_id:
                st.write(f"Person ID: {person_id}")
                images = images_by_id[person_id]
                cols = st.columns(2)
                for idx, image_path in enumerate(images):
                    image = cv2.imread(image_path)
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                    if show_polygons:
                        json_path = os.path.join(outputs_json_dir, f'ID{person_id:03}', os.path.basename(image_path).replace('.jpg', '.json').replace('.png', '.json'))
                        if os.path.exists(json_path):
                            image = draw_polygons_on_image(image, json_path)
                    
                    if idx < len(cols):
                        cols[idx].image(image, caption=f"{'Left Hand' if idx == 0 else 'Right Hand'}", use_column_width=True)
    else:
        st.write("No images found for the selected group.")