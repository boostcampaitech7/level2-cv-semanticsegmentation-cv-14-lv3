import os
import cv2
import json
import numpy as np
import torch
from torch.utils.data import Dataset

import pandas as pd

# 상수 정의
CLASSES = [
    'finger-1', 'finger-2', 'finger-3', 'finger-4', 'finger-5',
    'finger-6', 'finger-7', 'finger-8', 'finger-9', 'finger-10',
    'finger-11', 'finger-12', 'finger-13', 'finger-14', 'finger-15',
    'finger-16', 'finger-17', 'finger-18', 'finger-19', 'Trapezium',
    'Trapezoid', 'Capitate', 'Hamate', 'Scaphoid', 'Lunate',
    'Triquetrum', 'Pisiform', 'Radius', 'Ulna',
]

PALETTE = [
    (220, 20, 60), (119, 11, 32), (0, 0, 142), (0, 0, 230), (106, 0, 228),
    (0, 60, 100), (0, 80, 100), (0, 0, 70), (0, 0, 192), (250, 170, 30),
    (100, 170, 30), (220, 220, 0), (175, 116, 175), (250, 0, 30), (165, 42, 42),
    (255, 77, 255), (0, 226, 252), (182, 182, 255), (0, 82, 0), (120, 166, 157),
    (110, 76, 0), (174, 57, 255), (199, 100, 0), (72, 0, 118), (255, 179, 240),
    (0, 125, 92), (209, 0, 151), (188, 208, 182), (0, 220, 176),
]

CLASS2IND = {v: i for i, v in enumerate(CLASSES)}
IND2CLASS = {v: k for k, v in CLASS2IND.items()}

class XRayDataset(Dataset):
    def __init__(self, image_root, label_root, is_train=True, transforms=None,
                 use_cv=False, n_splits=5, fold=0):
        self.image_root = image_root
        self.label_root = label_root
        self.transforms = transforms
        self.use_cv = use_cv
        self.n_splits = n_splits
        self.fold = fold
        self.is_train = is_train

        # 이미지와 라벨 파일 목록 가져오기
        self.pngs = self._get_image_files()
        self.jsons = self._get_label_files()

        # train/valid 분할
        self._split_dataset(is_train)

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

    def _split_dataset(self, is_train):
        from sklearn.model_selection import GroupKFold

        _filenames = np.array(self.pngs)
        _labelnames = np.array(self.jsons)

        # 환자 단위로 그룹화
        groups = [os.path.dirname(fname) for fname in _filenames]

        gkf = GroupKFold(n_splits=self.n_splits)
        folds = list(gkf.split(_filenames, [0]*len(_filenames), groups))

        # fold 분할
        if is_train:
            train_indices = []
            for i, (_, fold_indices) in enumerate(folds):
                if i != self.fold:
                    train_indices.extend(fold_indices)
            self.filenames = list(_filenames[train_indices])
            self.labelnames = list(_labelnames[train_indices])
        else:
            _, val_indices = folds[self.fold]
            self.filenames = list(_filenames[val_indices])
            self.labelnames = list(_labelnames[val_indices])

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, item):
        image_name = self.filenames[item]
        image_path = os.path.join(self.image_root, image_name)

        image = cv2.imread(image_path)
        if image is None:
            print(f"이미지를 불러올 수 없습니다: {image_path}")
            image = np.zeros((2048, 2048, 3), dtype=np.uint8)

        image = image / 255.

        label_name = self.labelnames[item]
        label_path = os.path.join(self.label_root, label_name)

        label_shape = tuple(image.shape[:2]) + (len(CLASSES), )
        label = np.zeros(label_shape, dtype=np.uint8)

        with open(label_path, "r") as f:
            annotations = json.load(f)["annotations"]

        for ann in annotations:
            c = ann["label"]
            class_ind = CLASS2IND[c]
            points = np.array(ann["points"])

            class_label = np.zeros(image.shape[:2], dtype=np.uint8)
            cv2.fillPoly(class_label, [points], 1)
            label[..., class_ind] = class_label

        if self.transforms is not None:
            transformed = self.transforms(image=image, mask=label)
            image = transformed["image"]
            label = transformed["mask"]

        image = image.transpose(2, 0, 1)
        label = label.transpose(2, 0, 1)

        image = torch.from_numpy(image).float()
        label = torch.from_numpy(label).float()

        return image, label


class XRayInferenceDataset(Dataset):
    def __init__(self, image_root, transforms=None):
        self.image_root = image_root
        self.transforms = transforms

        self.filenames = self._get_image_files()

    def _get_image_files(self):
        return sorted({
            os.path.relpath(os.path.join(root, fname), start=self.image_root)
            for root, _dirs, files in os.walk(self.image_root)
            for fname in files
            if os.path.splitext(fname)[1].lower() == ".png"
        })

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, item):
        image_name = self.filenames[item]
        image_path = os.path.join(self.image_root, image_name)

        image = cv2.imread(image_path)
        if image is None:
            print(f"이미지를 불러올 수 없습니다: {image_path}")
            image = np.zeros((2048, 2048, 3), dtype=np.uint8)

        image = image / 255.

        if self.transforms is not None:
            transformed = self.transforms(image=image)
            image = transformed["image"]

        image = image.transpose(2, 0, 1)
        image = torch.from_numpy(image).float()

        return image, image_name

class RoiXRayDataset(Dataset):
    def __init__(self, image_root, label_root, csv_file, is_train=True, transforms=None,
                 use_cv=False, n_splits=5, fold=0):
        self.image_root = image_root
        self.label_root = label_root
        self.transforms = transforms
        self.use_cv = use_cv
        self.n_splits = n_splits
        self.fold = fold
        self.is_train = is_train

        # CSV 파일에서 bbox 정보를 읽어오기
        self.bbox_data = self._load_bbox_data(csv_file)

        # 이미지와 라벨 파일 목록 가져오기
        self.pngs = self._get_image_files()
        self.jsons = self._get_label_files()

        # train/valid 분할
        self._split_dataset(is_train)

    def _load_bbox_data(self, csv_file):
        """CSV 파일에서 bbox 데이터를 로드"""
        bbox_df = pd.read_csv(csv_file)
        bbox_dict = {}
        for idx, row in bbox_df.iterrows():
            image_name = row['image_name']
            bbox = eval(row['bbox'])  # bbox는 문자열 형태이므로 파싱
            bbox_dict[image_name] = bbox
        return bbox_dict

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

    def _split_dataset(self, is_train):
        from sklearn.model_selection import GroupKFold

        _filenames = np.array(self.pngs)
        _labelnames = np.array(self.jsons)

        # 환자 단위로 그룹화
        groups = [os.path.dirname(fname) for fname in _filenames]

        gkf = GroupKFold(n_splits=self.n_splits)
        folds = list(gkf.split(_filenames, [0]*len(_filenames), groups))

        # fold 분할
        if is_train:
            train_indices = []
            for i, (_, fold_indices) in enumerate(folds):
                if i != self.fold:
                    train_indices.extend(fold_indices)
            self.filenames = list(_filenames[train_indices])
            self.labelnames = list(_labelnames[train_indices])
        else:
            _, val_indices = folds[self.fold]
            self.filenames = list(_filenames[val_indices])
            self.labelnames = list(_labelnames[val_indices])

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, item):
        image_name = self.filenames[item]
        image_path = os.path.join(self.image_root, image_name)

        image = cv2.imread(image_path)
        if image is None:
            print(f"이미지를 불러올 수 없습니다: {image_path}")
            image = np.zeros((2048, 2048, 3), dtype=np.uint8)

        image = image / 255.

        label_name = self.labelnames[item]
        label_path = os.path.join(self.label_root, label_name)

        label_shape = tuple(image.shape[:2]) + (len(CLASSES), )
        label = np.zeros(label_shape, dtype=np.uint8)

        with open(label_path, "r") as f:
            annotations = json.load(f)["annotations"]

        for ann in annotations:
            c = ann["label"]
            class_ind = CLASS2IND[c]
            points = np.array(ann["points"])

            class_label = np.zeros(image.shape[:2], dtype=np.uint8)
            cv2.fillPoly(class_label, [points], 1)
            label[..., class_ind] = class_label

        # bbox 정보에 맞게 crop하기
        if image_name in self.bbox_data:
            bbox = self.bbox_data[image_name]  # CSV에서 가져온 bbox
            x_min, y_min, x_max, y_max = bbox
            image = image[y_min:y_max, x_min:x_max]
            label = label[y_min:y_max, x_min:x_max, :]
        else:
            print(f"None image_name : {image_name}")

        if self.transforms is not None:
            transformed = self.transforms(image=image, mask=label)
            image = transformed["image"]
            label = transformed["mask"]

        image = image.transpose(2, 0, 1)
        label = label.transpose(2, 0, 1)

        image = torch.from_numpy(image).float()
        label = torch.from_numpy(label).float()

        return image, label


class RoiXRayInferenceDataset(Dataset):
    def __init__(self, image_root, csv_file, transforms=None):
        self.image_root = image_root
        self.transforms = transforms

        self.filenames = self._get_image_files()

        # CSV 파일에서 bbox 정보를 읽어오기
        self.bbox_data = self._load_bbox_data(csv_file)

    def _load_bbox_data(self, csv_file):
        """CSV 파일에서 bbox 데이터를 로드"""
        bbox_df = pd.read_csv(csv_file)
        bbox_dict = {}
        for idx, row in bbox_df.iterrows():
            image_name = row['image_name']
            bbox = eval(row['bbox'])  # bbox는 문자열 형태이므로 파싱
            bbox_dict[image_name] = bbox
        return bbox_dict

    def _get_image_files(self):
        return sorted({
            os.path.relpath(os.path.join(root, fname), start=self.image_root)
            for root, _dirs, files in os.walk(self.image_root)
            for fname in files
            if os.path.splitext(fname)[1].lower() == ".png"
        })

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, item):
        image_name = self.filenames[item]
        image_path = os.path.join(self.image_root, image_name)

        image = cv2.imread(image_path)
        if image is None:
            print(f"이미지를 불러올 수 없습니다: {image_path}")
            image = np.zeros((2048, 2048, 3), dtype=np.uint8)

        image = image / 255.

        # bbox 정보에 맞게 crop하기
        if image_name in self.bbox_data:
            bbox = self.bbox_data[image_name]  # CSV에서 가져온 bbox
            x_min, y_min, x_max, y_max = bbox
            image = image[y_min:y_max, x_min:x_max]
        else:
            print(f"None image_name : {image_name}")

        if self.transforms is not None:
            transformed = self.transforms(image=image)
            image = transformed["image"]

        image = image.transpose(2, 0, 1)
        image = torch.from_numpy(image).float()

        return image, image_name