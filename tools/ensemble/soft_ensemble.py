import os
import sys
import cv2
import torch
import argparse
import numpy as np
import pandas as pd
import os.path as osp
import albumentations as A
import torch.nn.functional as F

from tqdm import tqdm
from omegaconf import OmegaConf
from torch.utils.data import Dataset, DataLoader

import warnings
warnings.filterwarnings('ignore')


class EnsembleDataset(Dataset):
    """
    Soft Voting을 위한 DataSet 클래스입니다. 이 클래스는 이미지를 로드하고 전처리하는 작업과
    구성 파일에서 지정된 변환을 적용하는 역할을 수행합니다.

    Args:
        fnames (set) : 로드할 이미지 파일 이름들의 set
        cfg (dict) : 이미지 루트 및 클래스 레이블 등 설정을 포함한 구성 객체
        tf_dict (dict) : 이미지에 적용할 Resize 변환들의 dict
    """    
    def __init__(self, fnames, cfg, tf_dict):
        self.fnames = np.array(sorted(fnames))
        self.image_root = cfg.image_root
        self.tf_dict = tf_dict
        self.ind2class = {i : v for i, v in enumerate(cfg.CLASSES)}

    def __len__(self):
        return len(self.fnames)
    
    def __getitem__(self, item):
        """
        지정된 인덱스에 해당하는 이미지를 로드하여 반환합니다.
        Args:
            item (int): 로드할 이미지의 index

        Returns:
            dict : "image", "image_name"을 키값으로 가지는 dict
        """        
        image_name = self.fnames[item]
        image_path = osp.join(self.image_root, image_name)
        image = cv2.imread(image_path)

        assert image is not None, f"{image_path} 해당 이미지를 찾지 못했습니다."
        
        image = image / 255.0
        return {"image" : image, "image_name" : image_name}

    def collate_fn(self, batch):
        """
        배치 데이터를 처리하는 커스텀 collate 함수입니다.

        Args:
            batch (list): __getitem__에서 반환된 데이터들의 list

        Returns:
            dict: 처리된 이미지들을 가지는 dict
            list: 이미지 이름으로 구성된 list
        """        
        images = [data['image'] for data in batch]
        image_names = [data['image_name'] for data in batch]
        inputs = {"images" : images}

        image_dict = self._apply_transforms(inputs)

        image_dict = {k : torch.from_numpy(v.transpose(0, 3, 1, 2)).float()
                      for k, v in image_dict.items()}
        
        for image_size, image_batch in image_dict.items():
            assert len(image_batch.shape) == 4, \
                f"collate_fn 내부에서 image_batch의 차원은 반드시 4차원이어야 합니다.\n \
                현재 shape : {image_batch.shape}"
            assert image_batch.shape == (len(batch), 3, image_size, image_size), \
                f"collate_fn 내부에서 image_batch의 shape은 ({len(batch)}, 3, {image_size}, {image_size})이어야 합니다.\n \
                현재 shape : {image_batch.shape}"

        return image_dict, image_names
    
    def _apply_transforms(self, inputs):
        """
        입력된 이미지에 변환을 적용합니다.

        Args:
            inputs (dict): 변환할 이미지를 포함하는 딕셔너리

        Returns:
            dict : 변환된 이미지들
        """        
        return {
            key: np.array(pipeline(**inputs)['images']) for key, pipeline in self.tf_dict.items()
        }


def encode_mask_to_rle(mask):
    # mask map으로 나오는 인퍼런스 결과를 RLE로 인코딩 합니다.
    pixels = mask.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

def load_models(cfg, device):
    """
    구성 파일에 지정된 경로에서 모델을 로드합니다.

    Args:
        cfg (dict): 모델 경로가 포함된 설정 객체
        device (torch.device): 모델을 로드할 장치 (CPU or GPU)

    Returns:
        dict: 처리 이미지 크기별로 모델을 그룹화한 dict
        int: 로드된 모델의 총 개수
    """    
    model_dict = {}
    model_count = 0

    print("\n======== Model Load ========")
    # inference 해야하는 이미지 크기 별로 모델 순차저장
    for key, paths in cfg.model_paths.items():
        if len(paths) == 0:
            continue
        model_dict[key] = []
        print(f"{key} image size 추론 모델 {len(paths)}개 불러오기 진행 시작")
        for path in paths:
            print(f"{osp.basename(path)} 모델을 불러오는 중입니다..", end="\t")
            model = torch.load(path).to(device)
            model.eval()
            model_dict[key].append(model)
            model_count += 1
            print("불러오기 성공!")
        print()

    print(f"모델 총 {model_count}개 불러오기 성공!\n")
    return model_dict, model_count


def save_results(cfg, filename_and_class, rles):
    """
    추론 결과를 csv 파일로 저장합니다.

    Args:
        cfg (dict): 출력 설정을 포함하는 구성 객체
        filename_and_class (list): 파일 이름과 클래스 레이블이 포함된 list
        rles (list): RLE로 인코딩된 세크멘테이션 마스크들을 가진 list
    """    
    classes, filename = zip(*[x.split("_") for x in filename_and_class])
    image_name = [os.path.basename(f) for f in filename]

    df = pd.DataFrame({
        "image_name": image_name,
        "class": classes,
        "rle": rles,
    })

    print("\n======== Save Output ========")
    print(f"{cfg.save_dir} 폴더 내부에 {cfg.output_name}을 생성합니다..", end="\t")
    os.makedirs(cfg.save_dir, exist_ok=True)

    output_path = osp.join(cfg.save_dir, cfg.output_name)
    try:
        df.to_csv(output_path, index=False)
    except Exception as e:
        print(f"{output_path}를 생성하는데 실패하였습니다.. : {e}")
        raise

    print(f"{osp.join(cfg.save_dir, cfg.output_name)} 생성 완료")



def soft_voting(cfg):
    """
    Soft Voting을 수행합니다. 여러 모델의 예측을 결합하여 최종 예측을 생성

    Args:
        cfg (dict): 설정을 포함하는 구성 객체
    """    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    fnames = {
        osp.relpath(osp.join(root, fname), start=cfg.image_root)
        for root, _, files in os.walk(cfg.image_root)
        for fname in files
        if osp.splitext(fname)[1].lower() == ".png"
    }

    tf_dict = {image_size : A.Resize(height=image_size, width=image_size) 
               for image_size, paths in cfg.model_paths.items() 
               if len(paths) != 0}
    
    print("\n======== PipeLine 생성 ========")
    for k, v in tf_dict.items():
        print(f"{k} 사이즈는 {v} pipeline으로 처리됩니다.")

    dataset = EnsembleDataset(fnames, cfg, tf_dict)
    
    data_loader = DataLoader(dataset=dataset,
                             batch_size=cfg.batch_size,
                             shuffle=False,
                             num_workers=cfg.num_workers,
                             drop_last=False,
                             collate_fn=dataset.collate_fn)

    model_dict, model_count = load_models(cfg, device)
    
    filename_and_class = []
    rles = []

    print("======== Soft Voting Start ========")
    with torch.no_grad():
        with tqdm(total=len(data_loader), desc="[Inference...]", disable=False) as pbar:
            for image_dict, image_names in data_loader:
                total_output = torch.zeros((cfg.batch_size, len(cfg.CLASSES), 2048, 2048)).to(device)
                for name, models in model_dict.items():
                    for model in models:
                        outputs = model(image_dict[name].to(device))
                        outputs = F.interpolate(outputs, size=(2048, 2048), mode="bilinear")
                        outputs = torch.sigmoid(outputs)
                        total_output += outputs
                        
                total_output /= model_count
                total_output = (total_output > cfg.threshold).detach().cpu().numpy()

                for output, image_name in zip(total_output, image_names):
                    for c, segm in enumerate(output):
                        rle = encode_mask_to_rle(segm)
                        rles.append(rle)
                        filename_and_class.append(f"{dataset.ind2class[c]}_{image_name}")
                
                pbar.update(1)

    save_results(cfg, filename_and_class, rles)

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./soft_voting_setting.yaml")

    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        cfg = OmegaConf.load(f)

    if cfg.root_path not in sys.path:
        sys.path.append(cfg.root_path)
    
    soft_voting(cfg)
