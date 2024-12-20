# Hand Bone Image Segmentation

## 📖 Overview

- Duration : 2024.11.13 10:00 ~ 2024.11.28 19:00
- 네이버 커넥트 재단 및 Upstage에서 주관하는 비공개 대회
- X-ray Hand bone 이미지를 이용해 Segmentation Task를 수행하는 모델을 개발하는 대회
- 하나의 이미지당 29개의 class를 가지고 있고 왼손, 오른손 동일한 양의 이미지가 존재


## 🧑‍💻 Contributors

|김태한|문채원|서동환|윤남규|이재훈|장지우
|:----:|:----:|:----:|:----:|:----:|:----:|
| [<img src="https://avatars.githubusercontent.com/u/84124094?v=4" alt="" style="width:100px;100px;">](https://github.com/taehan79-kim) <br/> | [<img src="https://github.com/user-attachments/assets/ea61c11c-c577-45bb-ae8e-64dffa192402" alt="" style="width:100px;100px;">](https://github.com/mooniswan) <br/> | [<img src="https://avatars.githubusercontent.com/u/87591965?v=4" alt="" style="width:100px;100px;">](https://github.com/Donghwan127) <br/> | [<img src="https://avatars.githubusercontent.com/u/152387005?v=4" alt="" style="width:100px;100px;">](https://github.com/Namgyu-Youn) <br/> | [<img src="https://github.com/user-attachments/assets/3ed91d99-0ad0-43ee-bb11-0aefc61a0a0e" alt="" style="width:100px;100px;">](https://github.com/syous154) <br/> | [<img src="https://github.com/user-attachments/assets/04f5faa7-05c4-4ecc-87f1-0befb53da70d" alt="" style="width:100px;100px;">](https://github.com/zangzoo) <br/> |

## 📝 Wrap up Report

<a href="https://github.com/boostcampaitech7/level2-cv-semanticsegmentation-cv-14-lv3/blob/main/docs/Wrap_up_Report_CV14.pdf">
  <img src="https://img.shields.io/badge/WrapUp_Report-white?style=for-the-badge&logo=adobeacrobatreader&logoColor=red" alt="Data-Centric report">

## Metrics

- Dice

<img width="494" alt="image" src="https://github.com/user-attachments/assets/34ebb94b-b230-4e45-9a51-30113299b999">


## 🔧 Tools

- 🧑‍💻 Programming : GitHub, VScode
- 👥 Communication : GitHub, Notion, Slack
- 🧐 Monitoring and report : WandB
- 💄 Visualization : Streamlit, Gradio, WandB
- 🧱 Deployment : Docker

## 📦 Folder Structure

```
📦level2-cv-semanticsegmentation-cv-14-lv3
 ┣ 📂archive
 ┃ ┣ 📜gpu_trainer.py
 ┃ ┣ 📜trainer_hook.py
 ┃ ┣ 📜train_hook.py
 ┃ ┗ 📜tta_inference.py
 ┣ 📂instance_seg
 ┃ ┣ 📜convert_dataset.py
 ┃ ┗ 📜yolo_train.py
 ┣ 📂model
 ┃ ┣ 📂duck_net
 ┃ ┣ 📂u3_effnet
 ┃ ┣ 📂u3_maxvit
 ┃ ┗ 📂u3_resnet
 ┣ 📂tools
 ┃ ┣ 📂2_stage
 ┃ ┃ ┗ 📜ROI_Extraction.py
 ┃ ┣ 📂ensemble
 ┃ ┃ ┣ 📜2class_ensemble.py
 ┃ ┃ ┣ 📜fusion_new.py
 ┃ ┃ ┣ 📜hard_ensemble.py
 ┃ ┃ ┣ 📜merge_wrist.py
 ┃ ┃ ┣ 📜soft_ensemble.py
 ┃ ┃ ┣ 📜soft_voting_setting.yaml
 ┃ ┃ ┗ 📜weight_ensemble.py
 ┃ ┣ 📂streamlit
 ┃ ┃ ┣ 📜aug_vis.py
 ┃ ┃ ┣ 📜vis.py
 ┃ ┃ ┣ 📜visualize.py
 ┃ ┃ ┗ 📜vis_test.py
 ┃ ┗ 📜csv_merger.py
 ┣ 📂utils
 ┃ ┣ 📜weight_init.py
 ┃ ┗ 📜__init__.py
 ┣ 📜dataset.py
 ┣ 📜functions.py
 ┣ 📜inference.py
 ┣ 📜loss.py
 ┣ 📜requirements.txt
 ┣ 📜sweep_config.yaml
 ┣ 📜train.py
 ┗ 📜trainer.py
```


## Used Architecture
### Semantic Segmentation Models
- U-Net 계열 : UNet, UNet++, UNet3+, DuckNet, swinUNETR
- Yolo 계열 : YOLOv8x-seg, YOLOv11x-seg
- DeepLab 계열 : DeepLabV3, DeepLabV3+


### Backbones(Encoder)
- ResNet 계열 : ResNet, ResNeXt, ResNeSt,
- EfficientNet 계열 : B4, B5, B6, B7, timm-b7, V2-L
- ViT 계열 : MiT, MaxViT
- ETC : HRNet, Swin-T, DUCK-Net


## LB Score

- Public Score
 <img width="806" alt="image" src="https://github.com/user-attachments/assets/b933d232-ecdb-41fe-ac3b-acf06a00311a">

- Private Score
 <img width="806" alt="image" src="https://github.com/user-attachments/assets/c96b67e0-4a76-4d7c-a36e-3e944dfdcc35">
