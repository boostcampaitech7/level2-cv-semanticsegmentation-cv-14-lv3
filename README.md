# Hand Bone Image Segmentation

## Overview

- Duration : 2024.11.13 10:00 ~ 2024.11.28 19:00
- 네이버 커넥트 재단 및 Upstage에서 주관하는 비공개 대회
- X-ray Hand bone 이미지를 이용해 Segmentation Task를 수행하는 모델을 개발하는 대회
- 하나의 이미지당 29개의 class를 가지고 있고 왼손, 오른손 동일한 양의 이미지가 존재

<a href="https://github.com/boostcampaitech7/level2-cv-semanticsegmentation-cv-14-lv3/tree/main/docs/Wrap_up_Report_CV14.pdf">

## Contributors

|김태한|문채원|서동환|윤남규|이재훈|장지우
|:----:|:----:|:----:|:----:|:----:|:----:|
| [<img src="https://avatars.githubusercontent.com/u/84124094?v=4" alt="" style="width:100px;100px;">](https://github.com/taehan79-kim) <br/> | [<img src="https://github.com/user-attachments/assets/ea61c11c-c577-45bb-ae8e-64dffa192402" alt="" style="width:100px;100px;">](https://github.com/mooniswan) <br/> | [<img src="https://avatars.githubusercontent.com/u/87591965?v=4" alt="" style="width:100px;100px;">](https://github.com/Donghwan127) <br/> | [<img src="https://avatars.githubusercontent.com/u/152387005?v=4" alt="" style="width:100px;100px;">](https://github.com/Namgyu-Youn) <br/> | [<img src="https://github.com/user-attachments/assets/3ed91d99-0ad0-43ee-bb11-0aefc61a0a0e" alt="" style="width:100px;100px;">](https://github.com/syous154) <br/> | [<img src="https://github.com/user-attachments/assets/04f5faa7-05c4-4ecc-87f1-0befb53da70d" alt="" style="width:100px;100px;">](https://github.com/zangzoo) <br/> |

## Wrap up Report

곧 채울 예정

## Metrics

- Dice

<img width="494" alt="image" src="https://github.com/user-attachments/assets/34ebb94b-b230-4e45-9a51-30113299b999">


## Tools

- github
- notion
- slack
- wandb

## Folder Structure

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

## Dataset Structure

```

📦data
     ┣ 📂test
     ┃    ┣ 📂DCM
     ┃         ┣ 📂ID040
     ┃         ┃     📜image1661319116107.png
     ┃         ┃     📜image1661319145363.png
     ┃         ┗ 📂ID041
     ┃                📜image1661319356239.png
     ┃                📜image1661319390106.png
     ┃
     ┣ 📂train
     ┃    ┣ 📂DCM
     ┃    ┃   ┣ 📂ID001
     ┃    ┃   ┃     image1661130828152_R.png
     ┃    ┃   ┃     image1661130891365_L.png
     ┃    ┃   ┗ 📂ID002
     ┃    ┃          image1661144206667.png
     ┃    ┃          image1661144246917.png
     ┃    ┃
     ┃    ┗ 📜outputs_json
     ┃               ┣ 📂ID001
     ┃               ┃     📜image1661130828152_R.json
     ┃               ┃     📜image1661130891365_L.json
     ┃               ┗ 📂ID002
                             📜image1661144206667.json
                             📜image1661144246917.json

```

- images : 1088
    - train : 800
    - test : 288
- class : 29
- image size :  (2048, 2048)

## Models

- UNet
- UNet++
- UNet3+
- YOLOv8x-seg
- YOLOv11x-seg
- DuckNet
- DeepLabV3
- swinUNETR


## Backbones

- ResNet
- ResNext
- HRNet
- EfficientNet
- Swin-T
- maxvit
- mit


## LB Score**

- Public Score
 <img width="806" alt="image" src="https://github.com/user-attachments/assets/b933d232-ecdb-41fe-ac3b-acf06a00311a">

- Private Score
 <img width="806" alt="image" src="https://github.com/user-attachments/assets/c96b67e0-4a76-4d7c-a36e-3e944dfdcc35">
