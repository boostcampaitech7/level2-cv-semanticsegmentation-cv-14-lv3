# Hand Bone Image Segmentation

## **📘**Overview

2024.11.13 10:00 ~ 2024.11.28 19:00


X-ray 이미지에서 사람의 뼈를 Segmentation 하는 인공지능 만들기


## **📘**Contributors

|김태한|문채원|서동횐|윤남규|이재훈|장지우
|:----:|:----:|:----:|:----:|:----:|:----:|
| [<img src="https://github.com/user-attachments/assets/366fc4d1-3716-4214-a6ef-87f0a4c6147f" alt="" style="width:100px;100px;">](https://github.com/taehan79-kim) <br/> | [<img src="https://github.com/user-attachments/assets/ea61c11c-c577-45bb-ae8e-64dffa192402" alt="" style="width:100px;100px;">](https://github.com/mooniswan) <br/> | [<img src="https://github.com/Donghwan12" alt="" style="width:100px;100px;">](https://github.com/Donghwan127) <br/> | [<img src="https://github.com/user-attachments/assets/6bc5913f-6e59-4aae-9433-3db2c7251978" alt="" style="width:100px;100px;">]([https://github.com/0522chan](https://github.com/Namgyu-Youn)) <br/> | [<img src="https://github.com/user-attachments/assets/3ed91d99-0ad0-43ee-bb11-0aefc61a0a0e" alt="" style="width:100px;100px;">](https://github.com/syous154) <br/> | [<img src="https://github.com/user-attachments/assets/04f5faa7-05c4-4ecc-87f1-0befb53da70d" alt="" style="width:100px;100px;">](https://github.com/zangzoo) <br/> |

## **📘**Wrap up Report

곧 채울 예정

## **📘**Metrics

- Dice

![image](https://github.com/user-attachments/assets/76719a2a-41eb-4698-b1d8-eec2bb4a3cee)

![image](https://github.com/user-attachments/assets/7b88489b-ba4b-4b0a-811f-d605e7a79fee)


## **📰**Tools

- github
- notion
- slack
- wandb

## **📰**Folder Structure

```

📦level2-cv-semanticsegmentation-cv-14-lv3-1
 ┣ 📂archive
 ┃ ┣ 📜gpu_trainer.py
 ┃ ┣ 📜trainer_hook.py
 ┃ ┣ 📜train_hook.py
 ┃ ┗ 📜tta_inference.py
 ┣ 📂docs
 ┃ ┣ 📜gdown_guide.md
 ┃ ┣ 📜github_guide.md
 ┃ ┗ 📜using_tmux_background.md
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

## **📰**Dataset Structure

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
- 29 class : finger-1, finger-2, finger-3, finger-4, finger-5, finger-6, finger-7, finger-8, finger-9, finger-10, finger-11, finger-12, finger-13, finger-14, finger-15, finger-16, finger-17, finger-18, finger-19, Trapezium, Trapezoid, Capitate, Hamate, Scaphoid, Lunate, Triquetrum, Pisiform, Radius, Ulna
- image size :  (2048, 2048)

## **📰**Models

- UNet
- UNet++
- UNet3+
- YOLOv8x-seg
- YOLOv11x-seg
- DuckNet
- DeepLabV3
- swinUNETR


## **📰**Backbones

- ResNet
- ResNext
- HRNet
- EfficientNet
- Swin-T
- maxvit
- mit

## **📰Experiments**
![image](https://github.com/user-attachments/assets/ab78108c-302c-4d4d-a22c-bbede89bfb9e)


![image](https://github.com/user-attachments/assets/38fd64fe-8c69-422d-b418-887d88720d37)



