import os


# KFold 중 사용할 fold [0, 1, 2, 3, 4]
KFOLD_N = 0

# pretrained된 pt를 사용할 경우
PRETRAINED = False
PRETRAINED_DIR = '/path/to/best_model.pt'

# 하이퍼 파라미터
BATCH_SIZE = 16
BATCH_SIZE_VALID = 1
LR = 1e-4
RANDOM_SEED = 42
NUM_EPOCHS = 100
VAL_EVERY = 5

# 저장 경로이자 추론 경로
SAVED_DIR = f"/data/ephemeral/home/ng-youn"

DATA_ROOT = "/data/ephemeral/home/data"

IMAGE_ROOT = DATA_ROOT + "/train/DCM"
LABEL_ROOT = DATA_ROOT + "/train/outputs_json"
TEST_IMAGE_ROOT = DATA_ROOT + "/test/DCM"

CLASSES = [
            'finger-1', 'finger-2', 'finger-3', 'finger-4', 'finger-5',
            'finger-6', 'finger-7', 'finger-8', 'finger-9', 'finger-10',
            'finger-11', 'finger-12', 'finger-13', 'finger-14', 'finger-15',
            'finger-16', 'finger-17', 'finger-18', 'finger-19', 'Trapezium',
            'Trapezoid', 'Capitate', 'Hamate', 'Scaphoid', 'Lunate',
            'Triquetrum', 'Pisiform', 'Radius', 'Ulna',
        ]
CLASS2IND = {v: i for i, v in enumerate(CLASSES)}
IND2CLASS = {v: k for k, v in CLASS2IND.items()}

PALETTE = [
    (220, 20, 60), (119, 11, 32), (0, 0, 142), (0, 0, 230), (106, 0, 228),
    (0, 60, 100), (0, 80, 100), (0, 0, 70), (0, 0, 192), (250, 170, 30),
    (100, 170, 30), (220, 220, 0), (175, 116, 175), (250, 0, 30), (165, 42, 42),
    (255, 77, 255), (0, 226, 252), (182, 182, 255), (0, 82, 0), (120, 166, 157),
    (110, 76, 0), (174, 57, 255), (199, 100, 0), (72, 0, 118), (255, 179, 240),
    (0, 125, 92), (209, 0, 151), (188, 208, 182), (0, 220, 176),
]



if not os.path.exists(SAVED_DIR):
    os.makedirs(SAVED_DIR)