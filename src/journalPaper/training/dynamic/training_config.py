from typing import Dict, List

# model architecture params
NUM_CLASSES:int = 3
MODEL_INPUT_SIZE:Dict[str, int] = {
    "EfficientNet-B1":240,
    "EfficientNet-B4":380,
    "Modified_HRNet":256,
}
# DATA PATHS
PRETRAINED:bool = True
PATH_TO_WEIGHTS:str = "/nfs/home/ddresvya/scripts/EngagementRecognition/weights_best_models/pose/fresh-bush-43.pth"# "/nfs/home/ddresvya/scripts/EngagementRecognition/weights_best_models/face/deep-capybara-42.pth" #   # or
PATH_TO_WEIGHTS_HRNET:str = "/nfs/home/ddresvya/scripts/simple-HRNet-master/pose_hrnet_w32_256x192.pth"
PATH_TO_DATA:str = '/nfs/scratch/ddresvya/Data/'# '/nfs/home/ddresvya/Data/'
DATA_TYPE = "pose" # or "face"

# training metaparams
NUM_EPOCHS:int = 100
OPTIMIZER:str = "AdamW"
AUGMENT_PROB:float = 0.05
EARLY_STOPPING_PATIENCE:int = 10
WEIGHT_DECAY:float = 0.0001

# scheduller
LR_SCHEDULLER:str = "Warmup_cyclic"
ANNEALING_PERIOD:int = 5
LR_MAX_CYCLIC:float = 0.005
LR_MIN_CYCLIC:float = 0.0001
LR_MIN_WARMUP:float = 0.00001
WARMUP_STEPS:int = 100
WARMUP_MODE:str = "linear"

# general params
LABEL_COLUMNS: List[str] = ["label_0", "label_1", "label_2"]
BEST_MODEL_SAVE_PATH:str = "best_models/"
NUM_WORKERS:int = 16
splitting_seed:int = 101095