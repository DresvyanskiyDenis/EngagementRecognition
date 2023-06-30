from typing import Dict, List

# model architecture params
NUM_CLASSES:int = 3
# training metaparams
NUM_EPOCHS:int = 100
OPTIMIZER:str = "AdamW"
EARLY_STOPPING_PATIENCE:int = 80
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