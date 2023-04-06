from typing import Dict

# model architecture params
NUM_CLASSES:int = 3
MODEL_INPUT_SIZE:Dict[str, int] = {
    "Modified_HRNet": 256,
}

# training metaparams
NUM_EPOCHS:int = 100
OPTIMIZER:str = "AdamW"
AUGMENT_PROB:float = 0.05
EARLY_STOPPING_PATIENCE:int = 35
WEIGHT_DECAY:float = 0.0001

# scheduller
LR_SCHEDULLER:str = "Warmup_cyclic"
ANNEALING_PERIOD:int = 5
LR_MAX_CYCLIC:float = 0.005
LR_MIN_CYCLIC:float = 0.0001
LR_MIN_WARMUP:float = 0.00001
WARMUP_STEPS:int = 100
WARMUP_MODE:str = "linear"

# gradual unfreezing
UNFREEZING_LAYERS_PER_EPOCH:int = 1
LAYERS_TO_UNFREEZE_BEFORE_START:int = 7

# Discriminative learning
DISCRIMINATIVE_LEARNING_INITIAL_LR:float = 0.005
DISCRIMINATIVE_LEARNING_MINIMAL_LR:float = 0.00005
DISCRIMINATIVE_LEARNING_MULTIPLICATOR:float = 0.85
DISCRIMINATIVE_LEARNING_STEP:int = 1
DISCRIMINATIVE_LEARNING_START_LAYER:int = -6


# general params
BEST_MODEL_SAVE_PATH:str = "best_models/"
NUM_WORKERS:int = 16
splitting_seed:int = 101095