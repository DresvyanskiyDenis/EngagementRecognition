import gc
import os
import sys

from src.NoXi.visual_subsystem.pose_subsystem.frame_wise_model.HRnet_training_wandb import get_data_loaders_from_data

sys.path.extend(["/work/home/dsu/datatools/"])
sys.path.extend(["/work/home/dsu/engagement_recognition_project_server/"])
sys.path.extend(["/work/home/dsu/simpleHRNet/"])

from collections import Callable
from functools import partial
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import wandb
import torch
import torchvision.transforms as T
from sklearn.metrics import recall_score, precision_score, f1_score, confusion_matrix
from sklearn.utils import compute_class_weight
from torchinfo import summary

from decorators.common_decorators import timer
from pytorch_utils.callbacks import TorchEarlyStopping, TorchMetricEvaluator
from pytorch_utils.generators.ImageDataGenerator import ImageDataLoader
from pytorch_utils.generators.pytorch_augmentations import pad_image_random_factor, grayscale_image, \
    collor_jitter_image_random, gaussian_blur_image_random, random_perspective_image, random_rotation_image, \
    random_crop_image, random_posterize_image, random_adjust_sharpness_image, random_equalize_image, \
    random_horizontal_flip_image, random_vertical_flip_image
from pytorch_utils.losses import FocalLoss, SoftFocalLoss
from src.NoXi.visual_subsystem.pose_subsystem.frame_wise_model.HRNet import load_HRNet_model, modified_HRNet
from src.NoXi.visual_subsystem.pose_subsystem.frame_wise_model.utils import load_NoXi_data_all_languages, \
    convert_image_to_float_and_scale

def validate_model(dataset:torch.utils.data.DataLoader, model:torch.nn.Module)->None:
    # specify val metrics
    val_metrics = {
        'recall': partial(recall_score, average='macro'),
        'precision': partial(precision_score, average='macro'),
        'f1_score:': partial(f1_score, average='macro'),
        'confusion_matrix': confusion_matrix
    }
    # specify device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # make evaluator, which will be used to evaluate the model
    metric_evaluator = TorchMetricEvaluator(generator = dataset,
                 model=model,
                 metrics=val_metrics,
                 device=device,
                 need_argmax=True,
                 need_softmax=True,
                 loss_func=None)

    # evaluate the model
    results = metric_evaluator()

    return results





def main():
    # params
    BATCH_SIZE=64
    NUM_CLASSES=5
    # load data
    train, dev, test = load_NoXi_data_all_languages(train_labels_as_categories=False,
                                                    dev_labels_as_categories=False,
                                                    test_labels_as_categories=False)

    train_gen, dev_gen, test_gen = get_data_loaders_from_data(train, dev, test, augment=True, augment_prob=0.05,
                                                              batch_size=BATCH_SIZE,
                                                              preprocessing_functions=[T.Resize(size=(256, 256)),
                                                                                       convert_image_to_float_and_scale,
                                                                                       T.Normalize(
                                                                                           mean=[0.485, 0.456, 0.406],
                                                                                           std=[0.229, 0.224, 0.225])
                                                                                       ])  # From HRNet

    # create model and load its weights



if __name__ == "__main__":
    main()
