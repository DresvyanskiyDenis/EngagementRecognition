#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Contains the script for training the VGGFace2 model on the IWSDS2023 dataset. The Weights and Biases library is used to
monitor and log the information about training process.

"""

__author__ = "Denis Dresvyanskiy"
__copyright__ = "Copyright 2022"
__credits__ = ["Denis Dresvyanskiy"]
__maintainer__ = "Denis Dresvyanskiy"
__email__ = "denis.dresvyanskiy@uni-ulm.de"

import sys
sys.path.extend(["/work/home/dsu/datatools/"])
sys.path.extend(["/work/home/dsu/engagement_recognition_project_server/"])

import gc
import numpy as np
import pandas as pd
import tensorflow as tf
import wandb
from typing import Optional

from functools import partial
from sklearn.metrics import recall_score, precision_score, f1_score
from sklearn.utils import compute_class_weight
from keras.callbacks import EarlyStopping
from wandb.integration.keras import WandbCallback

from tensorflow_utils.Losses import categorical_focal_loss
from tensorflow_utils.tensorflow_datagenerators.ImageDataLoader_tf2 import get_tensorflow_image_loader
from tensorflow_utils.tensorflow_datagenerators.tensorflow_image_augmentations import random_rotate90_image, \
    random_flip_vertical_image, random_flip_horizontal_image, random_crop_image, random_change_brightness_image, \
    random_change_contrast_image, random_change_saturation_image, random_worse_quality_image, \
    random_convert_to_grayscale_image
from tensorflow_utils.tensorflow_datagenerators.tensorflow_image_preprocessing import preprocess_image_VGGFace2
from src.IWSDS2023.visual_subsystem.facial_subsystem.frame_wise_models.utils import load_NoXi_data_all_languages
from tensorflow_utils.wandb_callbacks import WandB_LR_log_callback, WandB_val_metrics_callback
from tensorflow_utils.callbacks import get_annealing_LRreduce_callback, get_reduceLRonPlateau_callback
from tensorflow_utils.models.CNN_models import get_modified_VGGFace2_resnet_model


def create_VGGFace2_model(path_to_weights: str, num_classes: Optional[int] = 4) -> tf.keras.Model:
    """Creates the VGGFace2 model and loads weights for it using provided path.

    :param path_to_weights: str
            Path to the weights for VGGFace2 model.
    :param num_classes: int
            Number of classes to define last softmax layer .
    :return: tf.keras.Model
            Created tf.keras.Model with loaded weights.
    """
    model = get_modified_VGGFace2_resnet_model(dense_neurons_after_conv=(512,),
                                               dropout=0.3,
                                               regularization=tf.keras.regularizers.l2(0.0001),
                                               output_neurons=num_classes, pooling_at_the_end='avg',
                                               pretrained=True,
                                               path_to_weights=path_to_weights)
    return model


def train_model(train, dev, loss_func='categorical_crossentropy')->None:
    """ Creates and trains on the IWSDS2023 dataset the Keras Tensorflow model.
        Here, the model is VGGFace2.
        During the training all metaparams will be logged using the Weights and Biases library.
        Also, different augmentation methods will be applied (see down to the function).
        Overall, the function is designed only for the usage with Weights and Biases library.

    :param train: pd.DataFrame
                Pandas DataFrame with the following columns: [filename, class] or [filename, class_0, class_1, ...].
                Train dataset.
    :param dev: pd.DataFrame
                Pandas DataFrame with the following columns: [filename, class] or [filename, class_0, class_1, ...].
                Development dataset
    :param loss_func: str
                Type of the loss function to be applied. Either "categorical_crossentropy" or "focal_loss".
    :return: None
    """
    # metaparams
    metaparams = {
        "optimizer": "Adam",  # SGD, Nadam
        "learning_rate_max": 0.001,  # up to 0.0001
        "learning_rate_min": 0.00001,  # up to 0.000001
        "lr_scheduller": "Cyclic",  # "reduceLRonPlateau"
        "annealing_period": 5,
        "epochs": 30,
        "batch_size": 128,
        "augmentation_rate": 0.1,  # 0.2, 0.3
        "architecture": "VGGFace2_frozen_4_blocks",
        "dataset": "IWSDS2023",
        "num_classes": 5
    }

    augmentation_methods = [
        partial(random_rotate90_image, probability=metaparams["augmentation_rate"]),
        partial(random_flip_vertical_image, probability=metaparams["augmentation_rate"]),
        partial(random_flip_horizontal_image, probability=metaparams["augmentation_rate"]),
        partial(random_crop_image, probability=metaparams["augmentation_rate"]),
        partial(random_change_brightness_image, probability=metaparams["augmentation_rate"], min_max_delta=0.35),
        partial(random_change_contrast_image, probability=metaparams["augmentation_rate"], min_factor=0.5,
                max_factor=1.5),
        partial(random_change_saturation_image, probability=metaparams["augmentation_rate"], min_factor=0.5,
                max_factor=1.5),
        partial(random_worse_quality_image, probability=metaparams["augmentation_rate"], min_factor=25, max_factor=99),
        partial(random_convert_to_grayscale_image, probability=metaparams["augmentation_rate"])
    ]

    # initialization of Weights and Biases
    wandb.init(project="VGGFace2_FtF_training", config=metaparams)
    config = wandb.config

    # Metaparams initialization
    metrics = ['accuracy']
    if config.lr_scheduller == 'Cyclic':
        lr_scheduller = get_annealing_LRreduce_callback(highest_lr=config.learning_rate_max,
                                                        lowest_lr=config.learning_rate_min,
                                                        annealing_period=config.annealing_period)
    elif config.lr_scheduller == 'reduceLRonPlateau':
        lr_scheduller = get_reduceLRonPlateau_callback(monitoring_loss='val_loss', reduce_factor=0.1,
                                                       num_patient_epochs=4,
                                                       min_lr=config.learning_rate_min)
    else:
        raise Exception("You passed wrong lr_scheduller.")

    if config.optimizer == 'Adam':
        optimizer = tf.keras.optimizers.Adam(config.learning_rate_max)
    elif config.optimizer == 'Nadam':
        optimizer = tf.keras.optimizers.Nadam(config.learning_rate_max)
    elif config.optimizer == 'SGD':
        optimizer = tf.keras.optimizers.SGD(config.learning_rate_max)
    else:
        raise Exception("You passed wrong optimizer name.")

    # class weights
    class_weights = compute_class_weight(class_weight='balanced',
                                         classes=np.unique(np.argmax(train.iloc[:, 1:].values, axis=1, keepdims=True)),
                                         y=np.argmax(train.iloc[:, 1:].values, axis=1, keepdims=True).flatten())

    # loss function
    if loss_func == 'categorical_crossentropy':
        loss = tf.keras.losses.categorical_crossentropy
        train_class_weights = {i: class_weights[i] for i in range(config.num_classes)}
    elif loss_func == 'focal_loss':
        focal_loss_gamma = 2
        loss = categorical_focal_loss(alpha=class_weights, gamma=focal_loss_gamma)
        train_class_weights = None
    else:
        raise AttributeError(
            'Passed name of loss function is not acceptable. Possible variants are categorical_crossentropy or focal_loss.')
    wandb.config.update({'loss': loss})
    # model initialization
    model = create_VGGFace2_model(path_to_weights='/work/home/dsu/VGG_model_weights/resnet50_softmax_dim512/weights.h5',
                                  num_classes=config.num_classes)
    # freezing first 4 blocks of the ResNet50
    for i in range(141):
        model.layers[i].trainable = False

    # model compilation
    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
    model.summary()

    # create DataLoaders (DataGenerator)
    train_data_loader = get_tensorflow_image_loader(paths_and_labels=train, batch_size=metaparams["batch_size"],
                                                 augmentation=True,
                                                 augmentation_methods=augmentation_methods,
                                                 preprocessing_function=preprocess_image_VGGFace2,
                                                 clip_values=None,
                                                 cache_loaded_images=False)
    # transform labels in dev data to one-hot encodings
    dev = dev.__deepcopy__()
    dev = pd.concat([dev, pd.get_dummies(dev['class'], dtype="float32")], axis=1).drop(columns=['class'])

    dev_data_loader = get_tensorflow_image_loader(paths_and_labels=dev,
                                               batch_size=metaparams["batch_size"],
                                               augmentation=False,
                                               augmentation_methods=None,
                                               preprocessing_function=preprocess_image_VGGFace2,
                                               clip_values=None,
                                               cache_loaded_images=False)

    # create Keras Callbacks for monitoring learning rate and metrics on val_set
    lr_monitor_callback = WandB_LR_log_callback()
    val_metrics = {
        'val_recall': partial(recall_score, average='macro'),
        'val_precision': partial(precision_score, average='macro'),
        'val_f1_score:': partial(f1_score, average='macro')
    }
    val_metrics_callback = WandB_val_metrics_callback(dev_data_loader, val_metrics,
                                                      metric_to_monitor='val_recall')
    early_stopping_callback = EarlyStopping(monitor='val_loss', patience=10, verbose=1)

    # train process
    print("Loss used:%s" % (loss))
    print("FROZEN 4 LAYERS")
    print(config.batch_size)
    print("--------------------")
    model.fit(train_data_loader, epochs=config.epochs,
              class_weight=train_class_weights,
              validation_data=dev_data_loader,
              callbacks=[WandbCallback(),
                         lr_scheduller,
                         early_stopping_callback,
                         lr_monitor_callback,
                         val_metrics_callback])
    # clear RAM
    del train_data_loader, dev_data_loader
    del model
    gc.collect()
    tf.keras.backend.clear_session()


def main():
    print("START")
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    # load the data and labels
    train, dev, test = load_NoXi_data_all_languages()
    # shuffle one more time train data
    train = train.sample(frac=1).reset_index(drop=True)

    sweep_config = {
        'method': 'random',
        'metric': {
            'name': 'val_loss',
            'goal': 'minimize'
        },
        'parameters': {
            'optimizer': {
                'values': ['Adam', 'SGD', 'Nadam']
            },
            'learning_rate_max': {
                'distribution': 'uniform',
                'max': 0.001,
                'min': 0.0001
            },
            'learning_rate_min': {
                'distribution': 'uniform',
                'max': 0.00001,
                'min': 0.000001
            },
            'lr_scheduller': {
                'values': ['Cyclic', 'reduceLRonPlateau']
            },
            'augmentation_rate': {
                'values': [0.1, 0.2, 0.3]
            }
        }
    }

    # categorical crossentropy
    sweep_id = wandb.sweep(sweep_config, project='VGGFace2_FtF_training')
    wandb.agent(sweep_id, function=lambda: train_model(train, dev, 'categorical_crossentropy'), count=30,
                project='VGGFace2_FtF_training')
    tf.keras.backend.clear_session()
    gc.collect()
    # focal loss
    print("Wandb with focal loss")
    sweep_id = wandb.sweep(sweep_config, project='VGGFace2_FtF_training')
    wandb.agent(sweep_id, function=lambda: train_model(train, dev, 'focal_loss'), count=30,
                project='VGGFace2_FtF_training')
    tf.keras.backend.clear_session()
    gc.collect()


if __name__ == '__main__':
    main()
