#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Test file to check different aspects of the code in the frame_wise_models package.

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
from typing import Optional

from keras.callbacks import EarlyStopping
from functools import partial
from sklearn.metrics import recall_score, precision_score, f1_score
from sklearn.utils import compute_class_weight

from src.NoXi.visual_subsystem.frame_wise_models.utils import load_and_preprocess_data
from tensorflow_utils.Losses import categorical_focal_loss
from tensorflow_utils.tensorflow_datagenerators.ImageDataLoader_tf2 import get_tensorflow_generator
from tensorflow_utils.tensorflow_datagenerators.tensorflow_image_augmentations import random_rotate90_image, \
    random_flip_vertical_image, random_flip_horizontal_image, random_crop_image, random_change_brightness_image, \
    random_change_contrast_image, random_change_saturation_image, random_worse_quality_image, \
    random_convert_to_grayscale_image
from tensorflow_utils.wandb_callbacks import WandB_LR_log_callback, WandB_val_metrics_callback
from tensorflow_utils.callbacks import get_annealing_LRreduce_callback, get_reduceLRonPlateau_callback
from tensorflow_utils.tensorflow_datagenerators.tensorflow_image_preprocessing import preprocess_data_MobileNetv3


def create_MobileNetv3_model(num_classes: Optional[int] = 5) -> tf.keras.Model:
    # take model from keras zoo
    model = tf.keras.applications.MobileNetV3Small(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
    # extract last layer
    last_layer = model.layers[-1].output
    # stack global avg pooling, dropout and dense layer on top of it
    avg_pool = tf.keras.layers.GlobalAveragePooling2D()(last_layer)
    dropout_1 = tf.keras.layers.Dropout(0.3)(avg_pool)
    dense_1 = tf.keras.layers.Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.00001))(
        dropout_1)
    # create output softmax dense layer
    output = tf.keras.layers.Dense(num_classes, activation='softmax')(dense_1)
    # create a new model
    new_model = tf.keras.Model(inputs=model.inputs, outputs=[output])
    del model
    tf.keras.backend.clear_session()
    gc.collect()
    return new_model


def train_model(train, dev, loss_func='focal_loss'):
    # metaparams
    metaparams = {
        "optimizer": "Adam",  # SGD, Nadam
        "learning_rate_max": 0.001,  # up to 0.0001
        "learning_rate_min": 0.00001,  # up to 0.000001
        "lr_scheduller": "Cyclic",  # "reduceLRonPlateau"
        "annealing_period": 5,
        "epochs": 30,
        "batch_size": 256,
        "augmentation_rate": 0.1,  # 0.2, 0.3
        "architecture": "MobileNetv3_256_Dense",
        "dataset": "NoXi",
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
    # wandb.init(project="VGGFace2_FtF_training", config=metaparams)
    # config = wandb.config

    # Metaparams initialization
    metrics = ['accuracy']
    if metaparams['lr_scheduller'] == 'Cyclic':
        lr_scheduller = get_annealing_LRreduce_callback(highest_lr=metaparams['learning_rate_max'],
                                                        lowest_lr=metaparams['learning_rate_min'],
                                                        annealing_period=metaparams['annealing_period'])
    elif metaparams['lr_scheduller'] == 'reduceLRonPlateau':
        lr_scheduller = get_reduceLRonPlateau_callback(monitoring_loss='val_loss', reduce_factor=0.1,
                                                       num_patient_epochs=4,
                                                       min_lr=metaparams['learning_rate_min'])
    else:
        raise Exception("You passed wrong lr_scheduller.")

    if metaparams['optimizer'] == 'Adam':
        optimizer = tf.keras.optimizers.Adam(metaparams['learning_rate_max'])
    elif metaparams['optimizer'] == 'Nadam':
        optimizer = tf.keras.optimizers.Nadam(metaparams['learning_rate_max'])
    elif metaparams['optimizer'] == 'SGD':
        optimizer = tf.keras.optimizers.SGD(metaparams['learning_rate_max'])
    else:
        raise Exception("You passed wrong optimizer name.")

    # class weights
    class_weights = compute_class_weight(class_weight='balanced',
                                         classes=np.unique(np.argmax(train.iloc[:, 1:].values, axis=1, keepdims=True)),
                                         y=np.argmax(train.iloc[:, 1:].values, axis=1, keepdims=True).flatten())

    # loss function
    if loss_func == 'categorical_crossentropy':
        loss = tf.keras.losses.categorical_crossentropy
        train_class_weights = {i: class_weights[i] for i in range(metaparams['num_classes'])}
    elif loss_func == 'focal_loss':
        focal_loss_gamma = 2
        loss = categorical_focal_loss(alpha=class_weights, gamma=focal_loss_gamma)
        train_class_weights = None
    else:
        raise AttributeError(
            'Passed name of loss function is not acceptable. Possible variants are categorical_crossentropy or focal_loss.')
    # wandb.config.update({'loss': loss})
    # model initialization
    model = create_MobileNetv3_model(num_classes=metaparams['num_classes'])
    # freezing layers?

    for i, layer in enumerate(model.layers):
        print("%i:%s" % (i, layer.name))

    # for i in range(75): # up to block 8
    #    model.layers[i].trainable = False

    # model compilation
    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
    model.summary()

    # create DataLoaders (DataGenerator)
    train_data_loader = get_tensorflow_generator(paths_and_labels=train, batch_size=metaparams["batch_size"],
                                                 augmentation=True,
                                                 augmentation_methods=augmentation_methods,
                                                 preprocessing_function=preprocess_data_MobileNetv3,
                                                 clip_values=None,
                                                 cache_loaded_images=False)

    dev_for_keras_validation_set = dev.__deepcopy__()
    dev_for_keras_validation_set = pd.concat(
        [dev_for_keras_validation_set, pd.get_dummies(dev_for_keras_validation_set['class'], dtype="float32")],
        axis=1).drop(columns=['class'])

    dev_data_loader_for_keras_validation = get_tensorflow_generator(paths_and_labels=dev_for_keras_validation_set,
                                                                    batch_size=metaparams["batch_size"],
                                                                    augmentation=False,
                                                                    augmentation_methods=None,
                                                                    preprocessing_function=preprocess_data_MobileNetv3,
                                                                    clip_values=None,
                                                                    cache_loaded_images=False)

    # create Keras Callbacks for monitoring learning rate and metrics on val_set
    lr_monitor_callback = WandB_LR_log_callback()
    val_metrics = {
        'val_recall': partial(recall_score, average='macro'),
        'val_precision': partial(precision_score, average='macro'),
        'val_f1_score:': partial(f1_score, average='macro')
    }
    val_metrics_callback = WandB_val_metrics_callback(dev_data_loader_for_keras_validation, val_metrics,
                                                      metric_to_monitor='val_recall')
    early_stopping_callback = EarlyStopping(monitor='val_loss', patience=7, verbose=1)

    # train process
    print("Loss used:%s" % (loss))
    print("MobileNetv3, LAYERS ARE NOT FROZEN")
    print(metaparams['batch_size'])
    print("--------------------")
    model.fit(train_data_loader, epochs=metaparams['epochs'],
              class_weight=train_class_weights,
              validation_data=dev_data_loader_for_keras_validation,
              callbacks=[  # WandbCallback(),
                  lr_scheduller,
                  early_stopping_callback,
                  lr_monitor_callback,
                  val_metrics_callback])
    # clear RAM
    del train_data_loader, dev_data_loader_for_keras_validation
    del model
    gc.collect()
    tf.keras.backend.clear_session()


def main():
    print("START")
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    # loading data
    frame_step = 5
    path_to_data = "/Noxi_extracted/NoXi/extracted_faces/"
    # french data
    path_to_labels_french = "/media/external_hdd_1/NoXi_annotations_reliable_gold_standard_classification_with_additional_train_data/French"
    train_french, dev_french, test_french = load_and_preprocess_data(path_to_data, path_to_labels_french,
                                                                     frame_step)
    # english data
    path_to_labels_german = "/media/external_hdd_1/NoXi_annotations_reliable_gold_standard_classification_with_additional_train_data/German"
    train_german, dev_german, test_german = load_and_preprocess_data(path_to_data, path_to_labels_german,
                                                                     frame_step)
    # german data
    path_to_labels_english = "/media/external_hdd_1/NoXi_annotations_reliable_gold_standard_classification_with_additional_train_data/English"
    train_english, dev_english, test_english = load_and_preprocess_data(path_to_data, path_to_labels_english,
                                                                        frame_step)
    # all data
    train = pd.concat([train_french, train_german, train_english], axis=0)
    dev = pd.concat([dev_french, dev_german, dev_english], axis=0)
    test = pd.concat([test_french, test_german, test_english], axis=0)
    # shuffle one more time train data
    train = train.sample(frac=1).reset_index(drop=True)
    # clear RAM
    del train_english, train_french, train_german
    del dev_english, dev_french, dev_german
    del test, test_english, test_german, test_french
    gc.collect()
    print(dev)
    train = train.iloc[:1000]

    train_model(train, dev)


if __name__ == '__main__':
    main()
