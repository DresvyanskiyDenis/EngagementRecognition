#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Contains the script for training the sequence model (tf.keras.Model) on the IWSDS2023 dataset.
   The Weights and Biases library for the hyperparameter search is applied.

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
from typing import Optional, Tuple, Dict

from keras.callbacks import EarlyStopping
from wandb.integration.keras import WandbCallback
from functools import partial
from sklearn.metrics import recall_score, precision_score, f1_score

from src.IWSDS2023.visual_subsystem.facial_subsystem.sequence_models.sequence_data_loader import create_generator_from_pd_dictionary, \
    load_data
from tensorflow_utils.models.sequence_to_one_models import create_simple_RNN_network
from tensorflow_utils.Losses import categorical_focal_loss
from tensorflow_utils.wandb_callbacks import WandB_LR_log_callback, WandB_val_metrics_callback
from tensorflow_utils.callbacks import get_annealing_LRreduce_callback, get_reduceLRonPlateau_callback


def create_sequence_model(input_shape: Tuple[int, ...], neurons_on_layer: Tuple[int, ...],
                          num_classes: Optional[int] = 5) -> tf.keras.Model:
    """ Creates a sequence model based on the passed parameters.

    :param input_shape: Tuple[int,...]
            The input shape for the neural network
    :param neurons_on_layer: Tuple[int,...]
            The number of neurons on yeach layers passed as a tuple.
    :param num_classes: int
            THe number of classes in the classification task to make a final softmax layer with exact same
            number of neurons
    :return: tf.keras.Model
            RNN sequence-to-one model
    """

    sequence_model = create_simple_RNN_network(input_shape=input_shape, num_output_neurons=num_classes,
                                               neurons_on_layer=neurons_on_layer,
                                               rnn_type='LSTM',
                                               dnn_layers=(128,),
                                               need_regularization=True,
                                               dropout=True,
                                               output_activation='softmax')
    return sequence_model


def train_model(train: Dict[str, pd.DataFrame], dev: Dict[str, pd.DataFrame],
                loss_func: str = 'categorical_crossentropy') -> None:
    """ Does a hyperparameter search training the model many times varying the parameters defined by Weights and
    Biases lib (in the main function).

    :param train: Dict[str, pd.DataFrame]
                The dictionary in the format Dict[path_to_video_file->pd.DataFrame]
                The DataFrame has columns: [video_filename, frame_id, embedding_0, embedding_1, ..., label_0, label_1, ...]
                This is the dataset for model training.
    :param dev: Dict[str, pd.DataFrame]
                The dictionary in the format Dict[path_to_video_file->pd.DataFrame]
                The DataFrame has columns: [video_filename, frame_id, embedding_0, embedding_1, ..., label_0, label_1, ...]
                This is the dataset for validation of the model during training process.
    :param loss_func: str
                The type of loss function for training.
                Now only "categorical_crossentropy" and "focal_loss" are available.
    :return: None
    """
    # metaparams
    metaparams = {
        "optimizer": "Adam",  # SGD, Nadam
        "learning_rate_max": 0.001,  # up to 0.0001
        "learning_rate_min": 0.00001,  # up to 0.000001
        "lr_scheduller": "reduceLRonPlateau",  # "reduceLRonPlateau"
        "annealing_period": 10,
        "epochs": 100,
        "batch_size": 16,
        "architecture": "LSTM_no_attention",
        "dataset": "IWSDS2023",
        'type_of_labels': 'sequence_to_one',
        "num_classes": 5,
        'num_embeddings': 256,
        'num_layers': 3,
        'num_neurons': 256,
        'window_length': 40,
        'window_shift': 10,
        'window_stride': 1,
    }

    # initialization of Weights and Biases
    wandb.init(project="NoXi_Seq_to_One", config=metaparams)
    config = wandb.config

    # Metaparams initialization
    metrics = ['accuracy']
    if config.lr_scheduller == 'Cyclic':
        lr_scheduller = get_annealing_LRreduce_callback(highest_lr=config.learning_rate_max,
                                                        lowest_lr=config.learning_rate_min,
                                                        annealing_period=config.annealing_period)
    elif config.lr_scheduller == 'reduceLRonPlateau':
        lr_scheduller = get_reduceLRonPlateau_callback(monitoring_loss='val_loss', reduce_factor=0.1,
                                                       num_patient_epochs=7,
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
    class_weights = pd.concat(train.values(), axis=0).iloc[:, -metaparams['num_classes']:].values.sum(axis=0)
    class_weights = class_weights.sum() / (metaparams['num_classes'] + class_weights)
    # class_weights = class_weights / class_weights.sum()

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
    model = create_sequence_model(num_classes=config.num_classes,
                                  neurons_on_layer=tuple(config.num_neurons for i in range(config.num_layers)),
                                  input_shape=(config.window_length, config.num_embeddings))
    # freezing layers?
    for i, layer in enumerate(model.layers):
        print("%i:%s" % (i, layer.name))

    # model compilation
    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
    model.summary()

    # create DataLoaders (DataGenerator)
    train_data_loader = create_generator_from_pd_dictionary(embeddings_dict=train, num_classes=config.num_classes,
                                                            type_of_labels=config.type_of_labels,
                                                            window_length=config.window_length,
                                                            window_shift=int(np.round(config.window_length * 0.5)),
                                                            window_stride=config.window_stride,
                                                            batch_size=config.batch_size, shuffle=True,
                                                            preprocessing_function=None,
                                                            clip_values=None,
                                                            cache_loaded_seq=None
                                                            )

    # transform labels in dev data to one-hot encodings
    dev_data_loader = create_generator_from_pd_dictionary(embeddings_dict=dev, num_classes=config.num_classes,
                                                          type_of_labels=config.type_of_labels,
                                                          window_length=config.window_length,
                                                          window_shift=config.window_length,
                                                          window_stride=config.window_stride,
                                                          batch_size=config.batch_size, shuffle=False,
                                                          preprocessing_function=None,
                                                          clip_values=None,
                                                          cache_loaded_seq=None
                                                          )

    # create Keras Callbacks for monitoring learning rate and metrics on val_set
    lr_monitor_callback = WandB_LR_log_callback()
    val_metrics = {
        'val_recall': partial(recall_score, average='macro'),
        'val_precision': partial(precision_score, average='macro'),
        'val_f1_score:': partial(f1_score, average='macro')
    }
    val_metrics_callback = WandB_val_metrics_callback(dev_data_loader, val_metrics,
                                                      metric_to_monitor='val_recall',
                                                      log_class_distribution=True)
    early_stopping_callback = EarlyStopping(monitor='val_loss', patience=20, verbose=1)

    # train process
    print("Loss used:%s" % (loss))
    print("SEQUENCE TO ONE MODEL")
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


def run_sweep(sweep_name: str, window_length: int) -> None:
    """Runs the sweep (for hyperparameter search) using the Weights and Biases lib.

    :param sweep_name: str
                The name of the sweep you want to be.
    :param window_length: int
                The length of the sequences (windows). This is to generate the sequences of fixed length (for RNN model).

    :return: None
    """
    print("START OF SCRIPT111111111")
    # gpus = tf.config.experimental.list_physical_devices('GPU')
    # for gpu in gpus:
    #    tf.config.experimental.set_memory_growth(gpu, True)
    # load the data and labels
    train, dev, test = load_data()
    gc.collect()

    sweep_config = {
        'name': sweep_name,
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
                'max': 0.01,
                'min': 0.0001
            },
            'learning_rate_min': {
                'distribution': 'uniform',
                'max': 0.0001,
                'min': 0.000001
            },
            'lr_scheduller': {
                'values': ['Cyclic', 'reduceLRonPlateau']
            },
            'num_layers': {
                'values': [1, 2, 3]
            },
            'num_neurons': {
                'values': [64, 128, 256, 512]
            },
            'window_length': {
                'values': [window_length]
            }
        }
    }

    # categorical_crossentropy
    sweep_id = wandb.sweep(sweep_config, project='NoXi_Seq_to_One')
    wandb.agent(sweep_id, function=lambda: train_model(train, dev, 'categorical_crossentropy'), count=195,
                project='NoXi_Seq_to_One')
    tf.keras.backend.clear_session()
    gc.collect()


def main():
    print("CATEGORICAL RUN")
    run_sweep(sweep_name="categorical_crossentropy_loss_window_length_%i" % (int(sys.argv[1])),
              window_length=int(sys.argv[1]))


if __name__ == '__main__':
    main()
