#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Contains the script to investigate the influence of the emotional embeddings on the engagement recognition task.
"""

__author__ = "Denis Dresvyanskiy"
__copyright__ = "Copyright 2022"
__credits__ = ["Denis Dresvyanskiy"]
__maintainer__ = "Denis Dresvyanskiy"
__email__ = "denis.dresvyanskiy@uni-ulm.de"

import gc
import sys
from typing import Tuple, Dict

import pandas as pd
import wandb
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

from src.IWSDS2023.visual_subsystem.facial_subsystem.sequence_models.sequence_data_loader import load_embeddings_from_csv_file, \
    split_embeddings_according_to_file_path, load_data
from src.IWSDS2023.visual_subsystem.facial_subsystem.sequence_models.sequence_model_training import train_model


def load_emotional_embeddings() -> Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]:
    path_to_emotional_embeddings_train = "/work/home/dsu/NoXi_embeddings/All_languages/EmoVGGFace2/train_extracted_deep_embeddings.csv"
    path_to_emotional_embeddings_dev = "/work/home/dsu/NoXi_embeddings/All_languages/EmoVGGFace2/dev_extracted_deep_embeddings.csv"
    path_to_emotional_embeddings_test = "/work/home/dsu/NoXi_embeddings/All_languages/EmoVGGFace2/test_extracted_deep_embeddings.csv"

    # load embeddings
    train_embeddings = load_embeddings_from_csv_file(path_to_emotional_embeddings_train)
    dev_embeddings = load_embeddings_from_csv_file(path_to_emotional_embeddings_dev)
    test_embeddings = load_embeddings_from_csv_file(path_to_emotional_embeddings_test)
    # normalization
    scaler = StandardScaler()
    scaler = scaler.fit(train_embeddings.iloc[:, 1:-5])
    train_embeddings.iloc[:, 1:-5] = scaler.transform(train_embeddings.iloc[:, 1:-5])
    dev_embeddings.iloc[:, 1:-5] = scaler.transform(dev_embeddings.iloc[:, 1:-5])
    test_embeddings.iloc[:, 1:-5] = scaler.transform(test_embeddings.iloc[:, 1:-5])

    # split embeddings
    train_embeddings_split = split_embeddings_according_to_file_path(train_embeddings)
    dev_embeddings_split = split_embeddings_according_to_file_path(dev_embeddings)
    test_embeddings_split = split_embeddings_according_to_file_path(test_embeddings)

    return (train_embeddings_split, dev_embeddings_split, test_embeddings_split)


def concat_embeddings(embeddings_to_concat) -> Tuple[
    Dict[str, pd.DataFrame], Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]:
    concatenated_emb = embeddings_to_concat[0]
    labels_columns = ["label_0", "label_1", "label_2", "label_3", "label_4"]
    for dataset_num in range(len(concatenated_emb)):
        for video_filename in concatenated_emb[dataset_num]:
            for num_embeddings in range(1, len(embeddings_to_concat)):
                left_df = concatenated_emb[dataset_num][video_filename].set_index("frame_id")
                right_df = embeddings_to_concat[num_embeddings][dataset_num][video_filename]
                # drop some columns (they are duplicated) and change a little bit the right_df
                right_df = right_df.set_index("frame_id").drop(columns=labels_columns)
                right_df = right_df.drop(columns=["video_filename"])
                new_columns = list(right_df.columns)
                new_columns = [item.replace("embedding_", "emotional_emb_") for item in new_columns]
                right_df.columns = new_columns
                # concatenate dataframes
                concatenated_emb[dataset_num][video_filename] = pd.concat([left_df, right_df], axis=1,
                                                                          join="inner").reset_index()
                # change the order of columns for convenience
                old_columns = list(concatenated_emb[dataset_num][video_filename].columns)
                needed_columns = ["video_filename", "frame_id"]
                needed_columns = needed_columns + [item for item in old_columns if
                                                   item not in ["video_filename", "frame_id"] + labels_columns]
                needed_columns = needed_columns + labels_columns
                concatenated_emb[dataset_num][video_filename] = concatenated_emb[dataset_num][video_filename][
                    needed_columns]

    return concatenated_emb


def run_sweep(sweep_name: str, window_length: int) -> None:
    """Runs the sweep (for hyperparameter search) using the Weights and Biases lib.

    :param sweep_name: str
                The name of the sweep you want to be.
    :param window_length: int
                The length of the sequences (windows). This is to generate the sequences of fixed length (for RNN model).

    :return: None
    """
    print("Script with emotional embeddings")
    # load the data and labels, and concat them
    train_emotional, dev_emotional, test_emotional = load_emotional_embeddings()
    train_Xception, dev_Xception, test_Xception = load_data()
    train, dev, test = concat_embeddings(
        [(train_Xception, dev_Xception, test_Xception), (train_emotional, dev_emotional, test_emotional)])
    # clear RAM
    del train_emotional, dev_emotional, test_emotional
    del train_Xception, dev_Xception, test_Xception
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


if __name__ == '__main__':
    print("CATEGORICAL RUN")
    run_sweep(sweep_name="categorical_crossentropy_loss_window_length_%i" % (int(sys.argv[1])),
              window_length=int(sys.argv[1]))

