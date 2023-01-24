#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Contains the script for validation and test the sequence model (tf.keras.Model) on the IWSDS2023 data.

"""

__author__ = "Denis Dresvyanskiy"
__copyright__ = "Copyright 2022"
__credits__ = ["Denis Dresvyanskiy"]
__maintainer__ = "Denis Dresvyanskiy"
__email__ = "denis.dresvyanskiy@uni-ulm.de"

import os
import sys
sys.path.extend(["/work/home/dsu/datatools/"])
sys.path.extend(["/work/home/dsu/engagement_recognition_project_server/"])

import gc
import numpy as np
import pandas as pd
import tensorflow as tf
from typing import Optional, Tuple, Dict

from functools import partial
from sklearn.metrics import recall_score, precision_score, f1_score, accuracy_score, confusion_matrix


from tensorflow_utils.models.sequence_to_one_models import create_simple_RNN_network
from src.IWSDS2023.visual_subsystem.facial_subsystem.sequence_models.sequence_data_loader import \
    create_generator_from_pd_dictionary, load_data, load_data_cross_corpus


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
                                               dropout=True)
    return sequence_model


def validate_model_on_generator(model: tf.keras.Model, generator) -> None:
    """Validates provided model using generator (preferable tf.Dataset).
       Validation is done applying the following metrics: [accuracy, precision, recall, f1_score, confusion matrix]
    :param model: tf.Model
                Tensorflow model
    :param generator: Iterator
                Any generator that produces the output as a tuple: (features, labels)
    :return: None
    """

    metrics = {
        'accuracy': accuracy_score,
        'recall': partial(recall_score, average='macro'),
        'precision': partial(precision_score, average='macro'),
        'f1_score': partial(f1_score, average='macro'),
        'confusion_matrix': confusion_matrix,
    }
    # make predictions for data from generator and save ground truth labels
    total_predictions = np.zeros((0,))
    total_ground_truth = np.zeros((0,))
    for x, y in generator.as_numpy_iterator():
        predictions = model.predict(x, batch_size=64)
        predictions = predictions.squeeze().argmax(axis=-1).reshape((-1,))
        total_predictions = np.append(total_predictions, predictions)
        total_ground_truth = np.append(total_ground_truth, y.squeeze().argmax(axis=-1).reshape((-1,)))
    # calculate all provided metrics and save them as dict object
    # as Dict[metric_name->value]
    results = {}
    for key in metrics.keys():
        results[key] = metrics[key](total_ground_truth, total_predictions)
    # clear RAM
    del total_predictions, total_ground_truth
    gc.collect()
    conf_m = results.pop('confusion_matrix')
    # print results
    for key, value in results.items():
        print("Metric: %s, result:%f" % (key, value))
    # print confusion matrix separately one more time for better vision
    print("Confusion matrix")
    print(conf_m)
    return results


def validate_model(dev: Dict[str, pd.DataFrame], test: Dict[str, pd.DataFrame], weights_path: str,
                   window_length: int, num_neurons: Tuple[int, ...], num_layers: Tuple[int, ...]):
    """ Validates the model on the dev and test sets based on the provided parameters.
        Window_length is needed to define the data generator
        num_neurons, num_layers are needed to define the RNN sequence-to-one model

    :param dev: Dict[str, pd.DataFrame]
                The dictionary in the format Dict[path_to_video_file->pd.DataFrame]
                The DataFrame has columns: [video_filename, frame_id, embedding_0, embedding_1, ..., label_0, label_1, ...]
                This is the dataset for validation the model.
    :param test: Dict[str, pd.DataFrame]
                The dictionary in the format Dict[path_to_video_file->pd.DataFrame]
                The DataFrame has columns: [video_filename, frame_id, embedding_0, embedding_1, ..., label_0, label_1, ...]
                This is the dataset for testing the model.
    :param weights_path: str
                Path to the weights of the model to be validated
    :param window_length: int
                The length of the sequences (windows). THis is to generate the sequences of fixed length (for RNN model).
    :param neurons_on_layer: Tuple[int,...]
            The number of neurons on yeach layers passed as a tuple.
    :param num_classes: int
            The number of classes in the classification task to make a final softmax layer with exact same
            number of neurons
    :return: Tuple[Dict[str, float], Dict[str, float]]
            The results of validation and testing the model on the dev and test sets. Based on well-known metrics.
            The results for every dataset are contained in dictionaries.
            Keys are: accuracy, recall, precision, f1_score
    """
    # model params
    num_classes = 5
    num_embeddings = 256
    window_stride = 1
    batch_size = 16
    type_of_labels = 'sequence_to_one'
    loss = 'categorical_crossentropy'
    optimizer = 'Adam'
    metrics = None

    # model initialization
    model = create_sequence_model(num_classes=num_classes,
                                  neurons_on_layer=tuple(num_neurons for i in range(num_layers)),
                                  input_shape=(window_length, num_embeddings))

    # load weights
    model.load_weights(weights_path)

    # model compilation
    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
    model.summary()

    # transform labels in dev data to one-hot encodings
    dev_data_loader = create_generator_from_pd_dictionary(embeddings_dict=dev, num_classes=num_classes,
                                                          type_of_labels=type_of_labels,
                                                          window_length=window_length, window_shift=window_length,
                                                          window_stride=window_stride, batch_size=batch_size,
                                                          shuffle=False,
                                                          preprocessing_function=None,
                                                          clip_values=None,
                                                          cache_loaded_seq=None
                                                          )

    # transform labels in dev data to one-hot encodings
    test_data_loader = create_generator_from_pd_dictionary(embeddings_dict=test, num_classes=num_classes,
                                                           type_of_labels=type_of_labels,
                                                           window_length=window_length, window_shift=window_length,
                                                           window_stride=window_stride, batch_size=batch_size,
                                                           shuffle=False,
                                                           preprocessing_function=None,
                                                           clip_values=None,
                                                           cache_loaded_seq=None
                                                           )

    print("00000000000000000000000000000000000000000000000000000000")
    print(weights_path)
    dev_results = validate_model_on_generator(model, dev_data_loader)
    print('--------------------test-------------------')
    test_results = validate_model_on_generator(model, test_data_loader)
    print("00000000000000000000000000000000000000000000000000000000")
    return (dev_results, test_results)


def main():
    # params
    test_language = 'english'
    print('Start of the script...English')
    train, dev, test = load_data_cross_corpus(test_corpus=test_language)
    dir_with_models = "/work/home/dsu/Model_weights/weights_of_best_models/sequence_to_one_experiments/Cross_corpus/Facial_model/english/"
    params_for_testing_models = pd.read_csv(os.path.join(dir_with_models, "params_of_testing.csv"), delimiter=',').iloc[:20]
    params_for_testing_models["weights_path"] = params_for_testing_models.apply(lambda x:test_language+'_'+str(int(x['window_length']))+'_'+str(int(x["sweep_id"])) + ".h5",
                                                                                axis=1)

    results = pd.DataFrame(columns=["weights_path", "val_recall", "val_precision", "val_f1_score", "val_accuracy",
                                    "test_recall", "test_precision", "test_f1_score", "test_accuracy"])
    # calculate results
    for index, row in params_for_testing_models.iterrows():
        dev_res, test_res = validate_model(dev, test, dir_with_models + row["weights_path"], int(row["window_length"]),
                                           int(row["num_neurons"]), int(row["num_layers"]))
        results = results.append({"weights_path": row["weights_path"],
                                  "val_recall": dev_res["recall"], "val_precision": dev_res["precision"],
                                  "val_f1_score": dev_res["f1_score"], "val_accuracy": dev_res["accuracy"],
                                  "test_recall": test_res["recall"], "test_precision": test_res["precision"],
                                  "test_f1_score": test_res["f1_score"], "test_accuracy": test_res["accuracy"]},
                                 ignore_index=True)
    # write results to the csv file
    results.to_csv(os.path.join(dir_with_models, "results_of_testing.csv"), index=False)


if __name__ == '__main__':
    main()
