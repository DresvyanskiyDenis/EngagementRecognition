#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Contains the script for model validation on the IWSDS2023 development and test sets.

"""

__author__ = "Denis Dresvyanskiy"
__copyright__ = "Copyright 2022"
__credits__ = ["Denis Dresvyanskiy"]
__maintainer__ = "Denis Dresvyanskiy"
__email__ = "denis.dresvyanskiy@uni-ulm.de"

import sys
sys.path.extend(["/work/home/dsu/datatools/"])
sys.path.extend(["/work/home/dsu/engagement_recognition_project_server/"])

import tensorflow as tf
import gc
import numpy as np
from functools import partial
from sklearn.metrics import recall_score, precision_score, f1_score, confusion_matrix, accuracy_score

from src.IWSDS2023.visual_subsystem.facial_subsystem.frame_wise_models.utils import load_NoXi_data_all_languages, \
    load_NoXi_data_cross_corpus
from tensorflow_utils.tensorflow_datagenerators.ImageDataLoader_tf2 import get_tensorflow_image_loader
from tensorflow_utils.tensorflow_datagenerators.tensorflow_image_preprocessing import preprocess_data_Xception
from src.IWSDS2023.visual_subsystem.facial_subsystem.frame_wise_models.Cross_language_training import create_Xception_model



def validate_model(model:tf.keras.Model, generator)->None:
    """Validates provided model using generator (preferable tf.Dataset).
       Validation is done applying the following metrics: [accuracy, precision, recall, f1_score, confusion matrix]
    :param model: tf.Model
                Tensorflow model
    :param generator: Iterator
                Any generator that produces the output as a tuple: (features, labels)
    :return: None
    """

    metrics = {
        'val_accuracy': accuracy_score,
        'val_recall': partial(recall_score, average='macro'),
        'val_precision': partial(precision_score, average='macro'),
        'val_f1_score:': partial(f1_score, average='macro'),
        'confusion_matrix': confusion_matrix,
    }
    # make predictions for data from generator and save ground truth labels
    total_predictions = np.zeros((0,))
    total_ground_truth = np.zeros((0,))
    for x, y in generator.as_numpy_iterator():
        predictions = model.predict(x, batch_size=64)
        predictions = predictions.argmax(axis=-1).reshape((-1,))
        total_predictions = np.append(total_predictions, predictions)
        total_ground_truth = np.append(total_ground_truth, y.argmax(axis=-1).reshape((-1,)))
    # calculate all provided metrics and save them as dict object
    # as Dict[metric_name->value]
    results = {}
    for key in metrics.keys():
        results[key] = metrics[key](total_ground_truth, total_predictions)
    # clear RAM
    del total_predictions, total_ground_truth
    gc.collect()
    conf_m=results.pop('confusion_matrix')
    # print results
    for key, value in results.items():
        print("Metric: %s, result:%f"%(key, value))
    # print confusion matrix separately one more time for better vision
    print("Confusion matrix")
    print(conf_m)



def main():
    print("Start....9 English")
    # params
    train, dev, test = load_NoXi_data_cross_corpus(test_corpus='english')
    path_to_model_weights="/work/home/dsu/Model_weights/weights_of_best_models/frame_to_frame_experiments/Facial_model/Cross_language/English_9.h5"
    preprocess_function = preprocess_data_Xception
    batch_size = 32
    model_creation_function = create_Xception_model
    gc.collect()

    # transform labels in dev data to one-hot encodings
    dev_data_loader = get_tensorflow_image_loader(paths_and_labels=dev, batch_size=batch_size,
                                                 augmentation=False,
                                                 augmentation_methods=None,
                                                 preprocessing_function=preprocess_function,
                                                 clip_values=None,
                                                 cache_loaded_images=False)

    # transform labels in dev data to one-hot encodings
    test_data_loader = get_tensorflow_image_loader(paths_and_labels=test, batch_size=batch_size,
                                               augmentation=False,
                                               augmentation_methods=None,
                                               preprocessing_function=preprocess_function,
                                               clip_values=None,
                                               cache_loaded_images=False)

    # model initialization
    model = model_creation_function(num_classes=5)
    # load weights of trained model
    model.load_weights(path_to_model_weights)
    # model compilation
    model.compile(loss='categorical_crossentropy', optimizer='Adam')
    model.summary()

    validate_model(model, dev_data_loader)
    print("----------------------------test--------------------------------")
    validate_model(model, test_data_loader)

if __name__ == '__main__':
    main()