#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
TODO: add description
"""
from typing import Tuple

import numpy as np
import os
import pandas as pd
import shutil
import tensorflow as tf
from sklearn.utils import class_weight

from preprocessing.data_preprocessing.openFace_utils import extract_openface_FAU_from_images_in_dir
from src.DAiSEE.engagementRecognition_DAiSEE import load_labels_to_dict
from tensorflow_utils.callbacks import get_annealing_LRreduce_callback
from tensorflow_utils.models.Dense_models import get_Dense_model


def extract_FAU_features_from_all_subdirectories(path_to_dirs:str, path_to_extractor:str)->pd.DataFrame:
    # find all subdirectories in provided directory
    subdirectories=os.listdir(path_to_dirs)
    # create empty list to collect then all extracted features
    extracted_features=[]
    counter=0
    for subdirectory in subdirectories:
        # construct full path to subdirectory
        path_to_images=os.path.join(path_to_dirs, subdirectory)
        # extract features by OpenFace
        features=extract_openface_FAU_from_images_in_dir(path_to_images, path_to_extractor)
        if features is not None:
            extracted_features.append(features)
        print('----------------------------------%i, %i'%(counter, len(subdirectories)))
        counter+=1
    # concat all extracted features
    extracted_features=pd.concat(extracted_features, axis=0)
    return extracted_features

def add_labels_to_FAU_features_in_df(features:pd.DataFrame, path_to_labels:str,
                                     features_to_add:Tuple[str,...]=('engagement', 'boredom',
                                                      'confusion','frustration'))->pd.DataFrame:
    # TODO: write description
    labels=load_labels_to_dict(path_to_labels)
    # fill with NaN just for a while
    for feature_to_add in features_to_add:
        features[feature_to_add]=np.NaN
    # iterate through dataframe
    for row_idx in range(features.shape[0]):
        # get filename without frame number (to get access to the labels by filename)
        filename=features.iloc[row_idx]['filename'].split('_')[0]
        for feature_to_add in features_to_add:
            # add required labels to the row (change from NaN to the label value)
            features.iloc[row_idx][feature_to_add]=labels[filename][feature_to_add]
    return features


if __name__=="__main__":
    path_to_dir = r'E:\Databases\DAiSEE\DAiSEE\train_preprocessed\extracted_faces'
    path_to_extractor = r'C:\Users\Denis\PycharmProjects\OpenFace\FaceLandmarkImg.exe'
    features = extract_FAU_features_from_all_subdirectories(path_to_dir, path_to_extractor)
    features.to_csv(r'E:\Databases\DAiSEE\DAiSEE\dev_processed\FAU_features.csv', index=False)
    # !!!! do not forget to rename file!!! you saved train features to the dev_preprocess directory
    """
    features=pd.read_scv(r'E:\Databases\DAiSEE\DAiSEE\train_preprocessed\FAU_features.csv')
    features=add_labels_to_FAU_features_in_df(features, r'C:\Databases\DAiSEE\Labels\TrainLabels.csv')
    features.to_csv(r'E:\Databases\DAiSEE\DAiSEE\train_preprocessed\FAU_features_with_labels.csv')
    """
    # params
    num_classes = 4
    batch_size = 256
    epochs = 100
    highest_lr = 0.001
    lowest_lr = 0.0001
    momentum = 0.9
    optimizer = tf.keras.optimizers.SGD(highest_lr, momentum=momentum)
    loss = tf.keras.losses.sparse_categorical_crossentropy
    callbacks = [get_annealing_LRreduce_callback(highest_lr, lowest_lr, 5)]
    # create metrics
    metrics = [tf.keras.metrics.CategoricalAccuracy(), tf.keras.metrics.Recall()]

    input_shape=(features.shape[1]-5,)
    model=get_Dense_model(input_shape=input_shape,
                    dense_neurons=(128,64,32),
                    activations='relu',
                    dropout= 0.3,
                    regularization=tf.keras.regularizers.l2(0.0001),
                    output_neurons = 4,
                    activation_function_for_output='softmax')
    model.summary()

    # compute class weights
    # class weights
    class_weights_engagement = class_weight.compute_class_weight(class_weight='balanced',
                                                                 classes=np.unique(features['engagement']),
                                                                 y=features['engagement'].values.reshape((-1,)))
    class_weights_engagement /= class_weights_engagement.sum()
    class_weights_engagement=dict((i,class_weights_engagement[i]) for i in range(len(class_weights_engagement)))
    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
    model.fit(x=features.iloc[:-5], y=features['engagement'],epochs=epochs, batch_size=batch_size,callbacks=callbacks)