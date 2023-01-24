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
from sklearn.metrics import accuracy_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.utils import class_weight

from feature_extraction.openFace_utils import extract_openface_FAU_from_images_in_dir
from src.SPECOM2021.engagementRecognition_DAiSEE import load_labels_to_dict
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
        if filename not in labels.keys():
            continue
        for feature_to_add in features_to_add:
            # add required labels to the row (change from NaN to the label value)
            features[feature_to_add].iloc[row_idx]=labels[filename]._asdict()[feature_to_add]
        if row_idx%10000==0:
            print(row_idx)
    return features

class val_callback(tf.keras.callbacks.Callback):
    """Calculates the recall score at the end of each training epoch and saves the best weights across all the training
        process. At the end of training process, it will set weights of the model to the best found ones.
        # TODO: write, which types of metric functions it supports
    """

    def __init__(self, val_data:pd.DataFrame):
        super(val_callback, self).__init__()
        self.val_data = val_data

    def on_train_begin(self, logs=None):
        # Initialize the best as infinity.
        self.best_f1 = 0
        self.best_recall = 0
        self.best_acc = 0

    def on_train_end(self, logs=None):
        print('best f1_score on validation:', self.best_f1)
        print('best recall on validation:', self.best_recall)
        print('best accuracy on validation:', self.best_acc)

    def on_epoch_end(self, epoch, logs=None):
        # validation
        val_predictions = self.model.predict(x=self.val_data.iloc[:, :-5], batch_size=256)
        val_predictions = np.argmax(val_predictions, axis=-1).reshape((-1, 1))
        val_ground_truth = self.val_data['engagement'].values.reshape((-1, 1))
        acc=accuracy_score(val_ground_truth, val_predictions)
        recall=recall_score(val_ground_truth, val_predictions,average='macro')
        f1=f1_score(val_ground_truth, val_predictions, average='macro')
        print('\nval accuracy:', acc)
        print('val recall:', recall)
        print('val f1_score:', f1)
        if acc>self.best_acc: self.best_acc=acc
        if recall>self.best_recall: self.best_recall=recall
        if f1>self.best_f1: self.best_f1=f1


if __name__=="__main__":
    path_to_dir = r'D:\Databases\DAiSEE\DAiSEE\test_preprocessed\extracted_faces'
    path_to_extractor = r'C:\Users\Dresvyanskiy\Desktop\Projects\OpenFace\FaceLandmarkImg.exe'
    features = extract_FAU_features_from_all_subdirectories(path_to_dir, path_to_extractor)
    features.to_csv(r'D:\Databases\DAiSEE\DAiSEE\test_preprocessed\FAU_features.csv', index=False)

    """features=pd.read_csv(r'E:\Databases\DAiSEE\DAiSEE\dev_processed\FAU_features.csv')
    features=add_labels_to_FAU_features_in_df(features, r'E:\Databases\DAiSEE\DAiSEE\Labels\ValidationLabels.csv')
    features.to_csv(r'E:\Databases\DAiSEE\DAiSEE\dev_processed\FAU_features_with_labels.csv')"""

    '''path_to_train_features=r'E:\Databases\DAiSEE\DAiSEE\train_preprocessed\FAU_features_with_labels.csv'
    path_to_dev_features = r'E:\Databases\DAiSEE\DAiSEE\dev_processed\FAU_features_with_labels.csv'
    train_features=pd.read_csv(path_to_train_features)
    train_features=train_features[~train_features['engagement'].isna()]
    train_features=train_features.drop(columns=['Unnamed: 0'])
    dev_features=pd.read_csv(path_to_dev_features)
    dev_features = dev_features[~dev_features['engagement'].isna()]
    dev_features = dev_features.drop(columns=['Unnamed: 0'])
    # params
    num_classes = 4
    batch_size = 128
    epochs = 100
    highest_lr = 0.0001
    lowest_lr = 0.0001
    momentum = 0.9
    optimizer = tf.keras.optimizers.Adam(highest_lr)
    loss = tf.keras.losses.categorical_crossentropy
    callbacks = [val_callback(dev_features)]
    # create metrics
    metrics = [tf.keras.metrics.CategoricalAccuracy(), tf.keras.metrics.Recall()]

    input_shape=(train_features.shape[1]-5,)
    model=get_Dense_model(input_shape=input_shape,
                    dense_neurons=(256,512,256),
                    activations='elu',
                    dropout= 0.3,
                    regularization=tf.keras.regularizers.l2(0.0001),
                    output_neurons = 4,
                    activation_function_for_output='softmax')
    model.summary()

    # compute class weights
    # class weights
    class_weights_engagement = class_weight.compute_class_weight(class_weight='balanced',
                                                                 classes=np.unique(train_features['engagement']),
                                                                 y=train_features['engagement'].values.reshape((-1,)))
    #class_weights_engagement /= class_weights_engagement.sum()
    class_weights_engagement=dict((i,class_weights_engagement[i]) for i in range(len(class_weights_engagement)))
    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
    # shuffle train
    train_features=train_features.sample(frac=1)
    model.fit(x=StandardScaler().fit_transform(train_features.iloc[:,:-5].values),
              y=tf.keras.utils.to_categorical(train_features['engagement'].values, num_classes=num_classes),
              epochs=epochs,
              batch_size=batch_size, callbacks=callbacks,
              validation_data=(StandardScaler().fit_transform(dev_features.iloc[:,:-5].values),
                               tf.keras.utils.to_categorical(dev_features['engagement'].values, num_classes=num_classes))
              #class_weight=class_weights_engagement
              )'''
