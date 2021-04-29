#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
TODO: write description of module
"""
import math
import shutil
import time
from functools import partial
from typing import Optional, Tuple, Dict, NamedTuple, Iterable, List

import numpy as np
import pandas as pd
import os
import tensorflow as tf
from sklearn.metrics import recall_score, f1_score, accuracy_score
from sklearn.utils import class_weight

from keras_datagenerators import ImageDataLoader
from keras_datagenerators.ImageDataLoader_multilabel import ImageDataLoader_multilabel
from preprocessing.data_normalizing_utils import VGGFace2_normalization
from preprocessing.data_preprocessing.image_preprocessing_utils import save_image
from src.engagementRecognition_DAiSEE import load_labels_to_dict, \
    form_dataframe_of_relative_paths_to_data_with_multilabels
from tensorflow_utils.callbacks import best_weights_setter_callback, get_annealing_LRreduce_callback, \
    validation_with_generator_callback, validation_with_generator_callback_multilabel
from tensorflow_utils.models.CNN_models import get_modified_VGGFace2_resnet_model


def save_batch_of_images(path_to_save:str, images:np.ndarray, names:List[str])->None:
    if not os.path.exists(path_to_save):
        os.makedirs(path_to_save, exist_ok=True)
    if images.shape[0]!=len(names):
        raise AttributeError('array images and names should have the same length. Got images:%i, names:%i.'%(images.shape[0], len(names)))
    for i in range(images.shape[0]):
        save_image(images[i], path_to_output=names[i])



if __name__=='__maim__':
    # augment minority classes
    # params
    path_to_train_frames = r'C:\Databases\DAiSEE\train_preprocessed\extracted_faces'
    path_to_train_labels = r'C:\Databases\DAiSEE\Labels\TrainLabels.csv'
    path_to_dev_frames = r'C:\Databases\DAiSEE\dev_preprocessed\extracted_faces'
    path_to_dev_labels = r'C:\Databases\DAiSEE\Labels\ValidationLabels.csv'
    num_classes=4
    batch_size=64
    selected_class_to_augment=0
    num_extra_images=4000
    label_type_to_augment='engagement'
    output_path=r'C:\Databases\DAiSEE\train_preprocessed\augmentation'
    output_path=os.path.join(output_path, label_type_to_augment, str(selected_class_to_augment))

    # load labels
    dict_labels_train = load_labels_to_dict(path_to_train_labels)
    dict_labels_dev = load_labels_to_dict(path_to_dev_labels)
    # form dataframes with relative paths and labels
    labels_train = form_dataframe_of_relative_paths_to_data_with_multilabels(path_to_train_frames, dict_labels_train)
    labels_dev = form_dataframe_of_relative_paths_to_data_with_multilabels(path_to_dev_frames, dict_labels_dev)
    # add full path to them
    labels_train['filename'] = path_to_train_frames + '\\' + labels_train['filename']
    labels_dev['filename'] = path_to_dev_frames + '\\' + labels_dev['filename']
    labels_train['engagement'] = labels_train['engagement'].astype('float32')
    labels_train['boredom'] = labels_train['boredom'].astype('float32')
    labels_dev['engagement'] = labels_dev['engagement'].astype('float32')
    labels_dev['boredom'] = labels_dev['boredom'].astype('float32')
    # only labels with selected class remain
    labels_train=labels_train[labels_train[label_type_to_augment]==selected_class_to_augment]

    generator = ImageDataLoader_multilabel(paths_with_labels=labels_train, batch_size=batch_size,
                                           class_columns=['engagement', 'boredom'],
                                           num_classes=num_classes,
                                           horizontal_flip=0.5, vertical_flip=0,
                                           shift=0.5,
                                           brightness=0.5, shearing=0.5, zooming=0.5,
                                           random_cropping_out=0.5, rotation=0.5,
                                           scaling=None,
                                           channel_random_noise=0.5, bluring=0.5,
                                           worse_quality=0.5,
                                           mixup=0.5,
                                           prob_factors_for_each_class=(1, 0, 0, 0),
                                           pool_workers=10)

    # start to augment
    counter=0
    index=0
    while counter<num_extra_images:
        # get augmented images
        images,y = generator.__getitem__(index)
        # check if index is greater than length of provided by generator batches
        if index+1>=generator.__len__():
            index=0
        else:
            index+=1
        names=['image_%i'%(idx) for idx in range(counter, counter+batch_size, 1)]
        names=[os.path.join(output_path, names[i]) for i in range(len(names))]
        counter+=batch_size


