#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
TODO: write description of module
"""
import gc
import math
import shutil
import time
from collections import Counter
from functools import partial
from typing import Optional, Tuple, Dict, NamedTuple, Iterable, List

import numpy as np
import pandas as pd
import os
import tensorflow as tf
from sklearn.metrics import recall_score, f1_score, accuracy_score
from sklearn.utils import class_weight

from augmentation.oversampling import oversample_by_border_SMOTE
from tensorflow_utils.keras_datagenerators import ImageDataLoader
from tensorflow_utils.keras_datagenerators.ImageDataLoader_multilabel import ImageDataLoader_multilabel
from preprocessing.data_normalizing_utils import VGGFace2_normalization
from preprocessing.data_preprocessing.image_preprocessing_utils import save_image, load_image
from src.SPECOM2021.engagementRecognition_DAiSEE import load_labels_to_dict, \
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

def generate_minority_samples_by_SMOTE(paths_with_labels:pd.DataFrame, sampling_strategy)->Tuple[np.ndarray, np.ndarray]:
    # load all images
    image_shape=load_image(paths_with_labels['filename'].iloc[0]).shape
    images=np.zeros((paths_with_labels.shape[0],)+image_shape, dtype='uint8')
    for idx_df in range(paths_with_labels.shape[0]):
        images[idx_df]=load_image(paths_with_labels.filename.iloc[idx_df])
    images=images.reshape((images.shape[0],-1))
    labels=paths_with_labels['class'].values.reshape((-1,))
    images, labels = oversample_by_border_SMOTE(images, labels, sampling_strategy)
    images=images.reshape((-1,)+image_shape)
    return images, labels

def generate_several_times_data_by_SMOTE(paths_with_labels:pd.DataFrame, num_times:int,
                                         num_classes_every_time:Dict[int, int], output_path:str,
                                         num_classes_to_save:Tuple[int,...])->None:
    # create counters of generatred images
    counter= Counter()
    # check if dir exists
    if not os.path.exists(output_path):
        os.makedirs(output_path, exist_ok=True)
    # iterate through num_times
    for time_idx in range(num_times):
        # sample classes from df
        df_for_further_oversampling=pd.DataFrame(columns=paths_with_labels.columns)
        for class_num, values_in_class in num_classes_every_time.items():
            sampled_values=paths_with_labels[paths_with_labels['class']==class_num].sample(n=values_in_class)
            df_for_further_oversampling=df_for_further_oversampling.append(sampled_values)
        df_for_further_oversampling=df_for_further_oversampling.sample(frac=1)
        images, labels = generate_minority_samples_by_SMOTE(df_for_further_oversampling, sampling_strategy='not majority')
        # save generated minority classes
        for num_class_to_save in num_classes_to_save:
            # create dir with such class
            os.makedirs(os.path.join(output_path, str(num_class_to_save)), exist_ok=True)
            # find out indexes of particular class
            indices=labels.reshape((-1,))==num_class_to_save
            indices=np.where(indices)[0]
            # save images
            for idx_to_save in indices:
                save_image(img=images[idx_to_save], path_to_output=os.path.join(output_path, str(num_class_to_save), '%i.png'%counter[num_class_to_save]))
                counter[num_class_to_save]+=1
        del images
        del labels
        del df_for_further_oversampling
        gc.collect()




if __name__=='__main__':
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


    # augment images by SMOTE
    # Make major class less presented
    labels_train=labels_train.drop(columns=['boredom', 'frustration', 'confusion'])
    labels_train.columns=['filename','class']
    '''labels_train = pd.concat([labels_train[labels_train['class'] == 0],
                              labels_train[labels_train['class'] == 1],
                              labels_train[labels_train['class'] == 2].iloc[::10],
                              labels_train[labels_train['class'] == 3].iloc[::10]
                              ])'''
    generate_several_times_data_by_SMOTE(labels_train, num_times=10,
                                        num_classes_every_time={0:500, 1:700, 2:4000, 3:4000},
                                        output_path=r'C:\Databases\DAiSEE\train_preprocessed\SMOTE_images',
                                        num_classes_to_save=(0, 1))




    # augment images by generator and affine transformations
    '''# only labels with selected class remain
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
        counter+=batch_size'''
    a=1+2
