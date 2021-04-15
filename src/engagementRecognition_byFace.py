#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
TODO: write description of module
"""
import math
import time
from typing import Optional, Tuple, Dict, NamedTuple

import numpy as np
import pandas as pd
import cv2
import os
import tensorflow as tf

from preprocessing.data_preprocessing.image_preprocessing_utils import load_image, save_image, resize_image
from preprocessing.face_recognition_utils import recognize_the_most_confident_person_retinaFace, \
    extract_face_according_bbox, load_and_prepare_detector_retinaFace

class Label(NamedTuple):
    # TODO: write description
    boredom: int
    engagement: int
    confusion:int
    frustration:int



def extract_faces_from_dir(path_to_dir:str, output_dir:str,detector:object, resize:Optional[Tuple[int,int]])->None:
    # TODO: add description
    # get filenames of all images in dir
    image_filenames=os.listdir(path_to_dir)
    # create, if necessary, output dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    # extract faces from each image
    for image_filename in image_filenames:
        # load image by PIL.Image
        img=load_image(os.path.join(path_to_dir,image_filename))
        # calculate bounding boxes for the most confident person via provided detector
        bbox=recognize_the_most_confident_person_retinaFace(img, detector)
        bbox=tuple(max(0, _) for _ in bbox)
        if bbox is None or len(bbox)==0:
            continue
        # extract face from image according bbox
        face_img=extract_face_according_bbox(img, bbox) # it is in RGB format now
        # resize if needed
        if not resize is None:
            face_img=resize_image(face_img, resize)
        # save
        save_image(face_img, path_to_output=os.path.join(output_dir, image_filename))


def extract_faces_from_all_subdirectories_in_directory(path_to_dir:str, path_to_output:str,
                                                       resize:Optional[Tuple[int,int]])-> None:
    # TODO: add description
    # get subdirectories from provided dir
    subdirectories=os.listdir(path_to_dir)
    # load retinaFace face detector
    detector=load_and_prepare_detector_retinaFace('retinaface_mnet025_v2')
    # create output path, if it is not existed
    if not os.path.exists(path_to_output):
        os.makedirs(path_to_output, exist_ok=True)
    # go over all subdirectories
    counter=0
    overall=len(subdirectories)
    for subdirectory in subdirectories:
        # inside every subdirectory the subdirectory_frames subsubdirectory is localed. There are frames extracted
        # from video and we need to extract faces from every image
        # construct subdirectory_with_frames variable, which is name of subsubdirectory
        subdirectory_with_frames=subdirectory+'_frames'
        # create name of directory and the directory itself, in which the faces of images will be extracted
        output_dir_name=os.path.join(path_to_output, subdirectory)
        if not os.path.exists(output_dir_name):
            os.makedirs(output_dir_name, exist_ok=True)
        else:
            counter += 1
            continue
        # extract faces from defined subdirectory
        start=time.time()
        extract_faces_from_dir(path_to_dir=os.path.join(path_to_dir, subdirectory, subdirectory_with_frames),
                               output_dir=output_dir_name, detector=detector, resize=resize)
        print('processed:%i, overall:%i, time:%f'%(counter, overall, time.time()-start))
        counter+=1

def load_labels_to_dict(path:str)->Dict[str, Label]:
    # TODO:write description
    labels_df=pd.read_csv(path)
    labels_df['ClipID']=labels_df['ClipID'].apply(lambda x: x.split('.')[0])
    labels_dict=labels_df.set_index('ClipID').T.to_dict(Label)
    return labels_dict




def form_dataframe_of_relative_paths_to_data_with_labels(path_to_data:str, labels_dict:Dict[str,Label])-> pd.DataFrame:
    # TODO: write description
    directories_according_path=os.listdir(path_to_data)
    df_with_relative_paths_and_labels=pd.DataFrame(columns=['filename','class'])
    for dir in directories_according_path:
        img_filenames=os.listdir(os.path.join(path_to_data, dir))
        label=labels_dict[dir].engagement
        labels=[label for _ in range(len(img_filenames))]
        tmp_df=pd.DataFrame(data=np.array([img_filenames, labels]), columns=['filename', 'class'])
        df_with_relative_paths_and_labels.append(tmp_df)
    return df_with_relative_paths_and_labels


def tmp_model()->tf.keras.Model:
    model_tmp=tf.keras.applications.mobilenet_v2.MobileNetV2(input_shape=(224,224,3),include_top=False,
                                                             weights='imagenet', pooling='avg')
    x=tf.keras.layers.Dense(512, activation="relu")(model_tmp.output)
    x = tf.keras.layers.Dropout(0.4)(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.keras.layers.Dense(4, activation="softmax")(x)
    result_model=tf.keras.Model(inputs=model_tmp.inputs, outputs=[x])
    return result_model







if __name__ == '__main__':
    '''path_to_directory_with_frames=r'D:\Databases\DAiSEE\frames'
    path_to_output_directory=r'D:\Databases\DAiSEE\extracted_faces'
    resize=(224,224)
    extract_faces_from_all_subdirectories_in_directory(path_to_directory_with_frames, path_to_output_directory, resize)'''
    # params
    path_to_train_frames=r'D:\Databases\DAiSEE\train_preprocessed\frames'
    path_to_train_labels=r'D:\Databases\DAiSEE\Labels\TrainLabels.csv'
    path_to_dev_frames=r'D:\Databases\DAiSEE\dev_preprocessed\frames'
    path_to_dev_labels=r'D:\Databases\DAiSEE\Labels\ValidationLabels.csv'
    input_shape=(224,224,3)
    batch_size=32
    epochs=10
    lr=0.005
    optimizer=tf.keras.optimizers.Adam(lr)
    loss=tf.keras.losses.categorical_crossentropy
    # load labels
    dict_labels_train=load_labels_to_dict(path_to_train_labels)
    dict_labels_dev=load_labels_to_dict(path_to_dev_labels)
    # form dataframes with relative paths and labels
    labels_train=form_dataframe_of_relative_paths_to_data_with_labels(path_to_train_frames, dict_labels_train)
    labels_dev=form_dataframe_of_relative_paths_to_data_with_labels(path_to_dev_frames, dict_labels_dev)
    # create train generator
    train_data_generator=tf.keras.preprocessing.image.ImageDataGenerator(
    rotation_range=45, width_shift_range=0.4,
    height_shift_range=0.3, brightness_range=[-0.2, 0.2], shear_range=0.4, zoom_range=0.3,
    channel_shift_range=0.1,
    horizontal_flip=True, rescale=1./255).flow_from_dataframe(labels_train, directory=path_to_train_frames,
                                                              target_size=(224,224), batch_size=batch_size)
    # create dev generator
    dev_data_generator=tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255).flow_from_dataframe(labels_dev, directory=path_to_dev_frames,
                                                              target_size=(224,224), batch_size=batch_size)
    # create model
    model=tmp_model()
    model.compile(optimizer=optimizer, loss=loss)

    model.fit(train_data_generator, batch_size=batch_size, epochs=epochs, validation_data=dev_data_generator)


