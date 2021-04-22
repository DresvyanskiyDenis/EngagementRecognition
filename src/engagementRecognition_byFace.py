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
from sklearn.metrics import recall_score, f1_score
from sklearn.utils import class_weight

from keras_datagenerators import ImageDataLoader
from tensorflow_utils.callbacks import best_weights_setter_callback, get_annealing_LRreduce_callback

"""from preprocessing.data_preprocessing.image_preprocessing_utils import load_image, save_image, resize_image
from preprocessing.face_recognition_utils import recognize_the_most_confident_person_retinaFace, \
    extract_face_according_bbox, load_and_prepare_detector_retinaFace"""

class Label(NamedTuple):
    # TODO: write description
    boredom: int
    engagement: int
    confusion:int
    frustration:int



"""def extract_faces_from_dir(path_to_dir:str, output_dir:str,detector:object, resize:Optional[Tuple[int,int]])->None:
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
        counter+=1"""

def sort_images_according_their_class(path_to_images:str, output_path:str, path_to_labels:str):
    dict_labels=load_labels_to_dict(path_to_labels)
    dirs_with_images=os.listdir(path_to_images)
    # check if output path is existed
    if not os.path.exists(output_path):
        os.makedirs(output_path, exist_ok=True)
    num_classes=np.unique(np.array(list(x.engagement for x in dict_labels.values()))).shape[0]
    # create subdirectories for classes
    for num_class in range(num_classes):
        if not os.path.exists(os.path.join(output_path, str(num_class))):
            os.makedirs(os.path.join(output_path, str(num_class)), exist_ok=True)
    # copy images according their class
    for dir_with_images in dirs_with_images:
        if not dir_with_images in dict_labels.keys():
            continue
        class_num=dict_labels[dir_with_images].engagement
        image_filenames=os.listdir(os.path.join(path_to_images, dir_with_images))
        for image_filename in image_filenames:
            shutil.copy(os.path.join(path_to_images, dir_with_images, image_filename),
                    os.path.join(output_path, str(class_num), image_filename))





def load_labels_to_dict(path:str)->Dict[str, Label]:
    # TODO:write description
    labels_df=pd.read_csv(path)
    labels_df['ClipID']=labels_df['ClipID'].apply(lambda x: x.split('.')[0])
    #labels_df.columns=[labels_df.columns[0]]+[x.lower() for x in labels_df.columns[1:]]
    labels_dict=dict(
        (row[1].iloc[0],
         Label(*row[1].iloc[1:].values))
        for row in labels_df.iterrows()
    )
    return labels_dict




def form_dataframe_of_relative_paths_to_data_with_labels(path_to_data:str, labels_dict:Dict[str,Label])-> pd.DataFrame:
    # TODO: write description
    directories_according_path=os.listdir(path_to_data)
    df_with_relative_paths_and_labels=pd.DataFrame(columns=['filename','class'])
    for dir in directories_according_path:
        if not dir in labels_dict.keys():
            continue
        img_filenames=os.listdir(os.path.join(path_to_data, dir))
        img_filenames=[os.path.join(dir, x) for x in img_filenames]
        label=labels_dict[dir].engagement
        labels=[label for _ in range(len(img_filenames))]
        tmp_df=pd.DataFrame(data=np.array([img_filenames, labels]).T, columns=['filename', 'class'])
        df_with_relative_paths_and_labels=df_with_relative_paths_and_labels.append(tmp_df)
    return df_with_relative_paths_and_labels


def tmp_model(input_shape)->tf.keras.Model:
    model_tmp=tf.keras.applications.mobilenet_v2.MobileNetV2(input_shape=input_shape,include_top=False,
                                                             weights='imagenet', pooling='avg')
    x=tf.keras.layers.Dense(512, activation="relu")(model_tmp.output)
    x = tf.keras.layers.Dropout(0.4)(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.keras.layers.Dense(4, activation="softmax")(x)
    result_model=tf.keras.Model(inputs=model_tmp.inputs, outputs=[x])
    return result_model


def train_model(train_generator:Iterable[Tuple[np.ndarray, np.ndarray]], model:tf.keras.Model,
                optimizer:tf.keras.optimizers.Optimizer, loss:tf.keras.losses.Loss,
                epochs:int,
                val_generator:Iterable[Tuple[np.ndarray, np.ndarray]],
                metrics:List[tf.keras.metrics.Metric],
                callbacks:List[tf.keras.callbacks.Callback],
                path_to_save_results:str,
                class_weights:Optional[Dict[int,float]]=None)->tf.keras.Model:
    # create directory for saving results
    if not os.path.exists(path_to_save_results):
        os.makedirs(path_to_save_results)
    # compile model
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    model.fit(train_generator, epochs=epochs, callbacks=callbacks, validation_data=val_generator, verbose=2,
              class_weight=class_weights)
    return model


if __name__ == '__main__':
    '''path_to_directory_with_frames=r'D:\Databases\DAiSEE\frames'
    path_to_output_directory=r'D:\Databases\DAiSEE\extracted_faces'
    resize=(224,224)
    extract_faces_from_all_subdirectories_in_directory(path_to_directory_with_frames, path_to_output_directory, resize)'''
    # params
    path_to_train_frames=r'C:\Databases\DAiSEE\train_preprocessed\extracted_faces'
    path_to_train_labels=r'C:\Databases\DAiSEE\Labels\TrainLabels.csv'
    path_to_dev_frames=r'C:\Databases\DAiSEE\dev_preprocessed\extracted_faces'
    path_to_dev_labels=r'C:\Databases\DAiSEE\Labels\ValidationLabels.csv'
    '''output_path=r'D:\Databases\DAiSEE\dev_preprocessed\sorted_faces'
    sort_images_according_their_class(path_to_images=path_to_dev_frames, output_path=output_path,
                                      path_to_labels=path_to_dev_labels)'''
    input_shape=(224,224,3)
    num_classes=4
    batch_size=72
    epochs=30
    highest_lr=0.001
    lowest_lr = 0.00001
    momentum=0.9
    optimizer=tf.keras.optimizers.SGD(highest_lr, momentum=momentum)
    loss=tf.keras.losses.categorical_crossentropy
    # load labels
    dict_labels_train=load_labels_to_dict(path_to_train_labels)
    dict_labels_dev=load_labels_to_dict(path_to_dev_labels)
    # form dataframes with relative paths and labels
    labels_train=form_dataframe_of_relative_paths_to_data_with_labels(path_to_train_frames, dict_labels_train)
    labels_dev=form_dataframe_of_relative_paths_to_data_with_labels(path_to_dev_frames, dict_labels_dev)
    # add full path to them
    labels_train['filename']=path_to_train_frames+'\\'+labels_train['filename']
    labels_dev['filename'] = path_to_dev_frames +'\\'+ labels_dev['filename']
    labels_train['class']=labels_train['class'].astype('float32')
    labels_dev['class'] = labels_dev['class'].astype('float32')
    class_weights=class_weight.compute_class_weight(class_weight='balanced',classes=np.unique(labels_train['class']), y=labels_train['class'].values.reshape((-1,)))
    class_weights=dict((i,class_weights[i]) for i in range(len(class_weights)))
    #labels_train=labels_train.iloc[:640]
    #labels_dev = labels_dev.iloc[:640]
    # create generators
    train_gen=ImageDataLoader(paths_with_labels=labels_train, batch_size=batch_size,
                              preprocess_function=tf.keras.applications.mobilenet_v2.preprocess_input,
                              num_classes=num_classes,
                 horizontal_flip= 0.1, vertical_flip= None,
                 shift= 0.1,
                 brightness= 0.1, shearing= 0.1, zooming= 0.1,
                 random_cropping_out = 0.1, rotation = 0.1,
                 scaling= None,
                 channel_random_noise= 0.1, bluring= 0.1,
                 worse_quality= 0.1,
                 mixup = None,
                 pool_workers=10)

    dev_gen=ImageDataLoader(paths_with_labels=labels_dev, batch_size=batch_size,
                            preprocess_function=tf.keras.applications.mobilenet_v2.preprocess_input,
                            num_classes=num_classes,
                 horizontal_flip= None, vertical_flip= None,
                 shift= None,
                 brightness= None, shearing= None, zooming= None,
                 random_cropping_out = None, rotation = None,
                 scaling= None,
                 channel_random_noise= None, bluring= None,
                 worse_quality= None,
                 mixup = None,
                 pool_workers=10)
    # create model
    model=tmp_model(input_shape)
    # create callbacks
    callbacks=[best_weights_setter_callback(dev_gen, partial(f1_score, average='macro')),
               get_annealing_LRreduce_callback(highest_lr, lowest_lr, 5)]
    # create metrics
    metrics=[tf.keras.metrics.Recall()]
    model=train_model(train_gen, model, optimizer, loss, epochs,
                      None, metrics, callbacks, path_to_save_results='results',
                      class_weights=class_weights)
    model.save("results\\model.h5")
    model.save_weights("results\\model_weights.h5")



