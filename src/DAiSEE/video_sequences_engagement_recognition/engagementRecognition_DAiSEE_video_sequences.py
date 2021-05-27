#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
TODO: write description of module
"""
import math
import shutil
import time
from functools import partial
from typing import Optional, Tuple, Dict, NamedTuple, Iterable, List, Union

import numpy as np
import pandas as pd
import os
import tensorflow as tf
from sklearn.metrics import recall_score, f1_score, accuracy_score
from sklearn.utils import class_weight

from tensorflow_utils.Layers import Non_local_block_multi_head, Self_attention_non_local_block
from tensorflow_utils.keras_datagenerators.ImageDataLoader import ImageDataLoader
from tensorflow_utils.keras_datagenerators.ImageDataLoader_multilabel import ImageDataLoader_multilabel
from tensorflow_utils.keras_datagenerators.ImageDataPreprocessor import ImageDataPreprocessor
from preprocessing.data_normalizing_utils import VGGFace2_normalization
from tensorflow_utils.Losses import weighted_categorical_crossentropy
from tensorflow_utils.callbacks import best_weights_setter_callback, get_annealing_LRreduce_callback, \
    validation_with_generator_callback_multilabel, get_reduceLRonPlateau_callback
from tensorflow_utils.keras_datagenerators.VideoSequenceLoader import VideoSequenceLoader
from tensorflow_utils.models.CNN_models import get_modified_VGGFace2_resnet_model, _get_pretrained_VGGFace2_model

"""from preprocessing.data_preprocessing.image_preprocessing_utils import load_image, save_image, resize_image
from preprocessing.face_recognition_utils import recognize_the_most_confident_person_retinaFace, \
    extract_face_according_bbox, load_and_prepare_detector_retinaFace"""

class Label(NamedTuple):
    # TODO: write description
    boredom: int
    engagement: int
    confusion:int
    frustration:int



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

def form_dataframe_of_relative_paths_to_data_with_multilabels(path_to_data:str, labels_dict:Dict[str,Label])-> pd.DataFrame:
    # TODO: write description
    directories_according_path=os.listdir(path_to_data)
    df_with_relative_paths_and_labels=pd.DataFrame(columns=['filename', 'engagement', 'boredom', 'confusion','frustration'])
    for dir in directories_according_path:
        if not dir in labels_dict.keys():
            continue
        img_filenames=os.listdir(os.path.join(path_to_data, dir))
        img_filenames=[os.path.join(dir, x) for x in img_filenames]
        label=[labels_dict[dir].engagement, labels_dict[dir].boredom, labels_dict[dir].confusion, labels_dict[dir].frustration]
        labels=[[label[0], label[1],label[2],label[3]] for _ in range(len(img_filenames))]
        tmp_df=pd.DataFrame(data=np.concatenate([np.array(img_filenames).reshape(-1,1), np.array(labels)], axis=-1),
                            columns=['filename', 'engagement', 'boredom', 'confusion','frustration'])
        df_with_relative_paths_and_labels=df_with_relative_paths_and_labels.append(tmp_df)
    return df_with_relative_paths_and_labels

def construct_model(input_shape:Tuple[int,...],path_to_VGGFace_weights:str, num_classes:int)->tf.keras.Model:
    # load VGGFace2 and take all model up to last conv layer
    pretrained_VGGFace2 = _get_pretrained_VGGFace2_model(path_to_VGGFace_weights, pretrained=True)
    # stack on top of it non-local block (multi-head local attention)
    x = pretrained_VGGFace2.get_layer('activation_48').output # output from last resnet block
    x = Non_local_block_multi_head(num_heads=4,  output_channels=1024,
                 head_output_channels=None,
                 downsize_factor=8,
                 shortcut_connection=True,
                 relative_position_encoding=True)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    # construct model from it
    VGGFace_with_local_attention=tf.keras.Model(inputs=pretrained_VGGFace2.inputs, outputs=[x])
    # freeze parts of model
    for i in range(141): # freeze up to 4th block
        VGGFace_with_local_attention.layers[i].trainable = False
    # construct sequence-based model from former model
    input_layer=tf.keras.layers.Input(input_shape)
    time_distributed_local_atten=tf.keras.layers.TimeDistributed(VGGFace_with_local_attention)(input_layer)
    x = Self_attention_non_local_block(output_channels=1024, downsize_factor = 1,
                 mode='spatio-temporal', name_prefix ="global_attention",
                 relative_position_encoding=False)(time_distributed_local_atten)
    # global average pooling
    x = tf.keras.layers.TimeDistributed(tf.keras.layers.GlobalAveragePooling2D())(x)
    # now LSTM layers, but it can be simple Dense
    x = tf.keras.layers.LSTM(128, return_sequences=True, dropout=0.3)(x)
    x = tf.keras.layers.LSTM(128, return_sequences=False, dropout=0.3)(x)
    x = tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.0001))(x)
    output = tf.keras.layers.Dense(num_classes, activation='softmax')(x)

    final_model=tf.keras.Model(inputs=[input_layer], outputs=[output])
    return final_model


def train_model(train_generator:Iterable[Tuple[np.ndarray, np.ndarray]], model:tf.keras.Model,
                optimizer:tf.keras.optimizers.Optimizer, loss:tf.keras.losses.Loss,
                epochs:int,
                val_generator:Iterable[Tuple[np.ndarray, np.ndarray]],
                metrics:List[tf.keras.metrics.Metric],
                callbacks:List[tf.keras.callbacks.Callback],
                path_to_save_results:str,
                class_weights:Optional[Dict[int,float]]=None,
                loss_weights:Optional[Dict['str', float]]=None)->tf.keras.Model:
    # create directory for saving results
    if not os.path.exists(path_to_save_results):
        os.makedirs(path_to_save_results)
    # compile model
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics, loss_weights=loss_weights)
    model.fit(train_generator, epochs=epochs, callbacks=callbacks, validation_data=val_generator, verbose=2,
              class_weight=class_weights)
    return model

def generate_paths_with_labels_from_directory(path_to_dir:str, class_name:str, class_value:int)->pd.DataFrame:
    # TODO: write description
    filenames=os.listdir(path_to_dir)
    filenames=[os.path.join(path_to_dir, filename) for filename in filenames]
    class_values=[class_value for _ in range(len(filenames))]
    df=pd.DataFrame(data=np.array([filenames, class_values]).T,
                    columns=['filename', class_name])
    df[class_name]=df[class_name].astype('float32')
    return df


if __name__ == '__main__':
    # params
    path_to_train_frames=r'C:\Databases\DAiSEE\train_preprocessed\extracted_faces'
    path_to_train_labels=r'C:\Databases\DAiSEE\Labels\TrainLabels.csv'
    path_to_dev_frames=r'C:\Databases\DAiSEE\dev_preprocessed\extracted_faces'
    path_to_dev_labels=r'C:\Databases\DAiSEE\Labels\ValidationLabels.csv'
    path_to_test_frames=r'C:\Databases\DAiSEE\test_preprocessed\extracted_faces'
    path_to_test_labels=r'C:\Databases\DAiSEE\Labels\TestLabels.csv'

    path_to_save_model_and_results= '../results'

    input_shape=(224,224,3)
    num_classes=4
    num_frames_in_seq=20
    batch_size=14
    epochs=50
    highest_lr=0.001
    lowest_lr = 0.00001
    momentum=0.9
    # create output path
    if not os.path.exists(path_to_save_model_and_results):
        os.makedirs(path_to_save_model_and_results)
    optimizer=tf.keras.optimizers.SGD(highest_lr, momentum=momentum, decay=1e-2/epochs)
    loss=tf.keras.losses.categorical_crossentropy
    # load labels
    dict_labels_train=load_labels_to_dict(path_to_train_labels)
    dict_labels_dev=load_labels_to_dict(path_to_dev_labels)
    dict_labels_test = load_labels_to_dict(path_to_test_labels)
    # form dataframes with relative paths and labels
    labels_train=form_dataframe_of_relative_paths_to_data_with_multilabels(path_to_train_frames, dict_labels_train)
    labels_dev=form_dataframe_of_relative_paths_to_data_with_multilabels(path_to_dev_frames, dict_labels_dev)
    labels_test=form_dataframe_of_relative_paths_to_data_with_multilabels(path_to_test_frames, dict_labels_test)
    # add full path to filename
    labels_train['filename']=path_to_train_frames+'\\'+labels_train['filename']
    labels_dev['filename'] = path_to_dev_frames +'\\'+ labels_dev['filename']
    labels_test['filename'] = path_to_test_frames + '\\' + labels_test['filename']
    # convert labels into float32 type
    labels_train['engagement']=labels_train['engagement'].astype('float32')
    labels_train['boredom'] = labels_train['boredom'].astype('float32')
    labels_dev['engagement'] = labels_dev['engagement'].astype('float32')
    labels_dev['boredom'] = labels_dev['boredom'].astype('float32')
    labels_test['engagement'] = labels_test['engagement'].astype('float32')
    labels_test['boredom'] = labels_test['boredom'].astype('float32')
    labels_train['confusion']=labels_train['confusion'].astype('float32')
    labels_train['frustration'] = labels_train['frustration'].astype('float32')
    labels_dev['confusion'] = labels_dev['confusion'].astype('float32')
    labels_dev['frustration'] = labels_dev['frustration'].astype('float32')
    labels_test['confusion'] = labels_test['confusion'].astype('float32')
    labels_test['frustration'] = labels_test['frustration'].astype('float32')

    #labels_train=labels_train.iloc[:6400]
    #labels_dev = labels_dev.iloc[:640]

    # change labels_train to fit for generator VideoSequenceLoader
    labels_train[['filename', 'frame_num']] = labels_train['filename'].str.rsplit('_',1, expand=True)
    labels_train['frame_num']=labels_train['frame_num'].apply(lambda x: x.split('.')[0])
    labels_train = labels_train.drop(columns=['boredom', 'confusion', 'frustration'])
    labels_train=labels_train[['filename', 'frame_num', 'engagement']]
    labels_train.columns=['filename', 'frame_num', 'class']
    labels_train['frame_num']=labels_train['frame_num'].astype('int32')

    # change labels_dev to fit for generator VideoSequenceLoader
    labels_dev[['filename', 'frame_num']] = labels_dev['filename'].str.rsplit('_',1, expand=True)
    labels_dev['frame_num']=labels_dev['frame_num'].apply(lambda x: x.split('.')[0])
    labels_dev = labels_dev.drop(columns=['boredom', 'confusion', 'frustration'])
    labels_dev=labels_dev[['filename', 'frame_num', 'engagement']]
    labels_dev.columns=['filename', 'frame_num', 'class']
    labels_dev['frame_num']=labels_dev['frame_num'].astype('int32')

    # change labels_test to fit for generator VideoSequenceLoader
    labels_test[['filename', 'frame_num']] = labels_test['filename'].str.rsplit('_',1, expand=True)
    labels_test['frame_num']=labels_test['frame_num'].apply(lambda x: x.split('.')[0])
    labels_test = labels_test.drop(columns=['boredom', 'confusion', 'frustration'])
    labels_test=labels_test[['filename', 'frame_num', 'engagement']]
    labels_test.columns=['filename', 'frame_num', 'class']
    labels_test['frame_num']=labels_test['frame_num'].astype('int32')

    # create sequence generators
    train_gen=VideoSequenceLoader(paths_with_labels=labels_train, batch_size=batch_size,
                                  num_frames_in_seq=num_frames_in_seq, proportion_of_intersection=0.5,
                              preprocess_function=VGGFace2_normalization,
                              num_classes=num_classes,
                 horizontal_flip= 0.1, vertical_flip= 0,
                 shift= 0.1,
                 brightness= 0.1, shearing= 0.1, zooming= 0.1,
                 random_cropping_out = 0.1, rotation = 0.1,
                 scaling= None,
                 channel_random_noise= 0.1, bluring= 0.1,
                 worse_quality= 0.1,
                 num_pool_workers=8)


    dev_gen=VideoSequenceLoader(paths_with_labels=labels_dev, batch_size=batch_size,
                                  num_frames_in_seq=num_frames_in_seq, proportion_of_intersection=0.5,
                              preprocess_function=VGGFace2_normalization,
                              num_classes=num_classes,
                 horizontal_flip= None, vertical_flip= None,
                 shift= None,
                 brightness= None, shearing= None, zooming= None,
                 random_cropping_out = None, rotation = None,
                 scaling= None,
                 channel_random_noise= None, bluring= None,
                 worse_quality= None,
                 num_pool_workers=4)

    test_gen=VideoSequenceLoader(paths_with_labels=labels_test, batch_size=batch_size,
                                  num_frames_in_seq=num_frames_in_seq, proportion_of_intersection=0.5,
                              preprocess_function=VGGFace2_normalization,
                              num_classes=num_classes,
                 horizontal_flip= None, vertical_flip= None,
                 shift= None,
                 brightness= None, shearing= None, zooming= None,
                 random_cropping_out = None, rotation = None,
                 scaling= None,
                 channel_random_noise= None, bluring= None,
                 worse_quality= None,
                 num_pool_workers=4)

    # construct model
    model=construct_model(input_shape=(20,224,224,3),
                          path_to_VGGFace_weights=r'D:\PycharmProjects\Denis\vggface2_Keras\vggface2_Keras\model\resnet50_softmax_dim512\weights.h5',
                          num_classes=num_classes)
    model.summary()
    tf.keras.utils.plot_model(model, os.path.join(path_to_save_model_and_results, 'model_graph.png'), show_shapes=True)

    # create logger
    logger = open(os.path.join(path_to_save_model_and_results, 'val_logs.txt'), mode='w')
    logger.close()
    logger = open(os.path.join(path_to_save_model_and_results, 'val_logs.txt'), mode='a')
    # write training params and all important information:
    logger.write('# Train params:\n')
    logger.write('Database:%s\n'%"DAiSEE")
    logger.write('Epochs:%i\n'%epochs)
    logger.write('Highest_lr:%f\n'%highest_lr)
    logger.write('Lowest_lr:%f\n'%lowest_lr)
    logger.write('Optimizer:%s\n'%optimizer)
    logger.write('Loss:%s\n'% loss)
    logger.write('num_frames_in_seq:%i\n'%num_frames_in_seq)
    logger.write('Additional info:%s\n'%
                 'Local with global convolutional attention. Local attention consists of multi head non-local blocks, '
                  'Global attention is non-local block with mode spatio-temporal (it makes attention across timeline as well)')
    # create metrics
    metrics=[tf.keras.metrics.CategoricalAccuracy()]

    # create callbacks
    callbacks = [validation_with_generator_callback_multilabel(test_gen, metrics=(partial(f1_score, average='macro'),
                                                                                 accuracy_score,
                                                                                 partial(recall_score,
                                                                                         average='macro')),
                                                               num_label_types=1,
                                                               num_metric_to_set_weights=1,
                                                               logger=logger),
                 get_reduceLRonPlateau_callback(monitoring_loss = 'val_loss', reduce_factor = 0.2,
                                   num_patient_epochs= 2,
                                   min_lr = lowest_lr)]


    # train model
    model=train_model(train_generator=train_gen, model=model,
    optimizer=optimizer, loss=loss,
    epochs=epochs,
    val_generator=dev_gen,
    metrics=metrics,
    callbacks=callbacks,
    path_to_save_results=path_to_save_model_and_results,
    class_weights= None,
    loss_weights= None)

    model.save_weights(os.path.join(path_to_save_model_and_results, "model_weights.h5"))




