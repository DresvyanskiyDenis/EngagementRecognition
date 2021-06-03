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

from preprocessing.class_weights import get_class_weights_Effective_Number_of_Samples
from tensorflow_utils.Layers import Non_local_block_multi_head
from tensorflow_utils.keras_datagenerators.ImageDataLoader import ImageDataLoader
from tensorflow_utils.keras_datagenerators.ImageDataLoader_multilabel import ImageDataLoader_multilabel
from tensorflow_utils.keras_datagenerators.ImageDataPreprocessor import ImageDataPreprocessor
from preprocessing.data_normalizing_utils import VGGFace2_normalization
from tensorflow_utils.Losses import weighted_categorical_crossentropy, categorical_focal_loss
from tensorflow_utils.callbacks import best_weights_setter_callback, get_annealing_LRreduce_callback, validation_with_generator_callback_multilabel
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
        if len(img_filenames)==0:
            continue
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


def get_model_with_local_att(dense_neurons_after_conv: Tuple[int,...],
                                       dropout: float = 0.3,
                                       regularization:Optional[tf.keras.regularizers.Regularizer]=None,
                                       output_neurons: Union[Tuple[int,...], int] = 7, pooling_at_the_end: Optional[str] = None,
                                       pretrained: bool = True,
                                       path_to_weights: Optional[str] = None,
                                       multi_head_attention:bool=True) -> tf.keras.Model:
    pretrained_VGGFace2 = _get_pretrained_VGGFace2_model(path_to_weights, pretrained=pretrained)
    x=pretrained_VGGFace2.get_layer('activation_48').output
    if multi_head_attention:
        x = Non_local_block_multi_head(num_heads=4,  output_channels=1024,
                 head_output_channels=None,
                 downsize_factor=8,
                 shortcut_connection=True)(x)
        x = tf.keras.layers.BatchNormalization()(x)
    # take pooling or not
    if pooling_at_the_end is not None:
        if pooling_at_the_end=='avg':
            x=tf.keras.layers.GlobalAveragePooling2D()(x)
        elif pooling_at_the_end=='max':
            x=tf.keras.layers.GlobalMaxPooling2D()(x)
        else:
            raise AttributeError('Parameter pooling_at_the_end can be either \'avg\' or \'max\'. Got %s.'%(pooling_at_the_end))
    # create Dense layers
    for dense_layer_idx in range(len(dense_neurons_after_conv)-1):
        num_neurons_on_layer=dense_neurons_after_conv[dense_layer_idx]
        x = tf.keras.layers.Dense(num_neurons_on_layer, activation='relu', kernel_regularizer=regularization)(x)
        if dropout:
            x = tf.keras.layers.Dropout(dropout)(x)
    # pre-last Dense layer
    num_neurons_on_layer=dense_neurons_after_conv[-1]
    x = tf.keras.layers.Dense(num_neurons_on_layer, activation='relu')(x)
    # If outputs should be several, then create several layers, otherwise one
    if isinstance(output_neurons, tuple):
        output_layers=[]
        for num_output_neurons in output_neurons:
            if dropout:
                output_layer_i = tf.keras.layers.Dropout(dropout)(x)
            output_layer_i = tf.keras.layers.Dense(128, activation='relu')(output_layer_i)
            output_layer_i=tf.keras.layers.Dense(num_output_neurons, activation='softmax')(output_layer_i)
            #output_layer_i=tf.keras.layers.Reshape((-1, 1))(output_layer_i)
            output_layers.append(output_layer_i)
    else:
        output_layers = tf.keras.layers.Dense(output_neurons, activation='softmax')(x)
        # in tf.keras.Model it should be always a list (even when it has only 1 element)
        output_layers = [output_layers]
    # create model
    model=tf.keras.Model(inputs=pretrained_VGGFace2.inputs, outputs=output_layers)
    del pretrained_VGGFace2
    return model


def train_model(train_generator:Iterable[Tuple[np.ndarray, np.ndarray]], model:tf.keras.Model,
                optimizer:tf.keras.optimizers.Optimizer, loss:tf.keras.losses.Loss,
                epochs:int,
                val_generator:Iterable[Tuple[np.ndarray, np.ndarray]],
                metrics:List[tf.keras.metrics.Metric],
                callbacks:List[tf.keras.callbacks.Callback],
                path_to_save_results:str,
                class_weights:Optional[Dict[int,float]]=None,
                loss_weights:Optional[Dict[str, float]]=None)->tf.keras.Model:
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
    batch_size=64
    epochs=30
    highest_lr=0.0005
    lowest_lr = 0.00001
    momentum=0.9
    weighting_beta=0.99
    focal_loss_gamma=2
    output_path='results'
    # create output path
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    optimizer=tf.keras.optimizers.SGD(highest_lr, momentum=momentum, clipnorm=1., decay=1e-6)
    # load labels
    dict_labels_train = load_labels_to_dict(path_to_train_labels)
    dict_labels_dev = load_labels_to_dict(path_to_dev_labels)
    dict_labels_test = load_labels_to_dict(path_to_test_labels)
    # form dataframes with relative paths and labels
    labels_train = form_dataframe_of_relative_paths_to_data_with_multilabels(path_to_train_frames, dict_labels_train)
    labels_dev = form_dataframe_of_relative_paths_to_data_with_multilabels(path_to_dev_frames, dict_labels_dev)
    labels_test = form_dataframe_of_relative_paths_to_data_with_multilabels(path_to_test_frames, dict_labels_test)
    # add full path to filename
    labels_train['filename'] = path_to_train_frames + '\\' + labels_train['filename']
    labels_dev['filename'] = path_to_dev_frames + '\\' + labels_dev['filename']
    labels_test['filename'] = path_to_test_frames + '\\' + labels_test['filename']
    # convert labels into float32 type
    labels_train['engagement'] = labels_train['engagement'].astype('float32')
    labels_train['boredom'] = labels_train['boredom'].astype('float32')
    labels_dev['engagement'] = labels_dev['engagement'].astype('float32')
    labels_dev['boredom'] = labels_dev['boredom'].astype('float32')
    labels_test['engagement'] = labels_test['engagement'].astype('float32')
    labels_test['boredom'] = labels_test['boredom'].astype('float32')
    labels_train['confusion'] = labels_train['confusion'].astype('float32')
    labels_train['frustration'] = labels_train['frustration'].astype('float32')
    labels_dev['confusion'] = labels_dev['confusion'].astype('float32')
    labels_dev['frustration'] = labels_dev['frustration'].astype('float32')
    labels_test['confusion'] = labels_test['confusion'].astype('float32')
    labels_test['frustration'] = labels_test['frustration'].astype('float32')
    # add augmentation data and delete all non-engagement class values
    labels_train=labels_train.drop(columns=['boredom', 'confusion','frustration'])
    labels_dev = labels_dev.drop(columns=['boredom', 'confusion', 'frustration'])

    # add augmented images
    class_0_augmented=generate_paths_with_labels_from_directory(path_to_dir=r'C:\Databases\DAiSEE\train_preprocessed\augmentation\engagement\0',
                                                                class_name='engagement',
                                                                class_value=0)
    class_1_augmented = generate_paths_with_labels_from_directory(
        path_to_dir=r'C:\Databases\DAiSEE\train_preprocessed\augmentation\engagement\1',
        class_name='engagement',
        class_value=1)
    labels_train=pd.concat([labels_train, class_0_augmented, class_1_augmented], axis=0)
    # add SMOTE images
    class_0_SMOTE=generate_paths_with_labels_from_directory(path_to_dir=r'C:\Databases\DAiSEE\train_preprocessed\SMOTE_images\0',
                                                                class_name='engagement',
                                                                class_value=0)
    class_1_SMOTE = generate_paths_with_labels_from_directory(
        path_to_dir=r'C:\Databases\DAiSEE\train_preprocessed\SMOTE_images\1',
        class_name='engagement',
        class_value=1)
    labels_train = pd.concat([labels_train, class_0_SMOTE, class_1_SMOTE], axis=0)
    """# turn 4-class task into 2-class task
    labels_train.loc[(labels_train['engagement'] == 1),'engagement'] = 0
    labels_train.loc[(labels_train['engagement'] == 2),'engagement'] = 1
    labels_train.loc[(labels_train['engagement'] == 3),'engagement'] = 1

    labels_dev.loc[(labels_dev['engagement'] == 1),'engagement'] = 0
    labels_dev.loc[(labels_dev['engagement'] == 2),'engagement'] = 1
    labels_dev.loc[(labels_dev['engagement'] == 3),'engagement'] = 1
    num_classes = 2"""

    # class weights
    class_weights=get_class_weights_Effective_Number_of_Samples(labels=np.array(labels_train['engagement']).reshape((-1,)),
                                                                beta=weighting_beta)

    '''# undersampling
    labels_train=pd.concat([labels_train[labels_train['class']==0],
                            labels_train[labels_train['class'] == 1],
                            labels_train[labels_train['class'] == 2].iloc[::5],
                            labels_train[labels_train['class'] == 3].iloc[::5]
                            ])'''
    #labels_train=labels_train.iloc[:6400]
    #labels_dev = labels_dev.iloc[:640]
    # if we use ImageDataLoader, not multilabel
    #labels_train.columns=['filename', 'class']
    #labels_dev.columns = ['filename', 'class']

    # create generators
    train_gen=ImageDataLoader_multilabel(paths_with_labels=labels_train, batch_size=batch_size,
                                         class_columns=['engagement'],
                              preprocess_function=VGGFace2_normalization,
                              num_classes=num_classes,
                 horizontal_flip= 0.1, vertical_flip= 0,
                 shift= 0.1,
                 brightness= 0.1, shearing= 0.1, zooming= 0.1,
                 random_cropping_out = 0.1, rotation = 0.1,
                 scaling= None,
                 channel_random_noise= 0.1, bluring= 0.1,
                 worse_quality= 0.1,
                 mixup = 0.4,
                 pool_workers=12)

    dev_gen=ImageDataLoader_multilabel(paths_with_labels=labels_dev, batch_size=batch_size,
                                       class_columns=['engagement'],
                            preprocess_function=VGGFace2_normalization,
                            num_classes=num_classes,
                 horizontal_flip= None, vertical_flip= None,
                 shift= None,
                 brightness= None, shearing= None, zooming= None,
                 random_cropping_out = None, rotation = None,
                 scaling= None,
                 channel_random_noise= None, bluring= None,
                 worse_quality= None,
                 mixup = None,
                 pool_workers=4)

    test_gen=ImageDataLoader_multilabel(paths_with_labels=labels_test, batch_size=batch_size,
                                       class_columns=['engagement'],
                            preprocess_function=VGGFace2_normalization,
                            num_classes=num_classes,
                 horizontal_flip= None, vertical_flip= None,
                 shift= None,
                 brightness= None, shearing= None, zooming= None,
                 random_cropping_out = None, rotation = None,
                 scaling= None,
                 channel_random_noise= None, bluring= None,
                 worse_quality= None,
                 mixup = None,
                 pool_workers=4)

    # create model
    model=get_modified_VGGFace2_resnet_model(dense_neurons_after_conv=(1024,512),
                                       dropout=0.5,
                                       regularization=tf.keras.regularizers.l2(0.0001),
                                       output_neurons=(num_classes,), pooling_at_the_end='avg',
                                       pretrained= True,
                                       path_to_weights = r'D:\PycharmProjects\Denis\vggface2_Keras\vggface2_Keras\model\resnet50_softmax_dim512\weights.h5')
    # freeze model up to 4th block
    for i in range(141):
        model.layers[i].trainable=False
    model.summary()

    # create logger
    logger = open(os.path.join(path_to_save_model_and_results, 'val_logs.txt'), mode='w')
    logger.close()
    logger = open(os.path.join(path_to_save_model_and_results, 'val_logs.txt'), mode='a')
    # write training params and all important information:
    logger.write('# Train params:\n')
    logger.write('Database:%s\n' % "DAiSEE")
    logger.write('Epochs:%i\n' % epochs)
    logger.write('Highest_lr:%f\n' % highest_lr)
    logger.write('Lowest_lr:%f\n' % lowest_lr)
    logger.write('Optimizer:%s\n' % optimizer)
    logger.write('Loss:%s\n' % 'focal loss (gamma=2)')
    logger.write('Additional info:%s\n' %
                 'VGGFace2 model with 1024-512-128-4 dense layers. Engagement recognition task with 4 classes.')

    # create callbacks
    callbacks=[validation_with_generator_callback_multilabel(test_gen, metrics=(partial(f1_score, average='macro'),
                                                                        accuracy_score,
                                                                        partial(recall_score, average='macro')),
                                                                        num_label_types=4,
                                                                        num_metric_to_set_weights=2,
                                                                        logger=logger)]

    # create metrics
    metrics=[tf.keras.metrics.CategoricalAccuracy(),tf.keras.metrics.Recall()]

    # define focal loss
    losses = {'dense_3':categorical_focal_loss(alpha=class_weights, gamma=focal_loss_gamma),
    }
    """loss_weights={
        'dense_2': 1.0,
        'dense_4': 0.33,
        'dense_6': 0.33,
        'dense_8': 0.33
    }"""
    #losses=tf.keras.losses.categorical_crossentropy
    tf.keras.utils.plot_model(model, 'model.png')
    model=train_model(train_gen, model, optimizer, losses, epochs,
                      dev_gen, metrics, callbacks, path_to_save_results='results')
    model.save_weights(os.path.join(path_to_save_model_and_results, "model_weights.h5"))
    logger.close()



