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

from tensorflow_utils.Layers import Non_local_block_multi_head
from tensorflow_utils.keras_datagenerators.ImageDataLoader import ImageDataLoader
from tensorflow_utils.keras_datagenerators.ImageDataLoader_multilabel import ImageDataLoader_multilabel
from tensorflow_utils.keras_datagenerators.ImageDataPreprocessor import ImageDataPreprocessor
from preprocessing.data_normalizing_utils import VGGFace2_normalization
from tensorflow_utils.Losses import weighted_categorical_crossentropy
from tensorflow_utils.callbacks import best_weights_setter_callback, get_annealing_LRreduce_callback, \
    validation_with_generator_callback_multioutput_same_labeltype
from tensorflow_utils.models.CNN_models import get_modified_VGGFace2_resnet_model, _get_pretrained_VGGFace2_model

"""from preprocessing.data_preprocessing.image_preprocessing_utils import load_image, save_image, resize_image
from preprocessing.face_recognition_utils import recognize_the_most_confident_person_retinaFace, \
    extract_face_according_bbox, load_and_prepare_detector_retinaFace"""


class Label(NamedTuple):
    # TODO: write description
    boredom: int
    engagement: int
    confusion: int
    frustration: int


def sort_images_according_their_class(path_to_images: str, output_path: str, path_to_labels: str):
    dict_labels = load_labels_to_dict(path_to_labels)
    dirs_with_images = os.listdir(path_to_images)
    # check if output path is existed
    if not os.path.exists(output_path):
        os.makedirs(output_path, exist_ok=True)
    num_classes = np.unique(np.array(list(x.engagement for x in dict_labels.values()))).shape[0]
    # create subdirectories for classes
    for num_class in range(num_classes):
        if not os.path.exists(os.path.join(output_path, str(num_class))):
            os.makedirs(os.path.join(output_path, str(num_class)), exist_ok=True)
    # copy images according their class
    for dir_with_images in dirs_with_images:
        if not dir_with_images in dict_labels.keys():
            continue
        class_num = dict_labels[dir_with_images].engagement
        image_filenames = os.listdir(os.path.join(path_to_images, dir_with_images))
        for image_filename in image_filenames:
            shutil.copy(os.path.join(path_to_images, dir_with_images, image_filename),
                        os.path.join(output_path, str(class_num), image_filename))


def load_labels_to_dict(path: str) -> Dict[str, Label]:
    # TODO:write description
    labels_df = pd.read_csv(path)
    labels_df['ClipID'] = labels_df['ClipID'].apply(lambda x: x.split('.')[0])
    # labels_df.columns=[labels_df.columns[0]]+[x.lower() for x in labels_df.columns[1:]]
    labels_dict = dict(
        (row[1].iloc[0],
         Label(*row[1].iloc[1:].values))
        for row in labels_df.iterrows()
    )
    return labels_dict


def form_dataframe_of_relative_paths_to_data_with_labels(path_to_data: str,
                                                         labels_dict: Dict[str, Label]) -> pd.DataFrame:
    # TODO: write description
    directories_according_path = os.listdir(path_to_data)
    df_with_relative_paths_and_labels = pd.DataFrame(columns=['filename', 'class'])
    for dir in directories_according_path:
        if not dir in labels_dict.keys():
            continue
        img_filenames = os.listdir(os.path.join(path_to_data, dir))
        img_filenames = [os.path.join(dir, x) for x in img_filenames]
        label = labels_dict[dir].engagement
        labels = [label for _ in range(len(img_filenames))]
        tmp_df = pd.DataFrame(data=np.array([img_filenames, labels]).T, columns=['filename', 'class'])
        df_with_relative_paths_and_labels = df_with_relative_paths_and_labels.append(tmp_df)
    return df_with_relative_paths_and_labels


def form_dataframe_of_relative_paths_to_data_with_multilabels(path_to_data: str,
                                                              labels_dict: Dict[str, Label]) -> pd.DataFrame:
    # TODO: write description
    directories_according_path = os.listdir(path_to_data)
    df_with_relative_paths_and_labels = pd.DataFrame(
        columns=['filename', 'engagement', 'boredom', 'confusion', 'frustration'])
    for dir in directories_according_path:
        if not dir in labels_dict.keys():
            continue
        img_filenames = os.listdir(os.path.join(path_to_data, dir))
        img_filenames = [os.path.join(dir, x) for x in img_filenames]
        label = [labels_dict[dir].engagement, labels_dict[dir].boredom, labels_dict[dir].confusion,
                 labels_dict[dir].frustration]
        labels = [[label[0], label[1], label[2], label[3]] for _ in range(len(img_filenames))]
        tmp_df = pd.DataFrame(data=np.concatenate([np.array(img_filenames).reshape(-1, 1), np.array(labels)], axis=-1),
                              columns=['filename', 'engagement', 'boredom', 'confusion', 'frustration'])
        df_with_relative_paths_and_labels = df_with_relative_paths_and_labels.append(tmp_df)
    return df_with_relative_paths_and_labels


def get_modified_VGGFace2_resnet_model(dense_neurons_after_conv: Tuple[int, ...],
                                       dropout: float = 0.3,
                                       regularization: Optional[tf.keras.regularizers.Regularizer] = None,
                                       output_neurons: Union[Tuple[int, ...], int] = 7,
                                       pretrained: bool = True,
                                       path_to_weights: Optional[str] = None,
                                       multi_head_attention: bool = True,
                                       num_classes: int = 4) -> tf.keras.Model:
    pretrained_VGGFace2 = _get_pretrained_VGGFace2_model(path_to_weights, pretrained=pretrained)
    x = pretrained_VGGFace2.get_layer('activation_48').output

    class_outputs = []
    if multi_head_attention:
        for _ in range(num_classes):
            class_x = Non_local_block_multi_head(num_heads=2, output_channels=512,
                                                 head_output_channels=None,
                                                 downsize_factor=8,
                                                 shortcut_connection=True)(x)
            class_x = tf.keras.layers.BatchNormalization()(class_x)
            class_x = tf.keras.layers.GlobalAveragePooling2D()(class_x)

            for dense_layer_idx in range(len(dense_neurons_after_conv) - 1):
                num_neurons_on_layer = dense_neurons_after_conv[dense_layer_idx]
                class_x = tf.keras.layers.Dense(num_neurons_on_layer, activation='relu',
                                                kernel_regularizer=regularization)(class_x)
                if dropout:
                    class_x = tf.keras.layers.Dropout(dropout)(class_x)
                # pre-last Dense layer
            num_neurons_on_layer = dense_neurons_after_conv[-1]
            class_x = tf.keras.layers.Dense(num_neurons_on_layer, activation='relu')(class_x)
            class_x = tf.keras.layers.Dense(1, activation='sigmoid')(class_x)
            class_outputs.append(class_x)

    model = tf.keras.Model(inputs=pretrained_VGGFace2.inputs, outputs=class_outputs)
    del pretrained_VGGFace2
    return model


def train_model(train_generator: Iterable[Tuple[np.ndarray, np.ndarray]], model: tf.keras.Model,
                optimizer: tf.keras.optimizers.Optimizer, loss: tf.keras.losses.Loss,
                epochs: int,
                val_generator: Iterable[Tuple[np.ndarray, np.ndarray]],
                metrics: List[tf.keras.metrics.Metric],
                callbacks: List[tf.keras.callbacks.Callback],
                path_to_save_results: str,
                class_weights: Optional[Dict[int, float]] = None,
                loss_weights: Optional[Dict['str', float]] = None) -> tf.keras.Model:
    # create directory for saving results
    if not os.path.exists(path_to_save_results):
        os.makedirs(path_to_save_results)
    # compile model
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics, loss_weights=loss_weights)
    model.fit(train_generator, epochs=epochs, callbacks=callbacks, validation_data=val_generator, verbose=2,
              class_weight=class_weights)
    return model


def generate_paths_with_labels_from_directory(path_to_dir: str, class_name: str, class_value: int) -> pd.DataFrame:
    # TODO: write description
    filenames = os.listdir(path_to_dir)
    filenames = [os.path.join(path_to_dir, filename) for filename in filenames]
    class_values = [class_value for _ in range(len(filenames))]
    df = pd.DataFrame(data=np.array([filenames, class_values]).T,
                      columns=['filename', class_name])
    df[class_name] = df[class_name].astype('float32')
    return df


if __name__ == '__main__':
    '''path_to_directory_with_frames=r'D:\Databases\DAiSEE\frames'
    path_to_output_directory=r'D:\Databases\DAiSEE\extracted_faces'
    resize=(224,224)
    extract_faces_from_all_subdirectories_in_directory(path_to_directory_with_frames, path_to_output_directory, resize)'''
    # params
    path_to_train_frames = r'C:\Databases\DAiSEE\train_preprocessed\extracted_faces'
    path_to_train_labels = r'C:\Databases\DAiSEE\Labels\TrainLabels.csv'
    path_to_dev_frames = r'C:\Databases\DAiSEE\dev_preprocessed\extracted_faces'
    path_to_dev_labels = r'C:\Databases\DAiSEE\Labels\ValidationLabels.csv'
    '''output_path=r'D:\Databases\DAiSEE\dev_preprocessed\sorted_faces'
    sort_images_according_their_class(path_to_images=path_to_dev_frames, output_path=output_path,
                                      path_to_labels=path_to_dev_labels)'''
    input_shape = (224, 224, 3)
    num_classes = 2
    batch_size = 64
    epochs = 30
    highest_lr = 0.0005
    lowest_lr = 0.00001
    momentum = 0.9
    output_path = 'results'
    # create output path
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    optimizer = tf.keras.optimizers.SGD(highest_lr, momentum=momentum)
    loss = tf.keras.losses.categorical_crossentropy
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
    labels_train['confusion'] = labels_train['confusion'].astype('float32')
    labels_train['frustration'] = labels_train['frustration'].astype('float32')
    labels_dev['confusion'] = labels_dev['confusion'].astype('float32')
    labels_dev['frustration'] = labels_dev['frustration'].astype('float32')
    # add augmentation data and delete all non-engagement class values
    labels_train = labels_train.drop(columns=['boredom', 'confusion', 'frustration'])
    labels_dev = labels_dev.drop(columns=['boredom', 'confusion', 'frustration'])

    # add augmented images
    class_0_augmented = generate_paths_with_labels_from_directory(
        path_to_dir=r'C:\Databases\DAiSEE\train_preprocessed\augmentation\engagement\0',
        class_name='engagement',
        class_value=0)
    class_1_augmented = generate_paths_with_labels_from_directory(
        path_to_dir=r'C:\Databases\DAiSEE\train_preprocessed\augmentation\engagement\1',
        class_name='engagement',
        class_value=1)
    labels_train = pd.concat([labels_train, class_0_augmented, class_1_augmented], axis=0)
    # add SMOTE images
    class_0_SMOTE = generate_paths_with_labels_from_directory(
        path_to_dir=r'C:\Databases\DAiSEE\train_preprocessed\SMOTE_images\0',
        class_name='engagement',
        class_value=0)
    class_1_SMOTE = generate_paths_with_labels_from_directory(
        path_to_dir=r'C:\Databases\DAiSEE\train_preprocessed\SMOTE_images\1',
        class_name='engagement',
        class_value=1)
    labels_train = pd.concat([labels_train, class_0_SMOTE, class_1_SMOTE], axis=0)
    """ # turn 4-class task into 2-class task
    labels_train.loc[(labels_train['engagement'] == 1),'engagement'] = 0
    labels_train.loc[(labels_train['engagement'] == 2),'engagement'] = 1
    labels_train.loc[(labels_train['engagement'] == 3),'engagement'] = 1

    labels_dev.loc[(labels_dev['engagement'] == 1),'engagement'] = 0
    labels_dev.loc[(labels_dev['engagement'] == 2),'engagement'] = 1
    labels_dev.loc[(labels_dev['engagement'] == 3),'engagement'] = 1
    num_classes = 2"""
    # 1-vs-all mode
    labels_train['engagement_1'] = labels_train['engagement'].copy()
    labels_train['engagement_2'] = labels_train['engagement'].copy()
    labels_train['engagement_3'] = labels_train['engagement'].copy()
    labels_train.columns = ['filename', 'engagement_0', 'engagement_1', 'engagement_2', 'engagement_3']
    index_all = labels_train['engagement_0'] != 0
    index_not_all = labels_train['engagement_0'] == 0
    labels_train.loc[index_all, 'engagement_0'] = 0
    labels_train.loc[index_not_all, 'engagement_0'] = 1
    labels_train.loc[(labels_train['engagement_1'] != 1), 'engagement_1'] = 0
    labels_train.loc[(labels_train['engagement_1'] == 1), 'engagement_1'] = 1
    labels_train.loc[(labels_train['engagement_2'] != 2), 'engagement_2'] = 0
    labels_train.loc[(labels_train['engagement_2'] == 2), 'engagement_2'] = 1
    labels_train.loc[(labels_train['engagement_3'] != 3), 'engagement_3'] = 0
    labels_train.loc[(labels_train['engagement_3'] == 3), 'engagement_3'] = 1

    labels_dev['engagement_1'] = labels_dev['engagement'].copy()
    labels_dev['engagement_2'] = labels_dev['engagement'].copy()
    labels_dev['engagement_3'] = labels_dev['engagement'].copy()
    labels_dev.columns = ['filename', 'engagement_0', 'engagement_1', 'engagement_2', 'engagement_3']
    index_all = labels_dev['engagement_0'] != 0
    index_not_all = labels_dev['engagement_0'] == 0
    labels_dev.loc[index_all, 'engagement_0'] = 0
    labels_dev.loc[index_not_all, 'engagement_0'] = 1
    labels_dev.loc[(labels_dev['engagement_1'] != 1), 'engagement_1'] = 0
    labels_dev.loc[(labels_dev['engagement_1'] == 1), 'engagement_1'] = 1
    labels_dev.loc[(labels_dev['engagement_2'] != 2), 'engagement_2'] = 0
    labels_dev.loc[(labels_dev['engagement_2'] == 2), 'engagement_2'] = 1
    labels_dev.loc[(labels_dev['engagement_3'] != 3), 'engagement_3'] = 0
    labels_dev.loc[(labels_dev['engagement_3'] == 3), 'engagement_3'] = 1
    # class weights
    """class_weights_engagement=class_weight.compute_class_weight(class_weight='balanced',
                                                               classes=np.unique(labels_train['engagement']),
                                                               y=labels_train['engagement'].values.reshape((-1,)))
    #class_weights_engagement/=class_weights_engagement.sum()
    class_weights_engagement=dict((i,class_weights_engagement[i]) for i in range(len(class_weights_engagement)))"""

    """class_weights_boredom = class_weight.compute_class_weight(class_weight='balanced',
                                                                 classes=np.unique(labels_train['boredom']),
                                                                 y=labels_train['boredom'].values.reshape((-1,)))
    class_weights_boredom /= class_weights_boredom.sum()
    #class_weights_boredom = dict((i, class_weights_boredom[i]) for i in range(len(class_weights_boredom)))

    class_weights_confusion = class_weight.compute_class_weight(class_weight='balanced',
                                                                 classes=np.unique(labels_train['confusion']),
                                                                 y=labels_train['confusion'].values.reshape((-1,)))
    class_weights_confusion /= class_weights_confusion.sum()
    #class_weights_confusion = dict((i, class_weights_confusion[i]) for i in range(len(class_weights_confusion)))

    class_weights_frustration = class_weight.compute_class_weight(class_weight='balanced',
                                                                 classes=np.unique(labels_train['frustration']),
                                                                 y=labels_train['frustration'].values.reshape((-1,)))
    class_weights_frustration /= class_weights_frustration.sum()
    #class_weights_frustration = dict((i, class_weights_frustration[i]) for i in range(len(class_weights_frustration)))"""

    '''# Make major class less presented
    labels_train=pd.concat([labels_train[labels_train['class']==0],
                            labels_train[labels_train['class'] == 1],
                            labels_train[labels_train['class'] == 2].iloc[::5],
                            labels_train[labels_train['class'] == 3].iloc[::5]
                            ])'''
    #labels_dev=labels_train.sample(frac=1)
    #labels_train=labels_train.iloc[:640]
    #labels_dev = labels_dev.iloc[:640]
    # if we use ImageDataLoader, not multilabel
    # labels_train.columns=['filename', 'class']
    # labels_dev.columns = ['filename', 'class']
    # create generators
    train_gen = ImageDataLoader_multilabel(paths_with_labels=labels_train, batch_size=batch_size,
                                           class_columns=['engagement_0', 'engagement_1', 'engagement_2',
                                                          'engagement_3'],
                                           preprocess_function=VGGFace2_normalization,
                                           num_classes=num_classes,
                                           horizontal_flip=0.1, vertical_flip=0,
                                           shift=0.1,
                                           brightness=0.1, shearing=0.1, zooming=0.1,
                                           random_cropping_out=0.1, rotation=0.1,
                                           scaling=None,
                                           channel_random_noise=0.1, bluring=0.1,
                                           worse_quality=0.1,
                                           mixup=None,
                                           pool_workers=12)

    dev_gen = ImageDataLoader_multilabel(paths_with_labels=labels_dev, batch_size=batch_size,
                                         class_columns=['engagement_0', 'engagement_1', 'engagement_2', 'engagement_3'],
                                         preprocess_function=VGGFace2_normalization,
                                         num_classes=num_classes,
                                         horizontal_flip=0, vertical_flip=0,
                                         shift=0,
                                         brightness=0, shearing=0, zooming=0,
                                         random_cropping_out=0, rotation=0,
                                         scaling=None,
                                         channel_random_noise=0, bluring=0,
                                         worse_quality=0,
                                         mixup=None,
                                         pool_workers=8)
    # create model
    model = get_modified_VGGFace2_resnet_model(dense_neurons_after_conv=(128,),
                                               dropout=0.5,
                                               regularization=tf.keras.regularizers.l2(0.0001),
                                               output_neurons=(num_classes, num_classes, num_classes, num_classes),
                                               pretrained=True,
                                               path_to_weights=r'D:\PycharmProjects\Denis\vggface2_Keras\vggface2_Keras\model\resnet50_softmax_dim512\weights.h5')


    # freeze model
    for i in range(141):# end of 5th block - 172 (activation), end of 4th block - 141
        model.layers[i].trainable = False
    model.summary()
    tf.keras.utils.plot_model(model, show_shapes=True)
    # create logger
    logger = open(os.path.join(output_path, 'val_logs.txt'), mode='w')
    logger.close()
    logger = open(os.path.join(output_path, 'val_logs.txt'), mode='a')
    # write training params:
    logger.write('# Train params:\n'
                 'Epochs:%i\n'
                 'Highest_lr:%f\n'
                 'Lowest_lr:%f\n'
                 'Optimizer:%s\n'
                 'Loss:%s\n' %
                 (epochs, highest_lr, lowest_lr, optimizer, loss))
    # create callbacks
    callbacks = [validation_with_generator_callback_multioutput_same_labeltype(dev_gen, metrics=(partial(f1_score, average='macro'),
                                                                                 accuracy_score,
                                                                                 partial(recall_score,
                                                                                         average='macro')),
                                                                num_outputs=4,
                                                               num_metric_to_set_weights=1,
                                                               logger=logger),
                 get_annealing_LRreduce_callback(highest_lr, lowest_lr, 30)]
    # create metrics
    metrics = [tf.keras.metrics.BinaryAccuracy(), tf.keras.metrics.Recall()]
    # make weighted losses for model
    losses = {'dense_1': tf.keras.losses.binary_crossentropy,
              'dense_3': tf.keras.losses.binary_crossentropy,
              'dense_5': tf.keras.losses.binary_crossentropy,
              'dense_7': tf.keras.losses.binary_crossentropy
              }
    loss_weights = {
        'dense_1': 1.0,
        'dense_3': 1.0,
        'dense_5': 1.0,
        'dense_7': 1.0
    }
    # losses=tf.keras.losses.categorical_crossentropy
    tf.keras.utils.plot_model(model, 'results\\model.png')
    model = train_model(train_gen, model, optimizer, losses, epochs,
                        None, metrics, callbacks, path_to_save_results='results', loss_weights=loss_weights)
    model.save_weights("results\\model_weights.h5")
    model.save("results\\model.h5")
