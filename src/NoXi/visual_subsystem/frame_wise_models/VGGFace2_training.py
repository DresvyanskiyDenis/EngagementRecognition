import sys

from sklearn.metrics import recall_score, precision_score, f1_score

from tensorflow_utils.wandb_callbacks import WandB_LR_log_callback, WandB_val_metrics_callback

sys.path.extend(["/work/home/dsu/datatools/"])
sys.path.extend(["/work/home/dsu/engagement_recognition_project_server/"])

import copy
import gc
from typing import List, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
import os
import wandb
import glob

from keras.callbacks import EarlyStopping
from wandb.integration.keras import WandbCallback



from preprocessing.data_normalizing_utils import VGGFace2_normalization
from src.NoXi.preprocessing.data_preprocessing import generate_rel_paths_to_images_in_all_dirs
from src.NoXi.preprocessing.labels_preprocessing import read_noxi_label_file, transform_time_continuous_to_categorical, \
    clean_labels, average_from_several_labels, load_all_labels_by_paths, transform_all_labels_to_categorical, \
    combine_path_to_images_with_labels_many_videos, generate_paths_to_labels
from tensorflow_utils.callbacks import get_annealing_LRreduce_callback, get_reduceLRonPlateau_callback
from tensorflow_utils.keras_datagenerators.ImageDataLoader import ImageDataLoader
from tensorflow_utils.models.CNN_models import get_modified_VGGFace2_resnet_model


def create_VGGFace2_model(path_to_weights: str, num_classes: Optional[int] = 4) -> tf.keras.Model:
    """Creates the VGGFace2 model and loads weights for it using proviede path.

    :param path_to_weights: str
            Path to the weights for VGGFace2 model.
    :param num_classes: int
            Number of classes to define last softmax layer .
    :return: tf.keras.Model
            Created tf.keras.Model with loaded weights.
    """
    model = get_modified_VGGFace2_resnet_model(dense_neurons_after_conv=(512,),
                                               dropout=0.3,
                                               regularization=tf.keras.regularizers.l2(0.0001),
                                               output_neurons=num_classes, pooling_at_the_end='avg',
                                               pretrained=True,
                                               path_to_weights=path_to_weights)
    return model


def load_and_preprocess_data(path_to_data: str, path_to_labels: str,
                             class_barriers: np.array, frame_step: int) -> Tuple[
    pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """TODO: complete function

    :param path_to_data:
    :param path_to_labels:
    :return:
    """
    # generate paths to images (data)
    paths_to_images = generate_rel_paths_to_images_in_all_dirs(path_to_data, image_format="png")
    # generate paths to train/dev/test labels
    paths_train_labels = generate_paths_to_labels(os.path.join(path_to_labels, "train"))
    paths_dev_labels = generate_paths_to_labels(os.path.join(path_to_labels, "dev"))
    paths_test_labels = generate_paths_to_labels(os.path.join(path_to_labels, "test"))
    # load labels
    train_labels = load_all_labels_by_paths(paths_train_labels)
    dev_labels = load_all_labels_by_paths(paths_dev_labels)
    test_labels = load_all_labels_by_paths(paths_test_labels)
    del paths_train_labels, paths_dev_labels, paths_test_labels
    # transform labels to categorical
    train_labels = transform_all_labels_to_categorical(train_labels, class_barriers)
    dev_labels = transform_all_labels_to_categorical(dev_labels, class_barriers)
    test_labels = transform_all_labels_to_categorical(test_labels, class_barriers)
    # change the keys of train_labels/dev_labels/test_labels to have only the name with pattern name_of_video/novice_or_expert
    for key in list(train_labels.keys()):
        new_key=key.split(os.path.sep)[-2]+'/'
        new_key=new_key+'expert' if 'expert' in key.split(os.path.sep)[-1] else new_key+'novice'
        train_labels[new_key]=train_labels.pop(key)
    for key in list(dev_labels.keys()):
        new_key=key.split(os.path.sep)[-2]+'/'
        new_key = new_key + 'expert' if 'expert' in key.split(os.path.sep)[-1] else new_key + 'novice'
        dev_labels[new_key]=dev_labels.pop(key)
    for key in list(test_labels.keys()):
        new_key=key.split(os.path.sep)[-2]+'/'
        new_key = new_key + 'expert' if 'expert' in key.split(os.path.sep)[-1] else new_key + 'novice'
        test_labels[new_key]=test_labels.pop(key)
    # combine paths to images (data) with labels
    train_image_paths_and_labels = combine_path_to_images_with_labels_many_videos(paths_with_images=paths_to_images,
                                                                                  labels=train_labels,
                                                                                  sample_rate_annotations=25,
                                                                                  frame_step=frame_step)
    dev_image_paths_and_labels = combine_path_to_images_with_labels_many_videos(paths_with_images=paths_to_images,
                                                                                labels=dev_labels,
                                                                                sample_rate_annotations=25,
                                                                                frame_step=5)
    test_image_paths_and_labels = combine_path_to_images_with_labels_many_videos(paths_with_images=paths_to_images,
                                                                                 labels=test_labels,
                                                                                 sample_rate_annotations=25,
                                                                                 frame_step=5)
    # shuffle train data
    train_image_paths_and_labels = train_image_paths_and_labels.sample(frac=1).reset_index(drop=True)
    # create abs path for all paths instead of relative (needed for generator)
    train_image_paths_and_labels['filename']=train_image_paths_and_labels['filename'].apply(lambda x:os.path.join(path_to_data, x))
    dev_image_paths_and_labels['filename'] = dev_image_paths_and_labels['filename'].apply(lambda x: os.path.join(path_to_data, x))
    test_image_paths_and_labels['filename'] = test_image_paths_and_labels['filename'].apply(lambda x: os.path.join(path_to_data, x))
    # done
    return (train_image_paths_and_labels, dev_image_paths_and_labels, test_image_paths_and_labels)

def train():
    # metaparams
    metaparams = {
        "optimizer": "Adam",  # SGD, Nadam
        "learning_rate_max": 0.001,  # up to 0.0001
        "learning_rate_min": 0.00001,  # up to 0.000001
        "lr_scheduller": "Cyclic",  # "reduceLRonPlateau"
        "annealing_period": 5,
        "epochs": 20,
        "batch_size": 110,
        "augmentation_rate:": 0.1,  # 0.2, 0.3
        "loss_function": 'categorical_crossentropy',
        "architecture": "VGGFace2_full_training",
        "dataset": "NoXi_english"
    }

    # initialization of Weights and Biases
    wandb.init(project="VGGFace2_FtF_training", config=metaparams)
    config = wandb.config

    # loading data
    path_to_data = "/Noxi_extracted/NoXi/extracted_faces/"
    path_to_labels = "/media/external_hdd_1/Noxi_labels_gold_standard/English"
    class_barriers = np.array([0.45, 0.6, 0.8])
    frame_step = 5
    train, dev, test = load_and_preprocess_data(path_to_data, path_to_labels,
                                                class_barriers, frame_step)

    ################################# DANGER, DELETE IT
    train=train[:1000]
    dev=dev[:1000]
    print(train.shape, dev.shape)
    ######################################
    # Metaparams initialization
    metrics = ['accuracy']
    if config.lr_scheduller=='Cyclic':
        lr_scheduller = get_annealing_LRreduce_callback(highest_lr=config.learning_rate_max,
                                                    lowest_lr=config.learning_rate_min,
                                                    annealing_period=config.annealing_period)
    elif config.lr_scheduller=='reduceLRonPlateau':
        lr_scheduller=get_reduceLRonPlateau_callback(monitoring_loss = 'val_loss', reduce_factor = 0.1,
                                   num_patient_epochs = 3,
                                   min_lr = config.learning_rate_min)
    else:
        raise Exception("You passed wrong lr_scheduller.")

    if config.optimizer == 'Adam':
        optimizer = tf.keras.optimizers.Adam(config.learning_rate_max)
    elif config.optimizer == 'Nadam':
        optimizer = tf.keras.optimizers.Nadam(config.learning_rate_max)
    elif config.optimizer == 'SGD':
        optimizer = tf.keras.optimizers.SGD(config.learning_rate_max)
    else:
        raise Exception("You passed wrong optimizer name.")

    # model initialization
    model = create_VGGFace2_model(path_to_weights='/work/home/dsu/VGG_model_weights/resnet50_softmax_dim512/weights.h5',
                                  num_classes=4)
    model.compile(loss=config.loss_function, optimizer=optimizer, metrics=metrics)
    model.summary()

    # create DataLoaders (DataGenerator)
    train_data_loader = ImageDataLoader(paths_with_labels=train, batch_size=config.batch_size,
                                        preprocess_function=VGGFace2_normalization,
                                        num_classes=4,
                                        horizontal_flip=config.augmentation_rate,
                                        vertical_flip=config.augmentation_rate,
                                        shift=config.augmentation_rate,
                                        brightness=config.augmentation_rate, shearing=config.augmentation_rate,
                                        zooming=config.augmentation_rate,
                                        random_cropping_out=config.augmentation_rate, rotation=config.augmentation_rate,
                                        scaling=config.augmentation_rate,
                                        channel_random_noise=config.augmentation_rate, bluring=config.augmentation_rate,
                                        worse_quality=config.augmentation_rate,
                                        mixup=None,
                                        prob_factors_for_each_class=None,
                                        pool_workers=24)

    dev_data_loader = ImageDataLoader(paths_with_labels=dev, batch_size=config.batch_size,
                                      preprocess_function=VGGFace2_normalization,
                                      num_classes=4,
                                      horizontal_flip=0, vertical_flip=0,
                                      shift=0,
                                      brightness=0, shearing=0, zooming=0,
                                      random_cropping_out=0, rotation=0,
                                      scaling=0,
                                      channel_random_noise=0, bluring=0,
                                      worse_quality=0,
                                      mixup=None,
                                      prob_factors_for_each_class=None,
                                      pool_workers=16)

    # create Keras Callbacks for monitoring learning rate and metrics on val_set
    lr_monitor_callback =WandB_LR_log_callback(optimizer)
    val_metrics={
        'recall':recall_score,
        'precision':precision_score,
        'f1_score:':f1_score
    }
    val_metrics_callback=WandB_val_metrics_callback(dev_data_loader, val_metrics)
    early_stopping_callback=EarlyStopping(monitor='val_loss', patience=5, verbose=1)

    print(config.epochs, config.batch_size)
    print(dev.shape)
    print("####################################")
    print(train.shape)
    # train process
    print("new configuration")
    model.fit(train_data_loader, epochs=config.epochs, validation_data=dev_data_loader,
              callbacks=[WandbCallback(),
                         lr_scheduller,
                         early_stopping_callback,
                         lr_monitor_callback,
                         val_metrics_callback])
    # clear RAM
    del train, dev, test
    del train_data_loader, dev_data_loader
    del model
    gc.collect()
    tf.keras.backend.clear_session()

def main():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    sweep_config= {
        'method':'random',
        'metric':{
            'name':'val_loss',
            'goal':'minimize'
        },
        'parameters': {
            'optimizer':{
                'values':['Adam', 'SGD', 'Nadam']
            },
            'learning_rate_max':{
                'distribution': 'uniform',
                'max':0.001,
                'min':0.0001
            },
            'learning_rate_min':{
                'distribution': 'uniform',
                'max': 0.00001,
                'min': 0.000001
            },
            'lr_scheduller': {
                'values':['Cyclic', 'reduceLRonPlateau']
            },
            'augmentation_rate': {
                'values':[0.1, 0.2, 0.3]
            }
        }
    }
    sweep_id=wandb.sweep(sweep_config, project='VGGFace2_FtF_training')
    wandb.agent(sweep_id, function=train, count=20, project='VGGFace2_FtF_training')



if __name__ == '__main__':
    main()
