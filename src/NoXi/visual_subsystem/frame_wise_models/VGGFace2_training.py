import copy
from typing import List, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
import os
import glob

from preprocessing.data_normalizing_utils import VGGFace2_normalization
from src.NoXi.preprocessing.data_preprocessing import generate_rel_paths_to_images_in_all_dirs
from src.NoXi.preprocessing.labels_preprocessing import read_noxi_label_file, transform_time_continuous_to_categorical, \
    clean_labels, average_from_several_labels, load_all_labels_by_paths, transform_all_labels_to_categorical, \
    combine_path_to_images_with_labels_many_videos, generate_paths_to_labels
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
    # done
    return (train_image_paths_and_labels, dev_image_paths_and_labels, test_image_paths_and_labels)


def main():
    path_to_data = "/media/external_hdd_1/Noxi_extracted/NoXi/extracted_faces/"
    path_to_labels = "/media/external_hdd_1/Noxi_labels_gold_standard/English"
    class_barriers=np.array([0.45, 0.6, 0.8])
    frame_step=5
    train, dev, test = load_and_preprocess_data(path_to_data, path_to_labels,
                             class_barriers, frame_step)

    # model initialization and metaparams
    model=create_VGGFace2_model(path_to_weights='/work/home/dsu/VGG_model_weights/resnet50_softmax_dim512/weights.h5', num_classes=4)
    optimizer=tf.keras.optimizers.Adam(0.0001)
    loss=tf.keras.losses.categorical_crossentropy
    metrics=['accuracy']
    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
    model.summary()

    # create DataLoader (DataGenerator)
    data_loader=ImageDataLoader(paths_with_labels=train, batch_size=64, preprocess_function=VGGFace2_normalization ,
                 num_classes = 4,
                 horizontal_flip = 0.1, vertical_flip = 0.1,
                 shift = 0.1,
                 brightness = 0.1, shearing = 0.1, zooming = 0.1,
                 random_cropping_out = 0.1, rotation = 0.1,
                 scaling = 0.1,
                 channel_random_noise = 0.1, bluring = 0.1,
                 worse_quality = 0.1,
                 mixup = None,
                 prob_factors_for_each_class= None,
                 pool_workers= 16)

    # train process
    model.fit(data_loader, epochs=10)


if __name__ == '__main__':
    main()
