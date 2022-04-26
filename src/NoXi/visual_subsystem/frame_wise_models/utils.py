from typing import Tuple, Optional

import pandas as pd
import numpy as np
import os
from src.NoXi.preprocessing.data_preprocessing import generate_rel_paths_to_images_in_all_dirs
from src.NoXi.preprocessing.labels_preprocessing import load_all_labels_by_paths, \
    combine_path_to_images_with_labels_many_videos, generate_paths_to_labels





def load_and_preprocess_data(path_to_data: str, path_to_labels: str, frame_step: int, shuffle_train:Optional[bool]=False) -> Tuple[
    pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """

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
    # change the keys of train_labels/dev_labels/test_labels to have only the name with pattern name_of_video/novice_or_expert
    for key in list(train_labels.keys()):
        new_key = key.split(os.path.sep)[-2] + '/'
        new_key = new_key + 'expert' if 'expert' in key.split(os.path.sep)[-1] else new_key + 'novice'
        train_labels[new_key] = train_labels.pop(key)
    for key in list(dev_labels.keys()):
        new_key = key.split(os.path.sep)[-2] + '/'
        new_key = new_key + 'expert' if 'expert' in key.split(os.path.sep)[-1] else new_key + 'novice'
        dev_labels[new_key] = dev_labels.pop(key)
    for key in list(test_labels.keys()):
        new_key = key.split(os.path.sep)[-2] + '/'
        new_key = new_key + 'expert' if 'expert' in key.split(os.path.sep)[-1] else new_key + 'novice'
        test_labels[new_key] = test_labels.pop(key)
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
    if shuffle_train:
        train_image_paths_and_labels = train_image_paths_and_labels.sample(frac=1).reset_index(drop=True)
    # convert dev and test labels to the categories (it is easier to process them like this)
    dev_labels = np.argmax(dev_image_paths_and_labels.iloc[:, 1:].values, axis=1, keepdims=True)
    dev_image_paths_and_labels = dev_image_paths_and_labels.iloc[:, :1]
    dev_image_paths_and_labels['class'] = dev_labels

    test_labels = np.argmax(test_image_paths_and_labels.iloc[:, 1:].values, axis=1, keepdims=True)
    test_image_paths_and_labels = test_image_paths_and_labels.iloc[:, :1]
    test_image_paths_and_labels['class'] = test_labels
    # create abs path for all paths instead of relative (needed for generator)
    train_image_paths_and_labels['filename'] = train_image_paths_and_labels['filename'].apply(
        lambda x: os.path.join(path_to_data, x))
    dev_image_paths_and_labels['filename'] = dev_image_paths_and_labels['filename'].apply(
        lambda x: os.path.join(path_to_data, x))
    test_image_paths_and_labels['filename'] = test_image_paths_and_labels['filename'].apply(
        lambda x: os.path.join(path_to_data, x))
    # done
    return (train_image_paths_and_labels, dev_image_paths_and_labels, test_image_paths_and_labels)