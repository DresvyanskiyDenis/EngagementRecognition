import copy
from typing import List, Dict, Optional

import numpy as np
import pandas as pd
import tensorflow as tf
import os
import glob

from src.NoXi.preprocessing.data_preprocessing import generate_rel_paths_to_images_in_all_dirs
from src.NoXi.preprocessing.labels_preprocessing import read_noxi_label_file, transform_time_continuous_to_categorical, \
    clean_labels, average_from_several_labels, load_all_labels_by_paths, transform_all_labels_to_categorical, \
    combine_path_to_images_with_labels_many_videos
from tensorflow_utils.models.CNN_models import get_modified_VGGFace2_resnet_model



def create_VGGFace2_model(path_to_weights:str, num_classes:Optional[int]=4)->tf.keras.Model:
    """Creates the VGGFace2 model and loads weights for it using proviede path.

    :param path_to_weights: str
            Path to the weights for VGGFace2 model.
    :param num_classes: int
            Number of classes to define last softmax layer .
    :return: tf.keras.Model
            Created tf.keras.Model with loaded weights.
    """
    model=get_modified_VGGFace2_resnet_model(dense_neurons_after_conv=(512,),
                                       dropout= 0.3,
                                       regularization=tf.keras.regularizers.l2(0.0001),
                                       output_neurons= num_classes, pooling_at_the_end= 'avg',
                                       pretrained = True,
                                       path_to_weights = path_to_weights)
    return model

def load_and_preprocess_data(path_to_data:str, path_to_labels:str,
                             class_barriers:np.array, frame_step:int)->pd.DataFrame:
    """TODO: complete function

    :param path_to_data:
    :param path_to_labels:
    :return:
    """
    paths_to_frames=generate_rel_paths_to_images_in_all_dirs(path_to_data)
    paths_to_labels=glob.glob(os.path.join(path_to_labels,'**'))
    labels=load_all_labels_by_paths(paths_to_labels)
    labels=transform_all_labels_to_categorical(labels,class_barriers)

    paths_to_frames_with_labels=combine_path_to_images_with_labels_many_videos(paths_to_frames, labels,
                                                                               frame_step=frame_step)
    return paths_to_frames_with_labels



def main():
    path_to_data = ""
    path_to_labels = ""
    pass


if __name__ == '__main__':
    main()
