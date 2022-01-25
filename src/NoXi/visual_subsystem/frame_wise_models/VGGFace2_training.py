import copy
from typing import List, Dict, Optional

import numpy as np
import pandas as pd
import tensorflow as tf
import os
import glob

from src.NoXi.preprocessing.labels_preprocessing import read_noxi_label_file, transform_time_continuous_to_categorical, \
    clean_labels, average_from_several_labels
from tensorflow_utils.models.CNN_models import get_modified_VGGFace2_resnet_model


def generate_rel_paths_to_images_in_all_dirs(path: str, image_format: str = "jpg") -> pd.DataFrame:
    """Generates relative paths to all images with specified format.
       Returns it as a DataFrame

    :param path: str
            path where all images should be found
    :return: pd.DataFrame
            relative paths to images (including filename)
    """
    # define pattern for search (in every dir and subdir the image with specified format)
    pattern = path + r"\**\*." + image_format
    # searching via this pattern
    abs_paths = glob.glob(pattern)
    # find a relative path to it
    rel_paths = [os.path.relpath(item, path) for item in abs_paths]
    # create from it a DataFrame
    paths_to_images = pd.DataFrame(columns=['rel_path'], data=np.array(rel_paths))
    return paths_to_images


def load_labels_by_path_to_dir(path: str, file_extension: str = "annotation~") -> pd.DataFrame:
    """Loads NoXi labels from files with extension annotation~. There can be several annotation files, therefore
       this function loads all of them and average then.
       Labels will be presented in pandas DataFrame.

    :param path: str
            Path to the directory with labels. There could be several label files.
    :return: pd.DataFrame
            DataFrame with labels of one file (NoXi database)
    """
    # find all files for loading (all with provided extension)
    files_to_load = glob.glob(os.path.join(path, "*." + file_extension))
    # load all of them
    labels = []
    for file_to_load in files_to_load:
        label = read_noxi_label_file(os.path.join(path, file_to_load))
        label = clean_labels(label)
        labels.append(label)
    # average ground truth labels (if there are several annotations of one video file)
    labels = average_from_several_labels(labels)
    # create from it pandas DataFrame
    labels = pd.DataFrame(data=labels)
    return labels


def load_all_labels_by_paths(paths: List[str]) -> Dict[str, pd.DataFrame]:
    """Loads labels from all paths provided in list. It will save it as a dict.
       Keys are paths, values are DataFrames with labels

    :param paths: List[str]
            List of paths for labels needed to be loaded
    :return: Dict[str, pd.DataFrame]
            Loaded labels in format of Dict, where key is path to the file with labels
            and value is pandas Dataframe of labels
    """
    labels = {}
    for path in paths:
        label = load_labels_by_path_to_dir(path)
        labels[path] = label
    return labels


def transform_all_labels_to_categorical(labels: Dict[str, pd.DataFrame], class_barriers: np.array) -> Dict[
    str, pd.DataFrame]:
    """Transforms all labels in provided dict to categorical ones.

    :param labels: Dict[str, pd.DataFrame]
            labels in format filename->pd.DataFrame. The values are real numbers.
    :param class_barriers: np.array
            numpy array with class barries. For example, class_barriers=[0.1, 0.5] will divide
            all labels into three classes - one from all less than 0.1, second from 0.1 to 0.5, and third everything more than 0.5
    :return: Dict[str, pd.DataFrame]
            labels in format filename->pd.DataFrame. The values are categorical.
    """
    for key in labels.keys():
        # save column names of the dataframe
        columns = labels[key].columns
        # transform labels. To do it, we need to extract numpy array from dataframe
        transformed_labels = transform_time_continuous_to_categorical(labels[key].values, class_barriers)
        # assign a new dataframe with transformed labels to the corresponding path
        labels[key] = pd.DataFrame(columns=columns, data=transformed_labels)
    return labels

def combine_dataframe_of_paths_with_labels_one_video(paths:pd.DataFrame, labels:pd.DataFrame, frames_step:int=5)->pd.DataFrame:
    """Combines paths to the images with corresponding labels. The sample rate is needed, since not all the images are taken.
       For example, in the paths dataframe can be only every fifth frame of the video file.

    :param paths: pd.DataFrame
            Dataframe with paths to the frames of one videofile
    :param labels: pd.DataFrame
            Dataframe with labels for every frame of the videofile
    :param labels_sr: int
            Sample rate of the labels.
    :param frames_step: int
            Step, with which frames (images) were taken from the video.
    :return: pd.DataFrame
            Combined pandas DataFrame with paths and corresponding to them labels.
    """
    # choose indices for labels based on the provided step of frames
    indices_for_labels=np.arrange(paths.shape[0], step=frames_step)
    labels=labels[indices_for_labels]
    # copy dataframe for consequent changing without influences on the existing dataframe
    result_df=copy.deepcopy(paths)
    # combination of chosen labels with filenames
    result_df['label']=labels
    return result_df

def combine_path_to_images_with_labels_many_videos(paths_with_images: pd.DataFrame, labels: Dict[str, pd.DataFrame],
                                       sample_rate_annotations: Optional[int] = 25, frame_step:int=5) -> pd.DataFrame:
    """Combines paths to video frames (images) with corresponding to them labels.
       At the end, the fuinction returns pandas DataFrame as follows:
                            filename                    class
                            path/to/frame.extension     1/2/3 etc.


    :param paths_with_images: pd.DataFrame
            Pandas DataFrame with paths to video frames (images)
    :param labels: Dict[str, pd.DataFrame]
            Labels in format [path_to_labelfile->values]
    :param sample_rate_annotations: int
            Sample rate of the labels
    :param frame_step: int
            Step of frames extracted from the video file.
            For example, step 5 means that we have taken 0, 5th, 10th, 15th, 20th frames.
    :return: pd.DataFrame
            Combined paths to frames with corresponding to them labels. All videos are processed.
    """
    # create result dataframe with needed columns
    result_dataframe=pd.DataFrame(columns=['filename', 'class'])
    for path_to_label in labels.keys():
        labels_one_video=labels[path_to_label]
        # TODO: CHECK IT. Check also ranking of filenames (order should be numerical, not lexical)
        # search paths to images in dataframe according to the path to the labels (for example, 031_2016-04-06_Nottingham\\expert)
        df_with_paths_one_video=paths_with_images[path_to_label in paths_with_images['rel_path']]
        # combine labels with corresponding paths to images
        df_paths_labels_one_video=combine_dataframe_of_paths_with_labels_one_video(paths_with_images, labels_one_video, frame_step)
        # change names of columns for further easier processing
        df_paths_labels_one_video.columns=['filename', 'class']
        # append obtained dataframe to the result dataframe (which contains all paths and labels)
        result_dataframe=result_dataframe.append(df_paths_labels_one_video, axis=0)

    return result_dataframe

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
