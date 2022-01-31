import copy
import glob
import os
from typing import List, Dict, Optional

import numpy as np
import pandas as pd


def transform_time_continuous_to_categorical(labels: np.ndarray, class_barriers: np.ndarray) -> np.ndarray:
    """Transforms time continuous labels to categorical labels using provided class barriers.
       For example, array [0.1, 0.5, 0.1, 0.25, 0.8] with the class barriers [0.3, 0.6] will be transformed to
                          [0,   1,   0,   0,    2]

    :param labels: np.ndarray
            labels for transformation
    :param class_barriers: np.ndarray
            Class barriers for transformation. Denotes the
    :return: np.ndarray
            Transformed labels in numpy array
    """
    # transform labels with masked operations
    transformed_labels = copy.deepcopy(labels)
    for num_barrier in range(0, class_barriers.shape[0] - 1):
        left_barr = class_barriers[num_barrier]
        right_barr = class_barriers[num_barrier + 1]
        mask = np.where((labels >= left_barr) &
                        (labels < right_barr))
        transformed_labels[mask] = num_barrier + 1

    # for the very left barrier we need to do it separately
    mask = labels < class_barriers[0]
    transformed_labels[mask] = 0
    # for the very right barrier we need to do it separately
    mask = labels >= class_barriers[-1]
    transformed_labels[mask] = class_barriers.shape[0]

    return transformed_labels


def read_noxi_label_file(path: str) -> np.ndarray:
    """Reads the label file of NoXi database
       It is initially a bite array (flow).

    :param path: str
            path to the file to read
    :return: np.ndarray
            read and saved to the numpy array file
    """
    # read it as a bite array with ASCII encoding
    with open(path, 'r', encoding='ASCII') as reader:
        annotation = reader.read()
    # convert byte array to numpy array
    annotation = np.genfromtxt(annotation.splitlines(), dtype=np.float32, delimiter=';')
    return annotation


def clean_labels(labels: np.ndarray) -> np.ndarray:
    """Cleans provided array. THis includes NaN cleaning.

    :param labels: np.ndarray
            labels to clean
    :return: np.ndarray
            cleaned labels
    """
    # remove nan values by the mean of array
    means = np.nanmean(labels, axis=0)
    print("means:", means)
    for mean_idx in range(means.shape[0]):
        labels[np.isnan(labels[:, mean_idx]), mean_idx] = means[mean_idx]

    return labels


def average_from_several_labels(labels_list: List[np.ndarray]) -> np.ndarray:
    """Averages labels from different sources (annotaters) into one labels file.
       In NoXi, aprat from the labels itself there are confidences of the labeling.
       We will use them as a weights for averaging.


    :param labels_list: List[np.ndarray]
            List of labels, which are needed to average
    :return: np.ndarray
            Averaged labels
    """
    # normalization of confidences. We recalculate weights to normalize them across all confidences.
    confidences_normalization_sum = sum(item[:, 1] for item in labels_list)
    normalized_confidences = [item[:, 1] / confidences_normalization_sum for item in labels_list]
    # preparation of variables
    labels = [item[:, 0] for item in labels_list]

    # Recalculation of resulted labels
    result_labels = sum(item1 * item2 for item1, item2 in zip(labels, normalized_confidences))
    # add dimension to make it 2-d array
    result_labels = result_labels[..., np.newaxis]
    return result_labels


def load_all_labels_by_paths(paths: List[str]) -> Dict[str, np.ndarray]:
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
        label = read_noxi_label_file(path)
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



if __name__ == '__main__':
    # arr=np.array([0.17, 0.478, 0.569, 0.445, 0.987])
    # class_barrier=np.array([0.5])
    # print(transform_time_continuous_to_categorical(arr, class_barrier))

    # path=r'C:\Users\Dresvyanskiy\Desktop\tmp\engagement_expert_sandra.annotation~'
    # print(read_noxi_label_file(path))

    # tmp=np.array([[0, 0], [1, 1], [2, np.nan], [np.nan, 3], [np.nan, np.nan]])
    # print(tmp)
    # print(clean_labels_array(tmp))

    labels_1 = np.array([[0.5, 1], [0.6, 1], [0.7, 0.8], [0.4, 0.5]])
    labels_2 = np.array([[0.6, 0.3], [0.7, 1], [0.9, 0.7], [0.8, 0.9]])
    print(average_from_several_labels([labels_1, labels_2]))