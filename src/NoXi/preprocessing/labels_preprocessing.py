import copy
from typing import List

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
