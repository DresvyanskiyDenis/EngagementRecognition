import copy
from typing import List

import numpy as np
import pandas as pd


def transform_time_continuous_to_categorical(labels:np.ndarray, class_barriers:np.ndarray)-> np.ndarray:
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
    transformed_labels=copy.deepcopy(labels)
    for num_barrier in range(0, class_barriers.shape[0]-1):
        left_barr=class_barriers[num_barrier]
        right_barr=class_barriers[num_barrier+1]
        mask= np.where((labels>=left_barr) &
                       (labels<right_barr))
        transformed_labels[mask] = num_barrier+1

    # for the very left barrier we need to do it separately
    mask = labels < class_barriers[0]
    transformed_labels[mask] = 0
    # for the very right barrier we need to do it separately
    mask=labels>=class_barriers[-1]
    transformed_labels[mask]=class_barriers.shape[0]

    return transformed_labels




if __name__ == '__main__':
    arr=np.array([0.17, 0.478, 0.569, 0.445, 0.987])
    class_barrier=np.array([0.5])
    print(transform_time_continuous_to_categorical(arr, class_barrier))