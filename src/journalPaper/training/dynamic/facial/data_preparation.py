from functools import partial
from typing import Tuple, List, Callable, Optional, Dict, Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

import training_config
from decorators.common_decorators import timer
from pytorch_utils.data_loaders.ImageDataLoader_new import ImageDataLoader
from pytorch_utils.data_loaders.TemporalDataLoader import TemporalDataLoader
from pytorch_utils.data_loaders.pytorch_augmentations import pad_image_random_factor, grayscale_image, \
    collor_jitter_image_random, gaussian_blur_image_random, random_perspective_image, random_rotation_image, \
    random_crop_image, random_posterize_image, random_adjust_sharpness_image, random_equalize_image, \
    random_horizontal_flip_image, random_vertical_flip_image
from pytorch_utils.models.input_preprocessing import resize_image_saving_aspect_ratio, EfficientNet_image_preprocessor


def load_all_dataframes() -> Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]:
    """
     Loads all dataframes for the datasets AFEW-VA, AffectNet, RECOLA, SEMAINE, and SEWA, and split them into
        train, dev, and test sets.
    Returns: Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]
        The tuple of train, dev, and test data.
    """

    path_to_NoXi_train = "/nfs/home/ddresvya/Data/NoXi/prepared_data/faces/NoXi_face_train.csv"
    path_to_NoXi_dev = "/nfs/home/ddresvya/Data/NoXi/prepared_data/faces/NoXi_face_dev.csv"
    path_to_NoXi_test = "/nfs/home/ddresvya/Data/NoXi/prepared_data/faces/NoXi_face_test.csv"

    path_to_DAiSEE_train = "/nfs/home/ddresvya/Data/DAiSEE/prepared_data/faces/DAiSEE_face_train_labels.csv"
    path_to_DAiSEE_dev = "/nfs/home/ddresvya/Data/DAiSEE/prepared_data/faces/DAiSEE_face_dev_labels.csv"
    path_to_DAiSEE_test = "/nfs/home/ddresvya/Data/DAiSEE/prepared_data/faces/DAiSEE_face_test_labels.csv"

    # load dataframes
    NoXi_train = pd.read_csv(path_to_NoXi_train)
    NoXi_dev = pd.read_csv(path_to_NoXi_dev)
    NoXi_test = pd.read_csv(path_to_NoXi_test)

    DAiSEE_train = pd.read_csv(path_to_DAiSEE_train)
    DAiSEE_dev = pd.read_csv(path_to_DAiSEE_dev)
    DAiSEE_test = pd.read_csv(path_to_DAiSEE_test)

    # clear from NaNs
    NoXi_train = NoXi_train.dropna()
    NoXi_dev = NoXi_dev.dropna()
    NoXi_test = NoXi_test.dropna()

    DAiSEE_train = DAiSEE_train.dropna()
    DAiSEE_dev = DAiSEE_dev.dropna()
    DAiSEE_test = DAiSEE_test.dropna()

    # drop timestamps
    NoXi_train = NoXi_train.drop(columns=['timestep'])
    NoXi_dev = NoXi_dev.drop(columns=['timestep'])
    NoXi_test = NoXi_test.drop(columns=['timestep'])

    DAiSEE_train = DAiSEE_train.drop(columns=['timestamp'])
    DAiSEE_dev = DAiSEE_dev.drop(columns=['timestamp'])
    DAiSEE_test = DAiSEE_test.drop(columns=['timestamp'])

    # change path_to_frame column name to path
    NoXi_train = NoXi_train.rename(columns={"path_to_frame": "path"})
    NoXi_dev = NoXi_dev.rename(columns={"path_to_frame": "path"})
    NoXi_test = NoXi_test.rename(columns={"path_to_frame": "path"})

    DAiSEE_train = DAiSEE_train.rename(columns={"path_to_frame": "path"})
    DAiSEE_dev = DAiSEE_dev.rename(columns={"path_to_frame": "path"})
    DAiSEE_test = DAiSEE_test.rename(columns={"path_to_frame": "path"})

    # map 4-class labels to 3-class labels for DAiSEE. 4 classes were: highly disengaged, disengaged, engaged, highly engaged.
    # Now it would be 3 classes: disengaged, neutral, engaged
    # Remember that they are presented as one-hot vectors
    # we can simply add two columns that represent middle classes, while keep the other two columns as they are
    # then, drop old columns and rename new columns to the template 'label_0', 'label_1', 'label_2',
    DAiSEE_train['new_label_0'] = DAiSEE_train['label_0']
    DAiSEE_train['new_label_1'] = DAiSEE_train['label_1'] + DAiSEE_train['label_2']
    DAiSEE_train['new_label_2'] = DAiSEE_train['label_3']
    DAiSEE_train = DAiSEE_train.drop(columns=['label_0', 'label_1', 'label_2', 'label_3'])
    DAiSEE_train = DAiSEE_train.rename(
        columns={"new_label_0": "label_0", "new_label_1": "label_1", "new_label_2": "label_2"})

    DAiSEE_dev['new_label_0'] = DAiSEE_dev['label_0']
    DAiSEE_dev['new_label_1'] = DAiSEE_dev['label_1'] + DAiSEE_dev['label_2']
    DAiSEE_dev['new_label_2'] = DAiSEE_dev['label_3']
    DAiSEE_dev = DAiSEE_dev.drop(columns=['label_0', 'label_1', 'label_2', 'label_3'])
    DAiSEE_dev = DAiSEE_dev.rename(
        columns={"new_label_0": "label_0", "new_label_1": "label_1", "new_label_2": "label_2"})

    DAiSEE_test['new_label_0'] = DAiSEE_test['label_0']
    DAiSEE_test['new_label_1'] = DAiSEE_test['label_1'] + DAiSEE_test['label_2']
    DAiSEE_test['new_label_2'] = DAiSEE_test['label_3']
    DAiSEE_test = DAiSEE_test.drop(columns=['label_0', 'label_1', 'label_2', 'label_3'])
    DAiSEE_test = DAiSEE_test.rename(
        columns={"new_label_0": "label_0", "new_label_1": "label_1", "new_label_2": "label_2"})

    # map 5-class labels to 3-class labels for NoXi dataset. 5 classes were: highly disengaged, disengaged, neutral, engaged, highly engaged.
    # Now it would be 3 classes: disengaged, neutral, engaged
    # Remember that they are presented as one-hot vectors
    # we can add two first columns that represent disengagemend state (it would be then disengagement class in 3-class classification)
    # then, add two last columns that represent engagement state (it would be then engagement class in 3-class classification)
    # then, drop old columns and rename new columns to the template 'label_0', 'label_1', 'label_2'
    NoXi_train['new_label_0'] = NoXi_train['label_0'] + NoXi_train['label_1']
    NoXi_train['new_label_1'] = NoXi_train['label_2']
    NoXi_train['new_label_2'] = NoXi_train['label_3'] + NoXi_train['label_4']
    NoXi_train = NoXi_train.drop(columns=['label_0', 'label_1', 'label_2', 'label_3', 'label_4'])
    NoXi_train = NoXi_train.rename(
        columns={"new_label_0": "label_0", "new_label_1": "label_1", "new_label_2": "label_2"})

    NoXi_dev['new_label_0'] = NoXi_dev['label_0'] + NoXi_dev['label_1']
    NoXi_dev['new_label_1'] = NoXi_dev['label_2']
    NoXi_dev['new_label_2'] = NoXi_dev['label_3'] + NoXi_dev['label_4']
    NoXi_dev = NoXi_dev.drop(columns=['label_0', 'label_1', 'label_2', 'label_3', 'label_4'])
    NoXi_dev = NoXi_dev.rename(columns={"new_label_0": "label_0", "new_label_1": "label_1", "new_label_2": "label_2"})

    NoXi_test['new_label_0'] = NoXi_test['label_0'] + NoXi_test['label_1']
    NoXi_test['new_label_1'] = NoXi_test['label_2']
    NoXi_test['new_label_2'] = NoXi_test['label_3'] + NoXi_test['label_4']
    NoXi_test = NoXi_test.drop(columns=['label_0', 'label_1', 'label_2', 'label_3', 'label_4'])
    NoXi_test = NoXi_test.rename(columns={"new_label_0": "label_0", "new_label_1": "label_1", "new_label_2": "label_2"})

    # transform dataframes to Dict[str, pd.DataFrame] type, where keys are video names, and values are dataframes with
    # paths and labels for each frame
    NoXi_train = split_data_by_videoname(NoXi_train, position_of_videoname=-3)
    NoXi_dev = split_data_by_videoname(NoXi_dev, position_of_videoname=-3)
    NoXi_test = split_data_by_videoname(NoXi_test, position_of_videoname=-3)
    DAiSEE_train = split_data_by_videoname(DAiSEE_train, position_of_videoname=-3)
    DAiSEE_dev = split_data_by_videoname(DAiSEE_dev, position_of_videoname=-3)
    DAiSEE_test = split_data_by_videoname(DAiSEE_test, position_of_videoname=-3)
    # change paths from 'media/external_hdd_2/' to '/work/home/dsu/Datasets/'
    NoXi_train = {k: v.replace('media/external_hdd_2/', '/nfs/home/ddresvya/Data/') for k, v in NoXi_train.items()}
    NoXi_dev = {k: v.replace('media/external_hdd_2/', '/nfs/home/ddresvya/Data/') for k, v in NoXi_dev.items()}
    NoXi_test = {k: v.replace('media/external_hdd_2/', '/nfs/home/ddresvya/Data/') for k, v in NoXi_test.items()}
    DAiSEE_train = {k: v.replace('media/external_hdd_2/', '/nfs/home/ddresvya/Data/') for k, v in DAiSEE_train.items()}
    DAiSEE_dev = {k: v.replace('media/external_hdd_2/', '/nfs/home/ddresvya/Data/') for k, v in DAiSEE_dev.items()}
    DAiSEE_test = {k: v.replace('media/external_hdd_2/', '/nfs/home/ddresvya/Data/') for k, v in DAiSEE_test.items()}
    # concatenate datasets
    train = {**NoXi_train, **DAiSEE_train}
    dev = {**NoXi_dev, **DAiSEE_dev}
    test = {**NoXi_test, **DAiSEE_test}

    return train, dev, test

def split_data_by_videoname(df:pd.DataFrame, position_of_videoname:int)->Dict[str, pd.DataFrame]:
    """ Splits data represented in dataframes by video names.
    The provided data is represented as one big pd.DataFrame. The video names are stored in 'path' column,
    The function separates the data by video names and returns a dictionary where keys are video names and values are
    pd.DataFrames with data for each video.

    Args:
        df: pd.DataFrame
            The data to be separated by video names.
        position_of_videoname: int
            The position of video name in the 'path' column. For example, if the 'path' column contains
            '/work/DAiSEE/5993322/DAiSEE_train_0001.mp4', then the position of video name is -2.


    :return: Dict[str, pd.DataFrame]
        A dictionary where keys are video names and values are pd.DataFrames with data for each video.
    """
    if position_of_videoname >= 0:
        raise ValueError('The position of video name in the path column must be negative.')
    result = {}
    # create additional columns with names of video
    tmp_df = df.copy(deep=True)
    tmp_df['video_name'] = tmp_df['path'].apply(lambda x: x.split('/')[position_of_videoname])
    # get unique video names
    video_names = tmp_df['video_name'].unique()
    # split data by video names. Small trick - we embrace the video_name with '/' to make sure that we get only
    # the video name and not the part of the path
    for video_name in video_names:
        result[video_name] = tmp_df[tmp_df['path'].str.contains('/' + video_name + '/')]
    return result




def get_augmentation_function(probability: float) -> Dict[Callable, float]:
    """
    Returns a dictionary of augmentation functions and the probabilities of their application.
    Args:
        probability: float
            The probability of applying the augmentation function.

    Returns: Dict[Callable, float]
        A dictionary of augmentation functions and the probabilities of their application.

    """
    augmentation_functions = {
        pad_image_random_factor: probability,
        grayscale_image: probability,
        partial(collor_jitter_image_random, brightness=0.5, hue=0.3, contrast=0.3,
                saturation=0.3): probability,
        partial(gaussian_blur_image_random, kernel_size=(5, 9), sigma=(0.1, 5)): training_config.AUGMENT_PROB,
        random_perspective_image: probability,
        random_rotation_image: probability,
        partial(random_crop_image, cropping_factor_limits=(0.7, 0.9)): probability,
        random_posterize_image: probability,
        partial(random_adjust_sharpness_image, sharpness_factor_limits=(0.1, 3)): probability,
        random_equalize_image: probability,
        random_horizontal_flip_image: probability,
        random_vertical_flip_image: probability,
    }
    return augmentation_functions


def construct_data_loaders(train: Dict[str, pd.DataFrame], dev: Dict[str, pd.DataFrame],
                           window_size: float, stride: float, consider_timestamps: bool,
                           label_columns: List[str],
                           preprocessing_functions: List[Callable],
                           batch_size: int,
                           augmentation_functions: Optional[Dict[Callable, float]] = None,
                           num_workers: int = 8) \
        -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    train_data_loader = TemporalDataLoader(paths_with_labels=train, label_columns=label_columns,
                                           window_size=window_size, stride=stride,
                                           consider_timestamps=consider_timestamps,
                                           preprocessing_functions=preprocessing_functions,
                                           augmentation_functions=augmentation_functions,
                                           shuffle=False)

    dev_data_loader = TemporalDataLoader(paths_with_labels=dev, label_columns=label_columns,
                                         window_size=window_size, stride=stride,
                                         consider_timestamps=consider_timestamps,
                                         preprocessing_functions=preprocessing_functions,
                                         augmentation_functions=augmentation_functions,
                                         shuffle=False)

    train_dataloader = DataLoader(train_data_loader, batch_size=batch_size, num_workers=num_workers, drop_last=True,
                                  shuffle=True)
    dev_dataloader = DataLoader(dev_data_loader, batch_size=batch_size, num_workers=num_workers // 2, shuffle=False)

    return (train_dataloader, dev_dataloader)


def load_data_and_construct_dataloaders(model_type: str, batch_size: int,
                                        window_size: float, stride: float, consider_timestamps: bool,
                                        return_class_weights: Optional[bool] = False) -> \
        Union[Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader],
              Tuple[Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader], torch.Tensor]]:
    if model_type not in ['EfficientNet-B1', 'EfficientNet-B4']:
        raise ValueError('The model type should be either "EfficientNet-B1" or "EfficientNet-B4".')
    # load data. The data is represented as Dict[str, pd.DataFrame] where keys are video names and values are
    # pd.DataFrames with data for each video
    train, dev, test = load_all_dataframes()
    # define preprocessing functions
    if model_type == 'EfficientNet-B1':
        preprocessing_functions = [partial(resize_image_saving_aspect_ratio, expected_size=240),
                                   EfficientNet_image_preprocessor()]
    elif model_type == 'EfficientNet-B4':
        preprocessing_functions = [partial(resize_image_saving_aspect_ratio, expected_size=380),
                                   EfficientNet_image_preprocessor()]
    else:
        raise ValueError(f'The model type should be either "EfficientNet-B1" or "EfficientNet-B4".'
                         f'Got {model_type} instead.')
    # define augmentation functions
    augmentation_functions = get_augmentation_function(training_config.AUGMENT_PROB)
    # construct data loaders
    train_dataloader, dev_dataloader = construct_data_loaders(train, dev, window_size, stride, consider_timestamps,
                                                              preprocessing_functions=preprocessing_functions,
                                                              batch_size=batch_size,
                                                              augmentation_functions=augmentation_functions,
                                                              num_workers=training_config.NUM_WORKERS,
                                                              label_columns=training_config.LABEL_COLUMNS)

    if return_class_weights:
        # get all classes from Dict[str, pd.DataFrame] and calculate class weights
        all_labels = pd.concat([value for key, value in train.items()], axis=0)
        all_labels = all_labels.dropna()
        all_labels = np.array(all_labels[training_config.LABEL_COLUMNS].values)
        num_classes = all_labels.shape[1]
        class_weights = all_labels.sum(axis=0)
        class_weights = 1. / (class_weights / class_weights.sum())
        # normalize class weights
        class_weights = class_weights / class_weights.sum()
        class_weights = torch.tensor(class_weights.values, dtype=torch.float32)
        return ((train_dataloader, dev_dataloader), class_weights)

    return (train_dataloader, dev_dataloader)


@timer
def main():
    train_data_loader, dev_data_loader, test_data_loader = load_data_and_construct_dataloaders()
    for x, y in train_data_loader:
        print(x.shape, y.shape)
        print("-------------------")


if __name__ == "__main__":
    main()
