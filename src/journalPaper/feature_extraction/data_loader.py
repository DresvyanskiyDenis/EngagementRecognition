import os
from functools import partial
from typing import Tuple, List, Callable, Optional, Dict, Union

import pandas as pd
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as T

from pytorch_utils.data_loaders.ImageDataLoader_new import ImageDataLoader
from pytorch_utils.data_loaders.pytorch_augmentations import pad_image_random_factor, grayscale_image, \
    collor_jitter_image_random, gaussian_blur_image_random, random_perspective_image, random_rotation_image, \
    random_crop_image, random_posterize_image, random_adjust_sharpness_image, random_equalize_image, \
    random_horizontal_flip_image, random_vertical_flip_image
from pytorch_utils.data_preprocessing import convert_image_to_float_and_scale
from pytorch_utils.models.input_preprocessing import resize_image_saving_aspect_ratio, EfficientNet_image_preprocessor


def load_NoXi_and_DAiSEE_dataframes(path_to_data_NoXi: str, path_to_data_DAiSEE: str, data_type:str) -> Tuple[
    pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """ Loads train, dev, and test dataframes from the given paths. This function works only for the DAiSEE or NoXi datasets.
    The dataframes are stored in the .csv files and should have names:
    NoXi: [NoXi_facial_train.csv, NoXi_facial_dev.csv, NoXi_facial_test.csv]
    DAiSEE: [DAiSEE_facial_train_labels.csv, DAiSEE_facial_dev_labels.csv, DAiSEE_facial_test_labels.csv]

    Final train, dev, and test datasets will be concatenated from the dataframes of the same type (NoXi or DAiSEE).
    The final format of the dataframes is the following: [path, label_0, label_1, label_2]

    :param path_to_data_NoXi: str
        The path to the folder with the dataframes for NoXi dataset.
    :param path_to_data_DAiSEE: str
        The path to the folder with the dataframes for DAiSEE dataset.
    :param data_type: str
        The type of the data in datasets. It can be either 'face' or 'pose'.

    Returns: Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]
        The tuple of train, dev, and test data.

    """



    path_to_NoXi_train = os.path.join(path_to_data_NoXi, f'NoXi_{data_type}_train.csv')
    path_to_NoXi_dev = os.path.join(path_to_data_NoXi, f'NoXi_{data_type}_dev.csv')
    path_to_NoXi_test = os.path.join(path_to_data_NoXi, f'NoXi_{data_type}_test.csv')

    path_to_DAiSEE_train = os.path.join(path_to_data_DAiSEE, f'DAiSEE_{data_type}_train_labels.csv')
    path_to_DAiSEE_dev = os.path.join(path_to_data_DAiSEE, f'DAiSEE_{data_type}_dev_labels.csv')
    path_to_DAiSEE_test = os.path.join(path_to_data_DAiSEE, f'DAiSEE_{data_type}_test_labels.csv')

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

    # concatenate datasets
    train = pd.concat([DAiSEE_train, NoXi_train], ignore_index=True)
    dev = pd.concat([DAiSEE_dev, NoXi_dev], ignore_index=True)
    test = pd.concat([DAiSEE_test, NoXi_test], ignore_index=True)

    # change paths from 'media/external_hdd_2/*/prepared_data/{data_type}s' (example: media/external_hdd_2/NoXi/prepared_data/poses)
    # to provided data paths
    # NoXi
    if data_type == 'face':
        train['path'] = train['path'].apply(
            lambda x: x.replace(f'media/external_hdd_2/NoXi/prepared_data/{data_type}s',
                                path_to_data_NoXi))
        dev['path'] = dev['path'].apply(
            lambda x: x.replace(f'media/external_hdd_2/NoXi/prepared_data/{data_type}s',
                                path_to_data_NoXi))
        test['path'] = test['path'].apply(
            lambda x: x.replace(f'media/external_hdd_2/NoXi/prepared_data/{data_type}s',
                                path_to_data_NoXi))
        # DAiSEE
        train['path'] = train['path'].apply(
            lambda x: x.replace(f'media/external_hdd_2/DAiSEE/prepared_data/{data_type}s',
                                path_to_data_DAiSEE))
        dev['path'] = dev['path'].apply(
            lambda x: x.replace(f'media/external_hdd_2/DAiSEE/prepared_data/{data_type}s',
                                path_to_data_DAiSEE))
        test['path'] = test['path'].apply(
            lambda x: x.replace(f'media/external_hdd_2/DAiSEE/prepared_data/{data_type}s',
                                path_to_data_DAiSEE))
    elif data_type == 'pose':
    # there also can be another data paths, as we have re-extracted frames and directly saved them into the ssd disk
    # therefore, repeat the same procedure, but with '/work/home/dsu/Datasets/*/prepared_data/{data_type}s'
    # NoXi
        train['path'] = train['path'].apply(
            lambda x: x.replace(f'/work/home/dsu/Datasets/NoXi/prepared_data/{data_type}s',
                                path_to_data_NoXi))
        dev['path'] = dev['path'].apply(
            lambda x: x.replace(f'/work/home/dsu/Datasets/NoXi/prepared_data/{data_type}s',
                                path_to_data_NoXi))
        test['path'] = test['path'].apply(
            lambda x: x.replace(f'/work/home/dsu/Datasets/NoXi/prepared_data/{data_type}s',
                                path_to_data_NoXi))
        # DAiSEE
        train['path'] = train['path'].apply(
            lambda x: x.replace(f'/work/home/dsu/Datasets/DAiSEE/prepared_data/{data_type}s',
                                path_to_data_DAiSEE))
        dev['path'] = dev['path'].apply(
            lambda x: x.replace(f'/work/home/dsu/Datasets/DAiSEE/prepared_data/{data_type}s',
                                path_to_data_DAiSEE))
        test['path'] = test['path'].apply(
            lambda x: x.replace(f'/work/home/dsu/Datasets/DAiSEE/prepared_data/{data_type}s',
                                path_to_data_DAiSEE))
    else:
        raise ValueError(f'Wrong data type: {data_type}')

    return train, dev, test


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


def construct_data_loaders(train: pd.DataFrame, dev: pd.DataFrame, test: pd.DataFrame,
                           preprocessing_functions: List[Callable],
                           batch_size: int,
                           augmentation_functions: Optional[Dict[Callable, float]] = None,
                           num_workers: int = 8) \
        -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """ Constructs the data loaders for the train, dev and test sets.

    Args:
        train: pd.DataFrame
            The train set. It should contain the columns 'path'
        dev: pd.DataFrame
            The dev set. It should contain the columns 'path'
        test: pd.DataFrame
            The test set. It should contain the columns 'path'
        preprocessing_functions: List[Callable]
            A list of preprocessing functions to be applied to the images.
        batch_size: int
            The batch size.
        augmentation_functions: Optional[Dict[Callable, float]]
            A dictionary of augmentation functions and the probabilities of their application.
        num_workers: int
            The number of workers to be used by the data loaders.

    Returns: Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, torch.utils.data.DataLoader]
        The data loaders for the train, dev and test sets.

    """

    train_data_loader = ImageDataLoader(paths_with_labels=train, preprocessing_functions=preprocessing_functions,
                                        augmentation_functions=augmentation_functions, shuffle=True)

    dev_data_loader = ImageDataLoader(paths_with_labels=dev, preprocessing_functions=preprocessing_functions,
                                      augmentation_functions=None, shuffle=False)

    test_data_loader = ImageDataLoader(paths_with_labels=test, preprocessing_functions=preprocessing_functions,
                                       augmentation_functions=None, shuffle=False)

    train_dataloader = DataLoader(train_data_loader, batch_size=batch_size, num_workers=num_workers, drop_last=True,
                                  shuffle=True)
    dev_dataloader = DataLoader(dev_data_loader, batch_size=batch_size, num_workers=num_workers // 2, shuffle=False)
    test_dataloader = DataLoader(test_data_loader, batch_size=batch_size, num_workers=num_workers // 2, shuffle=False)

    return (train_dataloader, dev_dataloader, test_dataloader)


def load_data_and_construct_dataloaders(path_to_data_NoXi: str, path_to_data_DAiSEE: str, model_type: str,
                                        batch_size: int,
                                        data_type:str,
                                        return_class_weights: Optional[bool] = False,
                                        ) -> \
        Union[Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, torch.utils.data.DataLoader],
              Tuple[Tuple[
                        torch.utils.data.DataLoader, torch.utils.data.DataLoader, torch.utils.data.DataLoader], torch.Tensor]]:
    """
    Loads the data presented in pd.DataFrames and constructs the data loaders using them. It is a general function
    to assemble all functions defined above.
    Returns: Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, torch.utils.data.DataLoader]
        The train, dev and test data loaders.
        or
        Tuple[Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, torch.utils.data.DataLoader], torch.Tensor]
        The train, dev and test data loaders and the class weights calculated based on the training labels.

    :param path_to_data_NoXi: str
        The path to the data folder containing the NoXi data.
    :param path_to_data_DAiSEE: str
        The path to the data folder containing the DAiSEE data.
    :param model_type: str
        The type of the model to be used. It should be either 'EfficientNet-B1', 'EfficientNet-B4' or 'Modified_HRNet'.
    :param batch_size: int
        The batch size.
    :param return_class_weights: Optional[bool]
        Whether to return the class weights or not.
    :param data_type: str
        The type of data to be used. It should be either 'face' or 'pose'.

    """
    if model_type not in ['EfficientNet-B1', 'EfficientNet-B4', 'Modified_HRNet']:
        raise ValueError('The model type should be either "EfficientNet-B1", "EfficientNet-B4" or "Modified_HRNet".')
    # load pd.DataFrames
    train, dev, test = load_NoXi_and_DAiSEE_dataframes(path_to_data_NoXi, path_to_data_DAiSEE, data_type=data_type)
    # define preprocessing functions
    if model_type == 'EfficientNet-B1':
        preprocessing_functions = [partial(resize_image_saving_aspect_ratio, expected_size=240),
                                   EfficientNet_image_preprocessor()]
    elif model_type == 'EfficientNet-B4':
        preprocessing_functions = [partial(resize_image_saving_aspect_ratio, expected_size=380),
                                   EfficientNet_image_preprocessor()]
    elif model_type == 'Modified_HRNet':
        preprocessing_functions = [partial(resize_image_saving_aspect_ratio, expected_size=256),
                                   convert_image_to_float_and_scale,
                                   T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                   ]  # From HRNet
    else:
        raise ValueError(f'The model type should be either "EfficientNet-B1", "EfficientNet-B4", or "Modified_HRNet".'
                         f'Got {model_type} instead.')
    # define augmentation functions
    augmentation_functions = get_augmentation_function(training_config.AUGMENT_PROB)
    # construct data loaders
    train_dataloader, dev_dataloader, test_dataloader = construct_data_loaders(train, dev, test,
                                                                               preprocessing_functions,
                                                                               batch_size,
                                                                               augmentation_functions,
                                                                               num_workers=training_config.NUM_WORKERS)

    if return_class_weights:
        num_classes = train.iloc[:, 1:].shape[1]
        labels = train.iloc[:, 1:]
        labels = labels.dropna()
        class_weights = labels.sum(axis=0)
        class_weights = 1. / (class_weights / class_weights.sum())
        # normalize class weights
        class_weights = class_weights / class_weights.sum()
        class_weights = torch.tensor(class_weights.values, dtype=torch.float32)
        return ((train_dataloader, dev_dataloader, test_dataloader), class_weights)

    return (train_dataloader, dev_dataloader, test_dataloader)
