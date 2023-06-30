import os
import sys
from typing import List, Dict, Callable, Tuple, Optional

import numpy as np
import pandas as pd
import torch

from pytorch_utils.data_loaders.TemporalEmbeddingsLoader import TemporalEmbeddingsLoader
from pytorch_utils.data_loaders.TemporalLoadersStacker import TemporalLoadersStacker


def merge_dicts(dict1, dict2):
    return dict(dict1, **dict2)


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
    tmp_df['video_name'] = tmp_df['path'].apply(lambda x: os.path.join(*x.split('/')[position_of_videoname:position_of_videoname+2]))
    # get unique video names
    video_names = tmp_df['video_name'].unique()
    # split data by video names. Small trick - we embrace the video_name with '/' to make sure that we get only
    # the video name and not the part of the path
    for video_name in video_names:
        result[video_name] = tmp_df[tmp_df['path'].str.contains(video_name)]
    return result


def load_embeddings(paths:Dict[str,str])->Dict[str, pd.DataFrame]:
    # paths is a list of dicts, where every dict contains the database name (DAiSEE_train, DAiSEE_dev,... or NoXi_train,...)
    # and the path to the embeddings
    # load embeddings from files
    embeddings = {}
    for database_name, path in paths.items():
        embeddings[database_name] = pd.read_csv(path)
    # split embeddings inside each dataframe on dataframes with frames from one video
    for database_name, df in embeddings.items():
        embeddings[database_name] = split_data_by_videoname(df, position_of_videoname=-3)

    return embeddings

def construct_data_loaders(train: List[Dict[str, pd.DataFrame]], dev: List[Dict[str, pd.DataFrame]],
                           test: List[Dict[str, pd.DataFrame]],
                           window_size: float, stride: float, consider_timestamps: bool,
                           label_columns: List[str],
                           preprocessing_functions: List[Callable],
                           batch_size: int,
                           num_workers: int = 8) \
        -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, torch.utils.data.DataLoader]:

    train_data_loader = TemporalLoadersStacker(embeddings_with_labels_list= train,
                                       label_columns=label_columns,
                 window_size=window_size, stride=stride,
                 consider_timestamps=consider_timestamps,
                 preprocessing_functions=preprocessing_functions, shuffle=False)

    dev_data_loader = TemporalLoadersStacker(embeddings_with_labels_list= dev,
                                        label_columns=label_columns,
                    window_size=window_size, stride=stride,
                    consider_timestamps=consider_timestamps,
                    preprocessing_functions=preprocessing_functions, shuffle=False)

    test_data_loader = TemporalLoadersStacker(embeddings_with_labels_list= test,
                                        label_columns=label_columns,
                    window_size=window_size, stride=stride,
                    consider_timestamps=consider_timestamps,
                    preprocessing_functions=preprocessing_functions, shuffle=False)

    # create torch data loaders
    train_data_loader = torch.utils.data.DataLoader(train_data_loader, batch_size=batch_size, shuffle=True,
                                                    num_workers=num_workers)
    dev_data_loader = torch.utils.data.DataLoader(dev_data_loader, batch_size=batch_size, shuffle=False,
                                                    num_workers=num_workers)
    test_data_loader = torch.utils.data.DataLoader(test_data_loader, batch_size=batch_size, shuffle=False,
                                                    num_workers=num_workers)

    return train_data_loader, dev_data_loader, test_data_loader


def get_train_dev_test(path_to_embeddings:str, embeddings_type:List[str]):
    # all embeddings will be loaded from the path_to_embeddings folder
    # list of embeddings there should be:
    # DAiSEE_emo_embeddings_train.csv, DAiSEE_emo_embeddings_dev.csv, DAiSEE_emo_embeddings_test.csv
    # DAiSEE_face_embeddings_train.csv, DAiSEE_face_embeddings_dev.csv, DAiSEE_face_embeddings_test.csv
    # DAiSEE_pose_embeddings_train.csv, DAiSEE_pose_embeddings_dev.csv, DAiSEE_pose_embeddings_test.csv
    # NoXi_emo_embeddings_train.csv, NoXi_emo_embeddings_dev.csv, NoXi_emo_embeddings_test.csv
    # NoXi_face_embeddings_train.csv, NoXi_face_embeddings_dev.csv, NoXi_face_embeddings_test.csv
    # NoXi_pose_embeddings_train.csv, NoXi_pose_embeddings_dev.csv, NoXi_pose_embeddings_test.csv
    # load embeddings for each database and unite them
    # emo
    train_emo = load_embeddings({'DAiSEE_emo_train': os.path.join(path_to_embeddings, 'DAiSEE_emo_embeddings_train.csv'),
                                    'NoXi_emo_train': os.path.join(path_to_embeddings, 'NoXi_emo_embeddings_train.csv')})
    train_emo = merge_dicts(train_emo['DAiSEE_emo_train'], train_emo['NoXi_emo_train'])
    dev_emo = load_embeddings({'DAiSEE_emo_dev': os.path.join(path_to_embeddings, 'DAiSEE_emo_embeddings_dev.csv'),
                                    'NoXi_emo_dev': os.path.join(path_to_embeddings, 'NoXi_emo_embeddings_dev.csv')})
    dev_emo = merge_dicts(dev_emo['DAiSEE_emo_dev'], dev_emo['NoXi_emo_dev'])
    test_emo = load_embeddings({'DAiSEE_emo_test': os.path.join(path_to_embeddings, 'DAiSEE_emo_embeddings_test.csv'),
                                    'NoXi_emo_test': os.path.join(path_to_embeddings, 'NoXi_emo_embeddings_test.csv')})
    test_emo = merge_dicts(test_emo['DAiSEE_emo_test'], test_emo['NoXi_emo_test'])
    # face
    train_face = load_embeddings({'DAiSEE_face_train': os.path.join(path_to_embeddings, 'DAiSEE_face_embeddings_train.csv'),
                                    'NoXi_face_train': os.path.join(path_to_embeddings, 'NoXi_face_embeddings_train.csv')})
    train_face = merge_dicts(train_face['DAiSEE_face_train'], train_face['NoXi_face_train'])
    dev_face = load_embeddings({'DAiSEE_face_dev': os.path.join(path_to_embeddings, 'DAiSEE_face_embeddings_dev.csv'),
                                    'NoXi_face_dev': os.path.join(path_to_embeddings, 'NoXi_face_embeddings_dev.csv')})
    dev_face = merge_dicts(dev_face['DAiSEE_face_dev'], dev_face['NoXi_face_dev'])
    test_face = load_embeddings({'DAiSEE_face_test': os.path.join(path_to_embeddings, 'DAiSEE_face_embeddings_test.csv'),
                                    'NoXi_face_test': os.path.join(path_to_embeddings, 'NoXi_face_embeddings_test.csv')})
    test_face = merge_dicts(test_face['DAiSEE_face_test'], test_face['NoXi_face_test'])
    # pose
    train_pose = load_embeddings({'DAiSEE_pose_train': os.path.join(path_to_embeddings, 'DAiSEE_pose_embeddings_train.csv'),
                                    'NoXi_pose_train': os.path.join(path_to_embeddings, 'NoXi_pose_embeddings_train.csv')})
    train_pose = merge_dicts(train_pose['DAiSEE_pose_train'], train_pose['NoXi_pose_train'])
    dev_pose = load_embeddings({'DAiSEE_pose_dev': os.path.join(path_to_embeddings, 'DAiSEE_pose_embeddings_dev.csv'),
                                    'NoXi_pose_dev': os.path.join(path_to_embeddings, 'NoXi_pose_embeddings_dev.csv')})
    dev_pose = merge_dicts(dev_pose['DAiSEE_pose_dev'], dev_pose['NoXi_pose_dev'])
    test_pose = load_embeddings({'DAiSEE_pose_test': os.path.join(path_to_embeddings, 'DAiSEE_pose_embeddings_test.csv'),
                                    'NoXi_pose_test': os.path.join(path_to_embeddings, 'NoXi_pose_embeddings_test.csv')})
    test_pose = merge_dicts(test_pose['DAiSEE_pose_test'], test_pose['NoXi_pose_test'])
    # choose what will insert to train, dev, test depending on embeddings type that have been provided
    train = []
    dev = []
    test = []
    for emb_type in embeddings_type:
        if emb_type == 'emo':
            train.append(train_emo)
            dev.append(dev_emo)
            test.append(test_emo)
        elif emb_type == 'face':
            train.append(train_face)
            dev.append(dev_face)
            test.append(test_face)
        elif emb_type == 'pose':
            train.append(train_pose)
            dev.append(dev_pose)
            test.append(test_pose)
        else:
            raise ValueError('Unknown embeddings type')


def calculate_class_weights(dataset:Dict[str, pd.DataFrame], label_columns:List[str])->torch.Tensor:
    all_labels = pd.concat([dataset[key] for key in dataset.keys()], axis=0)
    all_labels = all_labels.dropna()
    all_labels = np.array(all_labels[label_columns].values)
    num_classes = all_labels.shape[1]
    class_weights = all_labels.sum(axis=0)
    class_weights = 1. / (class_weights / class_weights.sum())
    # normalize class weights
    class_weights = class_weights / class_weights.sum()
    class_weights = torch.tensor(class_weights, dtype=torch.float32)
    return class_weights




















