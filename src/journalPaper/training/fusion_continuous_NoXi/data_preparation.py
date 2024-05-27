import os
import sys
from typing import List, Dict, Callable, Tuple, Optional

import numpy as np
import pandas as pd
import torch

from src.journalPaper.training.fusion_continuous_NoXi.data_loader_multimodal import MultiModalTemporalDataLoader


def divide_df_into_sessions(df:pd.DataFrame):
    result ={}
    df['key_column'] = df.apply(lambda x: '/'.join(x['path'].split('/')[-3:-1]), axis=1)
    unique_keys = df['key_column'].unique()
    for key in unique_keys:
        result[key] = df[df['key_column'] == key]
    # delete the key_column
    df.drop(columns=['key_column'], inplace=True)
    for key in result.keys():
        result[key]= result[key].drop(columns=['key_column'])
    return result



def load_dataframes_3_modalities(face_path:str, pose_path:str, emo_path:str):
    face = pd.read_csv(face_path)
    pose = pd.read_csv(pose_path)
    emo = pd.read_csv(emo_path)
    # divide every dataframe into sessions
    face = divide_df_into_sessions(face)
    pose = divide_df_into_sessions(pose)
    emo = divide_df_into_sessions(emo)
    return face, pose, emo

def construct_data_loaders(train: List[Dict[str, pd.DataFrame]], dev: List[Dict[str, pd.DataFrame]],
                           window_size: float, stride: float, consider_timesteps: bool,
                           feature_columns: List[List[str]],
                           label_columns: List[str],
                           preprocessing_functions: List[Callable],
                           batch_size: int,
                           num_workers: int = 8) \
        -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:

    train_data_loader = MultiModalTemporalDataLoader(embeddings_with_labels=train,
                                          feature_columns=feature_columns,
                                          label_columns=label_columns,
                                          window_size=window_size,
                                          stride=stride,
                                          consider_timesteps=consider_timesteps,
                                          preprocessing_functions=preprocessing_functions,
                                          shuffle=False)
    dev_data_loader = MultiModalTemporalDataLoader(embeddings_with_labels=dev,
                                            feature_columns=feature_columns,
                                            label_columns=label_columns,
                                            window_size=window_size,
                                            stride=window_size,
                                            consider_timesteps=consider_timesteps,
                                            preprocessing_functions=preprocessing_functions,
                                            shuffle=False)

    # create torch data loaders
    train_data_loader = torch.utils.data.DataLoader(train_data_loader, batch_size=batch_size, shuffle=True,
                                                    num_workers=num_workers)
    dev_data_loader = torch.utils.data.DataLoader(dev_data_loader, batch_size=batch_size, shuffle=False,
                                                    num_workers=num_workers)

    return train_data_loader, dev_data_loader



def get_train_dev_dataloaders(paths_to_embeddings:Dict[str,str], window_size: float, stride: float, consider_timesteps: bool,
                              feature_columns: List[List[str]], label_columns: List[str],
                              preprocessing_functions: List[Callable], batch_size: int, num_workers: int = 8) \
        -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    # load data
    face_train, pose_train, emo_train = load_dataframes_3_modalities(paths_to_embeddings['face_train'],
                                                                        paths_to_embeddings['pose_train'],
                                                                        paths_to_embeddings['emo_train'])
    face_dev, pose_dev, emo_dev = load_dataframes_3_modalities(paths_to_embeddings['face_dev'],
                                                                paths_to_embeddings['pose_dev'],
                                                                paths_to_embeddings['emo_dev'])
    train = [face_train, pose_train, emo_train]
    dev = [face_dev, pose_dev, emo_dev]
    # make dataloaders
    train_loader, dev_loader = construct_data_loaders(train, dev,
                                  window_size, stride,
                                  consider_timesteps, feature_columns, label_columns,
                                preprocessing_functions, batch_size, num_workers)
    return train_loader, dev_loader