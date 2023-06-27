import os
import sys
from typing import List, Dict

import numpy as np
import pandas as pd



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


def load_embeddings(paths:Dict[str,str])->Dict[str, Dict[str, pd.DataFrame]]:
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









