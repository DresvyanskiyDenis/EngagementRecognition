from typing import Dict, Optional

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow_utils.tensorflow_datagenerators.sequence_loader_tf2 import Tensorflow_Callable, \
    get_tensorflow_sequence_loader


def load_embeddings_from_csv_file(embedding_file_path:str)->pd.DataFrame:
    """
    Loads embeddings from a csv file. They should be saved in the csv file with the columns in format:
    video_filename, frame_id, embedding_1, embedding_2, ..., label_0, label_1, ... (where labels are in the one-hot encoding form)
    :param embedding_file_path: str
            path to the csv file containing the embeddings
    :return: pd.DataFrame
            pandas dataframe containing the embeddings
    """
    embeddings = pd.read_csv(embedding_file_path)
    return embeddings

def split_embeddings_according_to_file_path(embeddings:pd.DataFrame)->Dict[str, pd.DataFrame]:
    """
    Splits the embeddings according to the file path in the first column.
    :param embeddings: pd.DataFrame
            pandas dataframe containing the embeddings
    :return: Dict[str, pd.DataFrame]
            dictionary containing the embeddings split by file path
    """
    # split the filename to two columns: video_filename and frame_id
    embeddings["frame_id"] = embeddings["filename"].apply(lambda x: x.split("/")[-1].split(".")[0].split("_")[-1])
    embeddings["frame_id"] = embeddings["frame_id"].astype('int32')
    embeddings.rename(columns={"filename": "video_filename"}, inplace=True)
    embeddings["video_filename"] = embeddings["video_filename"].apply(lambda x: x[:x.rfind("/")])
    # rearrange columns so that video_filename and the frame_id will be the first two columns
    embeddings = embeddings[["video_filename", "frame_id"] + embeddings.columns[2:-1].tolist()]

    split_embeddings={}
    for file_path in embeddings['video_filename'].unique():
        split_embeddings[file_path] = embeddings[embeddings['video_filename']==file_path]
    return split_embeddings

def create_generator_from_pd_dictionary(embeddings_dict:Dict[str, pd.DataFrame], num_classes:int, type_of_labels: str,
                                        window_length:int, window_shift:int, window_stride:int=1,
                                        batch_size:int=8, shuffle:bool=False,
                                        preprocessing_function: Optional[Tensorflow_Callable] = None,
                                        clip_values: Optional[bool] = None,
                                        cache_loaded_seq: Optional[bool] = None
                                        )->tf.data.Dataset:
    """

    :param embeddings_dict:
    :param num_classes:
    :param type_of_labels:
    :param window_length:
    :param window_shift:
    :param window_stride:
    :param batch_size:
    :param shuffle:
    :param preprocessing_function:
    :param clip_values:
    :param cache_loaded_seq:
    :return:
    """
    # go through the dictionary with dataframes, take them, and create a generator from it
    dataset_list = []
    for video_filename in embeddings_dict:
        embeddings_df = embeddings_dict[video_filename]
        embeddings_df = embeddings_df.drop(columns=['video_filename', 'frame_id'])
        tf_dataset_generator=get_tensorflow_sequence_loader(embeddings_and_labels=embeddings_df, num_classes=num_classes,
                                   batch_size=batch_size, type_of_labels=type_of_labels,
                                   window_size=window_length, window_shift=window_shift, window_stride=window_stride,
                                   shuffle=shuffle,
                             preprocessing_function=preprocessing_function,
                             clip_values=clip_values,
                             cache_loaded_seq=cache_loaded_seq)
        dataset_list.append(tf_dataset_generator)
    # concatenate all the generators to get only one
    result_dataset= dataset_list.pop(0)
    for dataset in dataset_list:
        result_dataset = result_dataset.concatenate(dataset)
    return result_dataset


def load_data():
    path_to_train_embeddings="/work/home/dsu/NoXi_embeddings/All_languages/train_extracted_deep_embeddings.csv"
    path_to_dev_embeddings="/work/home/dsu/NoXi_embeddings/All_languages/dev_extracted_deep_embeddings.csv"
    path_to_test_embeddings="/work/home/dsu/NoXi_embeddings/All_languages/test_extracted_deep_embeddings.csv"
    # load embeddings
    train_embeddings = load_embeddings_from_csv_file(path_to_train_embeddings)
    dev_embeddings = load_embeddings_from_csv_file(path_to_dev_embeddings)
    test_embeddings = load_embeddings_from_csv_file(path_to_test_embeddings)
    # split embeddings
    train_embeddings_split = split_embeddings_according_to_file_path(train_embeddings)
    dev_embeddings_split = split_embeddings_according_to_file_path(dev_embeddings)
    test_embeddings_split = split_embeddings_according_to_file_path(test_embeddings)

    return train_embeddings_split, dev_embeddings_split, test_embeddings_split


if __name__ == '__main__':
    pass