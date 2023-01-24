#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Contains the functions to make a tf.Dataset generator for sequence data.
"""

__author__ = "Denis Dresvyanskiy"
__copyright__ = "Copyright 2022"
__credits__ = ["Denis Dresvyanskiy"]
__maintainer__ = "Denis Dresvyanskiy"
__email__ = "denis.dresvyanskiy@uni-ulm.de"


from typing import Dict, Optional

import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

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
    embeddings = embeddings[["video_filename", "frame_id"] + embeddings.columns[1:-1].tolist()]

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
    """Creates a tensorflow.Dataset data generator. To do so, we need to pass the embeddings_dict, which will be preprocessed.
    The generator generates the sequences with specified window_length, window_shift, and window_stride from provided
    embedding vectors (usually extracted from every frame of some videofiles). THis is a dicrionary type, because every
    videofile is separated from each other to keep it consistent for windows forming.
    The format of such dictionaries are: Dict[str, pd.DataFrame] (Dict[path_to_video_file->pd.DataFrame])
    The DataFrame has columns: [video_filename, frame_id, embedding_0, embedding_1, ..., label_0, label_1, ...]


    :param embeddings_dict: Dict[str, pd.DataFrame]
                The dictionary in the format Dict[path_to_video_file->pd.DataFrame]
                The DataFrame has columns: [video_filename, frame_id, embedding_0, embedding_1, ..., label_0, label_1, ...]
                This is the dataset you want to convert into tensorflow generator.
    :param num_classes: int
            The number of classes in the classification task to make a final softmax layer with exact same
            number of neurons
    :param type_of_labels: str
            The type of labels to generate for every window. Two options are possible: ["sequence_to_one", "sequence_to_sequence"]
    :param window_length: int
            The length of the sequences (windows). THis is to generate the sequences of fixed length (for RNN model).
            The length should be specified in terms of frame numbers (not seconds)
    :param window_shift: int
            The length of window shift during data generation.
    :param window_stride: int
            THe stride of the window for generating data.
    :param batch_size: int
            The batch size for training the model.
    :param shuffle: bool
            Specifies, does the training data need to be shuffled before training or not.
    :param preprocessing_function: Optional[Tensorflow_Callable]
            Specifies, whether we need to apply preprocessing function before passing the data to the model or not.
    :param clip_values: bool
            Specifies, whether we need to clip the values of obtained data to [0, 1] before passing the data to model or not.
    :param cache_loaded_seq: bool
            Specifies, whether we need to cache the data or not. It makes the training be faster, but requires a lot of RAM.
    :return: tf.data.Dataset
            The sequence data generator.
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
    print('Start loading data...')
    path_to_train_embeddings="/work/home/dsu/IWSDS2023/NoXi_embeddings/All_languages/Xception_model/train_extracted_deep_embeddings.csv"
    path_to_dev_embeddings="/work/home/dsu/IWSDS2023/NoXi_embeddings/All_languages/Xception_model/dev_extracted_deep_embeddings.csv"
    path_to_test_embeddings="/work/home/dsu/IWSDS2023/NoXi_embeddings/All_languages/Xception_model/test_extracted_deep_embeddings.csv"
    # load embeddings
    train_embeddings = load_embeddings_from_csv_file(path_to_train_embeddings)
    dev_embeddings = load_embeddings_from_csv_file(path_to_dev_embeddings)
    test_embeddings = load_embeddings_from_csv_file(path_to_test_embeddings)
    # normalization
    scaler = StandardScaler()
    scaler=scaler.fit(train_embeddings.iloc[:, 1:-5])
    train_embeddings.iloc[:, 1:-5] = scaler.transform(train_embeddings.iloc[:, 1:-5])
    dev_embeddings.iloc[:, 1:-5] = scaler.transform(dev_embeddings.iloc[:, 1:-5])
    test_embeddings.iloc[:, 1:-5] = scaler.transform(test_embeddings.iloc[:, 1:-5])


    # split embeddings
    train_embeddings_split = split_embeddings_according_to_file_path(train_embeddings)
    dev_embeddings_split = split_embeddings_according_to_file_path(dev_embeddings)
    test_embeddings_split = split_embeddings_according_to_file_path(test_embeddings)

    return train_embeddings_split, dev_embeddings_split, test_embeddings_split

def load_data_cross_corpus(test_corpus:str):
    if not test_corpus in ('english', 'german', 'french'):
        raise ValueError('The test corpus should be one of the following: english, german, french')
    print('Start loading data...')
    # paths to the data
    path_to_train_embeddings_english = "/work/home/dsu/IWSDS2023/NoXi_embeddings/Cross-corpus/English/train_embeddings.csv"
    path_to_dev_embeddings_english = "/work/home/dsu/IWSDS2023/NoXi_embeddings/Cross-corpus/English/dev_embeddings.csv"
    path_to_test_embeddings_english = "/work/home/dsu/IWSDS2023/NoXi_embeddings/Cross-corpus/English/test_embeddings.csv"

    path_to_train_embeddings_french = "/work/home/dsu/IWSDS2023/NoXi_embeddings/Cross-corpus/French/train_embeddings.csv"
    path_to_dev_embeddings_french = "/work/home/dsu/IWSDS2023/NoXi_embeddings/Cross-corpus/French/dev_embeddings.csv"
    path_to_test_embeddings_french = "/work/home/dsu/IWSDS2023/NoXi_embeddings/Cross-corpus/French/test_embeddings.csv"

    path_to_train_embeddings_german= "/work/home/dsu/IWSDS2023/NoXi_embeddings/Cross-corpus/German/train_embeddings.csv"
    path_to_dev_embeddings_german = "/work/home/dsu/IWSDS2023/NoXi_embeddings/Cross-corpus/German/dev_embeddings.csv"
    path_to_test_embeddings_german = "/work/home/dsu/IWSDS2023/NoXi_embeddings/Cross-corpus/German/test_embeddings.csv"

    # load all embeddings in dictionary
    english_embeddings = {'train': pd.read_csv(path_to_train_embeddings_english),
                          'dev': pd.read_csv(path_to_dev_embeddings_english),
                          'test': pd.read_csv(path_to_test_embeddings_english)}

    french_embeddings = {'train': pd.read_csv(path_to_train_embeddings_french),
                         'dev': pd.read_csv(path_to_dev_embeddings_french),
                         'test': pd.read_csv(path_to_test_embeddings_french)}
    german_embeddings = {'train': pd.read_csv(path_to_train_embeddings_german),
                         'dev': pd.read_csv(path_to_dev_embeddings_german),
                         'test': pd.read_csv(path_to_test_embeddings_german)}
    # concatenate embeddings and divide on train, dev, and test parts
    embeddings = {'english': english_embeddings, 'french': french_embeddings, 'german': german_embeddings}
    # test part
    test_part = embeddings.pop(test_corpus)
    test_part = pd.concat([value for key,value in test_part.items()], axis=0, ignore_index=True)
    # train part
    train_part = [item.pop('train') for item in list(embeddings.values())] + [item.pop('test') for item in list(embeddings.values())]
    train_part = pd.concat(train_part, axis=0, ignore_index=True)
    # dev part
    dev_part = [item.pop('dev') for item in list(embeddings.values())]
    dev_part = pd.concat(dev_part, axis=0, ignore_index=True)
    # apply normalization
    scaler = StandardScaler()
    scaler = scaler.fit(train_part.iloc[:, 1:-5])
    train_part.iloc[:, 1:-5] = scaler.transform(train_part.iloc[:, 1:-5])
    dev_part.iloc[:, 1:-5] = scaler.transform(dev_part.iloc[:, 1:-5])
    test_part.iloc[:, 1:-5] = scaler.transform(test_part.iloc[:, 1:-5])

    # split embeddings
    train_part = split_embeddings_according_to_file_path(train_part)
    dev_part = split_embeddings_according_to_file_path(dev_part)
    test_part = split_embeddings_according_to_file_path(test_part)

    # done
    return train_part, dev_part, test_part


if __name__ == '__main__':
    train, dev, test = load_data_cross_corpus('english')
    print(train)