from typing import Callable, List, Dict, Union

import pandas as pd
import numpy as np
import sklearn.preprocessing as preprocessing
from sklearn.decomposition import PCA
from torch.utils.data import Dataset

class ImageDataLoader(Dataset):
    def __init__(self, dataframe:pd.DataFrame, window_length:int, window_shift:int, labels_included:bool = False,
                 scaler:Union[str, object, None] = None):
        """

        :param dataframe: pd.DataFrame
                        dataframe with the following columns: ['filename', 'embedding_0', 'embedding_1', ..., 'embedding_n'
        (optional): 'label_0', 'label_1', ..., 'label_n']
                        filename is a full path to the image/frame, while the base name should be like 'frame_<number>.png'
        :param window_length: int
                        length of the sliding window (in frames)
        :param window_shift: int
                        shift of the sliding window (in frames)
        :param labels_included: bool
                        whether the dataframe contains labels or not
        :param scaler: Union[str, object, None]
                        scaler to be used for normalization of the dataframe
                        can be None, 'minmax', 'standard', 'pca', or already trained (fit) one from sklearn library
        """
        self.dataframe = dataframe
        self.labels_included = labels_included
        self.window_length = window_length
        self.window_shift = window_shift
        # define which scaler will be used
        if scaler is not None:
            if isinstance(scaler, str):
                if scaler == "standard": self.scaler = preprocessing.StandardScaler()
                elif scaler == "minmax": self.scaler = preprocessing.MinMaxScaler()
                elif scaler == "PCA": self.scaler = PCA(n_components = 100) # WARNING: HARD CODED
                self._fit_scaler = False
            else:
                self.scaler = scaler
                self._fit_scaler = True

        # split the dataframe according to the file path
        self.split_embeddings = self._split_df_according_to_file_path(self.dataframe)
        # cut every file on provided windows and concatenate them in overall big dataframe
        self.windows = self._cut_every_file_on_windows(self.split_embeddings, self.window_length, self.window_shift)
        # indentify start of the labels in columns
        self.labels_start_idx = list(self.windows.columns).index("label_0")
        # apply scaler to the dataframe
        self.windows = self._scaling(self.windows)

    def __len__(self):
        return sum(value.shape[0] for value in self.split_embeddings.values())


    def __getitem__(self, idx):
        data = self.windows.iloc[idx,:self.labels_start_idx]
        labels = self.windows.iloc[idx,self.labels_start_idx:]
        return data, labels

    def _scaling(self, df):
        # apply scaling function to the dataframe
        if self.scaler is not None:
            # if scaler was not provided, but chosen, we need to fit (train) it first
            if self._fit_scaler:
                self.scaler = self.scaler.fit(df.iloc[:,:self.labels_start_idx])
            # apply scaler to the dataframe
            df.iloc[:,:self.labels_start_idx] = self.scaler.transform(df.iloc[:,:self.labels_start_idx])

        return df



    def _split_df_according_to_file_path(self, df) -> Dict[str, pd.DataFrame]:
        """
        Splits the embeddings according to the file path in the first column.
        :param embeddings: pd.DataFrame
                pandas dataframe containing the embeddings
        :return: Dict[str, pd.DataFrame]
                dictionary containing the embeddings split by file path
        """
        # split the filename to two columns: video_filename and frame_id
        df["frame_id"] = df["filename"].apply(lambda x: x.split("/")[-1].split(".")[0].split("_")[-1])
        df["frame_id"] = df["frame_id"].astype('int32')
        df.rename(columns={"filename": "video_filename"}, inplace=True)
        df["video_filename"] = df["video_filename"].apply(lambda x: x[:x.rfind("/")])
        # rearrange columns so that video_filename and the frame_id will be the first two columns
        embeddings = df[["video_filename", "frame_id"] + df.columns[1:-1].tolist()]

        split_embeddings = {}
        for file_path in embeddings['video_filename'].unique():
            split_embeddings[file_path] = embeddings[embeddings['video_filename'] == file_path].sort_values(by="frame_id")
        return split_embeddings

    def _cut_every_file_on_windows(self, files:Dict[str, pd.DataFrame], window_length:int, window_shift:int)->pd.DataFrame:
        result_df = pd.DataFrame(columns = files.columns[2:]) # drop first two columns: video_filename and frame_id
        for filename, values in files.items():
            windows = self.__cut_df_on_windows(values, window_length, window_shift)
            windows = np.concatenate(windows, axis=0)
            windows = pd.DataFrame(windows, columns = result_df.columns)
            result_df = pd.concat([result_df, windows], axis=0, ignore_index=True)
        return result_df


    def __cut_df_on_windows(self, df, window_length, window_shift):
        windows = []
        start_idx = 0
        end_idx = window_length
        if end_idx > df.shape[0]:
            raise ValueError("The window is bigger than the whole sequence!")
        while True:
            chunk = df.iloc[start_idx:end_idx].values
            windows.append(chunk)
            start_idx += window_shift
            end_idx += window_shift
            if end_idx > df.shape[0]:
                end_idx = df.shape[0]
                start_idx = end_idx - window_shift
                chunk = df.iloc[start_idx:end_idx].values
                windows.append(chunk)
                break
        return windows