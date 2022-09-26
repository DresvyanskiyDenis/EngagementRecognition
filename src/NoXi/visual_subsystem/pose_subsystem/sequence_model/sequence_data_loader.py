from typing import Callable, List, Dict, Union

import pandas as pd
import numpy as np
import sklearn.preprocessing as preprocessing
import torch
from sklearn.decomposition import PCA
from torch.utils.data import Dataset
import scipy
from src.NoXi.visual_subsystem.pose_subsystem.frame_wise_model.utils import load_NoXi_data_all_languages


class SequenceDataLoader(Dataset):
    def __init__(self, dataframe:pd.DataFrame, window_length:int, window_shift:int, labels_included:bool = False,
                 scaler:Union[str, object, None] = None, sequence_to_one:bool=True):
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
        self.sequence_to_one = sequence_to_one
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
        # indentify start of the labels in columns. We need -1, since first columns will be deleted later
        self.labels_start_idx = list(self.dataframe.columns).index("label_0") - 1
        # apply scaler to the dataframe
        if scaler is not None:
            filename = self.dataframe[:,0]
            labels = self.dataframe.iloc[:, self.labels_start_idx:]
            scaled_dataframe = self._scaling(self.dataframe.iloc[:,1:self.labels_start_idx])
            final_df = pd.concat([filename, scaled_dataframe, labels], axis=1)
            self.dataframe = final_df

        # split the dataframe according to the file path
        self.split_embeddings = self._split_df_according_to_file_path(self.dataframe)
        # cut every file on provided windows and concatenate them in overall big dataframe
        self.windows = self._cut_every_file_on_windows(self.split_embeddings, self.window_length, self.window_shift)
        # delete two first columns, since they are useless (filename and frame_id)
        self.windows = self.windows[:,:,2:]
        self.windows = self.windows.astype(np.float32)

    def __len__(self):
        return self.windows.shape[0]

    def clear_RAM(self):
        del self.dataframe
        del self.split_embeddings
        self.dataframe = None
        self.split_embeddings = None

    def __getitem__(self, idx):
        data = self.windows[idx][:,:self.labels_start_idx]
        labels = self.windows[idx][:,self.labels_start_idx:]
        if self.sequence_to_one:
            normalization_sum = np.sum(labels)
            labels = np.sum(labels, axis=0)/normalization_sum
        return data, labels

    def _scaling(self, df):
        # apply scaling function to the dataframe
        if self.scaler is not None:
            # if scaler was not provided, but chosen, we need to fit (train) it first
            if not self._fit_scaler:
                self.scaler = self.scaler.fit(df.iloc[:])
            # apply scaler to the dataframe
            df.iloc[:] = self.scaler.transform(df.iloc[:])

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
        all_windows =[]
        for filename, values in files.items():
            windows = self.__cut_df_on_windows(values, window_length, window_shift)
            windows = [window[np.newaxis,...] for window in windows]
            windows = np.concatenate(windows, axis=0)
            all_windows.append(windows)
        # concatenate all_windows into one array
        all_windows = np.concatenate(all_windows, axis=0)
        return all_windows


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
                start_idx = end_idx - window_length
                chunk = df.iloc[start_idx:end_idx].values
                windows.append(chunk)
                break
        return windows


if __name__=="__main__":
    # load data
    train = pd.read_csv("/work/home/dsu/NoXi/NoXi_embeddings/All_languages/Pose_model/embeddings_train.csv")

    data_loader = SequenceDataLoader(dataframe=train, window_length=40, window_shift=10, labels_included=True,
                 scaler="standard")

    train_generator = torch.utils.data.DataLoader(data_loader, batch_size=128, shuffle=True,
                                                  num_workers=16, pin_memory=False)

    for i, (data, labels) in enumerate(train_generator):
        print(i, data.shape, labels.shape)