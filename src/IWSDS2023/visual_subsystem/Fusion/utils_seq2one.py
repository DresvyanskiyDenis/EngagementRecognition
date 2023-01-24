from functools import reduce, partial
from typing import Union, Tuple, List, Optional, Dict

import torch
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.decomposition import PCA
from torch.utils.data import Dataset


class Nflow_FusionSequenceDataLoader(Dataset):
    # TODO: comment this class, since things get too complicated
    possible_scalers = {
        "standard": preprocessing.StandardScaler,
        "PCA": PCA
    }

    def __init__(self, embeddings:List[pd.DataFrame], window_length:int, window_shift:int,
                 scalers:Union[List[str], List[object]]=None, labels_included:bool=False,
                 PCA_components:Optional[int]=None, sequence_to_one:bool=False,
                 data_as_list:Optional[bool]=False):
        self.embeddings = embeddings
        self.window_length = window_length
        self.window_shift = window_shift
        self.labels_included = labels_included
        self.scalers = scalers
        self.sequence_to_one = sequence_to_one
        self.data_as_list = data_as_list

        # check if the scalers are provided, and create ones, if the type of the scalers are provided (it is needed to fit it later)
        if self.scalers is not None:
            self._is_scaler_fit = True
            if isinstance(self.scalers[0], str):
                new_scalers = []
                for scaler in self.scalers:
                    if scaler in self.possible_scalers:
                        if scaler == "PCA": new_scalers.append(self.possible_scalers[scaler](n_components=PCA_components))
                        else: new_scalers.append(self.possible_scalers[scaler]())
                    else:
                        raise ValueError("Scaler %s is not supported!"%scaler)
                self.scalers = list(new_scalers)

        # split labels from embeddings
        if self.labels_included:
            self.labels = []
            for df in self.embeddings:
                self.labels.append(self.__splitting_labels_from_embeddings(df))

        # apply scaling to the data
        if self.scalers is not None:
            for i in range(len(embeddings)):
                # 1: because the first column if the filenames
                # save the first column, since it is the filename
                filenames = self.embeddings[i].iloc[:, 0]
                # scale the data
                scaled_data, scaler = self._scaling(self.embeddings[i].iloc[:,1:], self.scalers[i], fit=self._is_scaler_fit)
                # concatenate with filenames column
                self.embeddings[i] = pd.concat([filenames, pd.DataFrame(scaled_data)], axis=1)
                self.scalers[i] = scaler

        # check congruity between embeddings
        self._check_embeddings_congruity()

        # split the dataframe according to the file path
        self.split_embeddings = [self._split_df_according_to_file_path(df) for df in self.embeddings]


        # cutting the embeddings on windows
        # TODO: CHECK THE CONGRUITY OF LABELS AND DATA
        self.windows = []
        for split_df in self.split_embeddings:
            self.windows.append(self._cut_every_file_on_windows(split_df, self.window_length, self.window_shift))
        # delete two first columns, since they are useless (filename and frame_id)
        for i, window in enumerate(self.windows):
            self.windows[i] = window[:,:,2:].astype(np.float32)
        if self.labels_included:
            self.labels = self._split_df_according_to_file_path(self.labels)
            self.labels = self._cut_every_file_on_windows(self.labels, self.window_length, self.window_shift)
            self.labels = self.labels[:,:,2:].astype(np.float32)



    def __splitting_labels_from_embeddings(self, df_with_emb_and_lbs):
        labels = []
        # take all labels columns + filename (for further merging)
        labels_columns = ["filename"]+[col for col in df_with_emb_and_lbs.columns if "label" in col]
        labels.append(df_with_emb_and_lbs[labels_columns])
        # delete labels columns from the dataframe
        labels_columns = [col for col in df_with_emb_and_lbs.columns if "label" in col]
        df_with_emb_and_lbs.drop(columns=labels_columns, inplace=True)
        # merge labels into one dataframe and delete those, who incogurent
        # fuse embeddings into one dataframe based on their filenames
        labels = [df.set_index("filename") for df in labels]
        cols = list(labels[0].columns)
        num_labels = labels[0].shape[1]
        labels = reduce(lambda x,y: pd.merge(x,y, on='filename', how='inner'), labels).dropna()
        # take first num_labels, since all others are just duplicates from other dataframes
        # also, rename first num_labels columns to their old format, since they were renamed by pd.merge function
        labels.columns = cols + list(labels.columns)[len(cols):]
        labels = labels.iloc[:, :num_labels]
        # return back filenames
        labels.reset_index(inplace=True)
        return labels

    def _scaling(self, data, scaler, fit:bool=True):
        if scaler is None:
            return data, None
        if fit:
            scaler.fit(data)
        return scaler.transform(data), scaler


    def _check_embeddings_congruity(self):
        # TODO: check it
        # take only frames that only present in all dataframes
        ref_df = self.embeddings[0]
        if self.labels_included: ref_labels = self.labels[0]
        ref_df = ref_df[ref_df.set_index(['filename']).index.isin(self.embeddings[1].set_index(['filename']).index)]
        for i in range(0, len(self.embeddings)):
            self.embeddings[i] = self.embeddings[i][self.embeddings[i].set_index(['filename']).index.isin(ref_df.set_index(['filename']).index)]
        # sort the dataframes by filenames
        for i in range(len(self.embeddings)):
            self.embeddings[i] = self.embeddings[i].sort_values(by=['filename'])
        # reindex the dataframes
        for i in range(len(self.embeddings)):
            self.embeddings[i].set_index("filename", inplace=True)
            self.embeddings[i].reindex(index=ref_df['filename'])
            self.embeddings[i].reset_index(inplace=True)
        # check congruity of ref_labels with embeddings
        if self.labels_included:
            ref_labels = ref_labels[ref_labels.set_index(['filename']).index.isin(ref_df.set_index(['filename']).index)]
            ref_labels.set_index("filename", inplace=True)
            ref_labels.reindex(index=ref_df['filename'])
            ref_labels.reset_index(inplace=True)
            self.labels = ref_labels

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

    def __getitem__(self, idx):
        data_flows = []
        # extract the data (windows) from all dataflows
        for i in range(len(self.windows)):
            data = self.windows[i][idx]
            # add newaxis if the data should be presented as np.array
            if not self.data_as_list:
                data = data[np.newaxis,...]
            data_flows.append(data)
        # concatenate the data from all dataflows across the "flow" axis, if needed
        if not self.data_as_list:
            data_flows = np.concatenate(data_flows, axis=0)

        if self.labels_included:
            labels = self.labels[idx]
            if self.sequence_to_one:
                normalization_sum = np.sum(labels)
                labels = np.sum(labels, axis=0) / normalization_sum
            return data_flows, labels
        else:
            return data_flows

    def __len__(self):
        return self.windows[0].shape[0]