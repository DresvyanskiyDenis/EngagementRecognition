import os
from typing import List, Union, Optional

import torch
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.decomposition import PCA
from torch.utils.data import Dataset


class FusionDataLoader(Dataset):
    def __init__(self, embeddings:List[pd.DataFrame], scaling:Union[str, object]=None, labels_included:bool=False,
                 PCA_components:Optional[int]=None):
        self.embeddings = embeddings
        self.labels_included = labels_included
        self.scaling = scaling

        # check if the scaler is provided, and create one, if the type of the scaler is provided (it is needed to fit it later)
        if self.scaling is not None:
            self._is_scaler_fit = True
            # if str (type) provided, create scaler
            if isinstance(self.scaling, str):
                # check if the scaler is available
                if not self.scaling in ("standard", "minmax", "PCA"):
                    raise ValueError(f"Unknown scaler type: {self.scaling}")
                self._is_scaler_fit = False
                if self.scaling == "standard": self.scaler = preprocessing.StandardScaler()
                elif self.scaling == "minmax": self.scaler = preprocessing.MinMaxScaler()
                elif self.scaling == "PCA":
                    if PCA_components is None: raise ValueError("PCA_components must be provided.")
                    self.PCA_components = PCA_components
                    self.scaler = PCA(n_components = PCA_components)

        # fuse embeddings into one dataframe
        self._fuse_embeddings_into_one_df()

        # check what is the labels start (in terms of columns)
        # we need to do -1, since first column will be deleted later
        self.labels_start_idx = self.embeddings[0].shape[1]
        if self.labels_included:
            self.labels_start_idx = list(self.embeddings[0].columns).index("label_0") - 1

        # delete filename column
        self.embeddings.drop(columns=['filename'], inplace=True)
        # scaling
        if self.scaling is not None:
            self._scaling()

    def _fuse_embeddings_into_one_df(self):
        # rename columns for further processing
        for num, embeddings_df in enumerate(self.embeddings):
            cols = list(embeddings_df.columns)
            if self.labels_included:
                end_idx = list(self.embeddings[0].columns).index("label_0") - 1
            else:
                end_idx = embeddings_df.shape[1]
            cols[1:end_idx] = [x+"_df_%i"%num for x in cols[1:end_idx]]
            embeddings_df.columns = cols

        # fuse embeddings into one dataframe based on their filenames
        self.embeddings = [df.set_index("filename") for df in self.embeddings]
        self.embeddings = pd.concat(self.embeddings, axis=1, ignore_index=False).dropna()
        # return back filenames
        self.embeddings.reset_index(inplace=True)
        # cut off labels to delete them from dataframe and concat at the end of the columns
        # this will help to delete duplicated columns
        if self.labels_included:
            # TODO: come up with an idea how to delete all duplicated columns with labels
            # also, the procedure with renaming columns at the start of this function seem redundant and too stupid
            # can you rewrite it somehow?
            pass



    def _scaling(self):
        # apply scaling function to the dataframe
        if self.scaler is not None:
            # if scaler was not provided, but chosen, we need to fit (train) it first
            if not self._is_scaler_fit:
                self.scaler = self.scaler.fit(self.embeddings.iloc[:, -self.labels_start_idx:])
            # apply scaler to the dataframe
            self.embeddings.iloc[:] = self.scaler.transform(self.embeddings.iloc[:])

    def __len__(self):
        return self.embeddings.shape[0]


    def __getitem__(self, idx):
        data = self.embeddings.iloc[idx, -self.labels_start_idx:].values
        labels = self.embeddings.iloc[idx, :-self.labels_start_idx].values
        if self.labels_included:
            return data, labels
        else:
            return data


def cut_filenames_to_original_names(df:pd.DataFrame):
    df['filename'] = df['filename'].apply(lambda x: os.path.join(*x.split(os.path.sep)[-3:]))
    return df


if __name__=="__main__":
    train_1 = pd.read_csv("/work/home/dsu/NoXi/NoXi_embeddings/All_languages/Pose_model/embeddings_dev.csv")
    train_1 = cut_filenames_to_original_names(train_1)
    train_2 = pd.read_csv("/work/home/dsu/NoXi/NoXi_embeddings/All_languages/Xception_model/dev_extracted_deep_embeddings.csv")
    train_2 = cut_filenames_to_original_names(train_2)
    generator = FusionDataLoader([train_1, train_2], scaling="PCA", PCA_components=100,
                                 labels_included=True)

    for x,y in generator:
        print(x.shape, y.shape)
        break