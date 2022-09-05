import os
from functools import reduce
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

        # split labels from embeddings
        if self.labels_included:
            self.labels = self.__splitting_labels_from_embeddings(self.embeddings)

        # fuse embeddings into one dataframe
        self._fuse_embeddings_into_one_df()

        # check the congruity of the labels and embeddings
        if self.labels_included:
            self.check_congruity()
        # delete filename column
        self.embeddings.drop(columns=['filename'], inplace=True)
        self.labels.drop(columns=['filename'], inplace=True)
        # scaling
        if self.scaling is not None:
            self._scaling()

    def __splitting_labels_from_embeddings(self, df_with_emb_and_lbs):
        labels = []
        for df in df_with_emb_and_lbs:
            # take all labels columns + filename (for further merging)
            labels_columns = ["filename"]+[col for col in df.columns if "label" in col]
            labels.append(df[labels_columns])
            # delete labels columns from the dataframe
            labels_columns = [col for col in df.columns if "label" in col]
            df.drop(columns=labels_columns, inplace=True)
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


    def _fuse_embeddings_into_one_df(self):
        # rename columns for further processing
        for num, embeddings_df in enumerate(self.embeddings):
            cols = list(embeddings_df.columns)
            cols[1:] = [x+"_df_%i"%num for x in cols[1:]]
            embeddings_df.columns = cols

        # fuse embeddings into one dataframe based on their filenames
        self.embeddings = [df.set_index("filename") for df in self.embeddings]
        self.embeddings = pd.concat(self.embeddings, axis=1, ignore_index=False).dropna()
        # return back filenames
        self.embeddings.reset_index(inplace=True)

    def check_congruity(self):
        if self.embeddings.shape[0] != self.labels.shape[0]:
            raise ValueError("Number of embeddings and labels are not equal! "
                             "Something in either embeddings or labels preprocessing went wrong...")
        # reindex labels so that it has the same order as embeddings
        self.labels.set_index("filename", inplace=True)
        self.labels.reindex(index =self.embeddings['filename'])
        self.labels.reset_index(inplace=True)

    def _scaling(self):
        # apply scaling function to the dataframe
        if self.scaler is not None:
            # if scaler was not provided, but chosen, we need to fit (train) it first
            if not self._is_scaler_fit:
                self.scaler = self.scaler.fit(self.embeddings)
            # apply scaler to the dataframe
            self.embeddings = self.scaler.transform(self.embeddings.iloc[:])

    def __len__(self):
        return self.embeddings.shape[0]


    def __getitem__(self, idx):
        data = self.embeddings[idx]
        if self.labels_included:
            labels = self.labels.iloc[idx].values
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
    train_3 = pd.read_csv("/work/home/dsu/NoXi/NoXi_embeddings/All_languages/EmoVGGFace2//dev_extracted_deep_embeddings.csv")
    train_3 = cut_filenames_to_original_names(train_3)
    generator = FusionDataLoader([train_1, train_2, train_3], scaling="PCA", PCA_components=10,
                                 labels_included=True)

    for i, (x,y) in enumerate(generator):
        print(i, x.shape, y.shape)
