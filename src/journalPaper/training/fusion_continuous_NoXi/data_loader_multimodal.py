from collections import OrderedDict
from typing import Callable, List, Dict, Union, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from pytorch_utils.data_loaders.TemporalEmbeddingsLoader import TemporalEmbeddingsLoader


class MultiModalTemporalDataLoader(Dataset):

    def __init__(self, embeddings_with_labels:List[Dict[str, pd.DataFrame]], feature_columns:List[List[str]], label_columns:List[str],
                 window_size:Union[int, float], stride:Union[int, float],
                 consider_timesteps:Optional[bool]=False, timesteps_column:Optional[str]=None,
                 preprocessing_functions:List[Callable]=None, shuffle:bool=False):
        super().__init__()
        self.embeddings_with_labels = embeddings_with_labels
        self.feature_columns = feature_columns
        self.label_columns = label_columns
        self.window_size = window_size
        self.stride = stride
        self.consider_timesteps = consider_timesteps
        self.timesteps_column = timesteps_column
        self.preprocessing_functions = preprocessing_functions
        self.shuffle = shuffle

        # synchronize the dataframes
        self.synchronize_dataframes()

        # create loaders for every dict
        self.loaders = []
        for idx, embeddings_with_labels in enumerate(self.embeddings_with_labels):
            loader = TemporalEmbeddingsLoader(embeddings_with_labels=embeddings_with_labels,
                                              label_columns=label_columns,
                                              feature_columns=feature_columns[idx],
                                              window_size=window_size,
                                              stride=stride,
                                              consider_timestamps=consider_timesteps,
                                              preprocessing_functions=preprocessing_functions,
                                              only_consecutive_windows=False,
                                              shuffle=False)
            self.loaders.append(loader)




    def synchronize_dataframes(self):
        """ Synchronizes passed dataframes based on the several columns (using pandas merge)
        The columns for synchronization are formed from the "path" column that have the following format:
        */Session_id/Identity/Identity_timestep.png
        where Identity can be either 'Expert_video' or 'Novice_video'
        and timestep is the timestep of the video with _ that divides seconds and milliseconds.

        from the path, we will form new unique for every row identificator that will be used for synchronization
        it will be the following: Session_id/Identity_timestep
        """
        # check if the keys are the same for all the dataframes
        keys = list(self.embeddings_with_labels[0].keys())
        for i in range(1, len(self.embeddings_with_labels)):
            if keys != list(self.embeddings_with_labels[i].keys()):
                raise ValueError("The keys of the dataframes are not the same")
        # go over the keys and synchronize the dataframes
        for session in keys:
            # get the dataframes and form in them new key_column
            for i in range(len(self.embeddings_with_labels)):
                df = self.embeddings_with_labels[i][session]
                df['key_column'] = df.apply(lambda x: x['path'].split('/')[-3]+'/'+x['path'].split('/')[-1], axis=1)
                self.embeddings_with_labels[i][session] = df
            # get intersection of the key_columns for all the dataframes
            intersection = self.embeddings_with_labels[0][session]['key_column']
            for i in range(1, len(self.embeddings_with_labels)):
                intersection = intersection[intersection.isin(self.embeddings_with_labels[i][session]['key_column'])]
            # synchronize the dataframes
            for i in range(len(self.embeddings_with_labels)):
                df = self.embeddings_with_labels[i][session]
                self.embeddings_with_labels[i][session] = df[df['key_column'].isin(intersection)]
                self.embeddings_with_labels[i][session] = self.embeddings_with_labels[i][session].drop(columns=['key_column'])


    def __len__(self):
        return self.loaders[0].__len__()

    def __getitem__(self, idx:int):
        data = []
        labels = []
        for loader in self.loaders:
            data_, labels_ = loader.__getitem__(idx)
            data.append(data_)
            labels.append(labels_)
        return data, labels


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



if __name__ == "__main__":
    # test the synchronization
    pose_dev = pd.read_csv("/nfs/scratch/ddresvya/NoXi/NoXi/prepared_data/embeddings/static/NoXi_pose_embeddings_train.csv")
    face_dev = pd.read_csv("/nfs/scratch/ddresvya/NoXi/NoXi/prepared_data/embeddings/static/NoXi_face_embeddings_train.csv")
    emo_dev = pd.read_csv("/nfs/scratch/ddresvya/NoXi/NoXi/prepared_data/embeddings/static/NoXi_affective_embeddings_train.csv")
    # divide every dataframe into sessions
    pose_dev = divide_df_into_sessions(pose_dev)
    face_dev = divide_df_into_sessions(face_dev)
    emo_dev = divide_df_into_sessions(emo_dev)

    # make data loader
    loader = MultiModalTemporalDataLoader(embeddings_with_labels=[pose_dev, face_dev, emo_dev],
                                          feature_columns=[['embedding_%i'%i for i in range(256)],
                                                           ['embedding_%i'%i for i in range(256)],
                                                           ['embedding_%i'%i for i in range(256)]],
                                          label_columns=['engagement', 'timestep'],
                                          window_size=4.,
                                          stride=2.,
                                          consider_timesteps=True,
                                          preprocessing_functions=None,
                                          shuffle=False)

    for x, y in loader:
        # compare ys between each other (they are tensors)
        comp1 = torch.equal(y[0][:,0], y[1][:,0]) & torch.equal(y[0][:,0], y[2][:,0])
        comp2 = torch.equal(y[0][:,1], y[1][:,1]) & torch.equal(y[0][:,1], y[2][:,1])
        if not comp1 or not comp2:
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! NOT EQUAL!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        timesteps = torch.cat([y[0][:,1].unsqueeze(1), y[1][:,1].unsqueeze(1), y[2][:,1].unsqueeze(1)], dim=1)
        labels = torch.cat([y[0][:,0].unsqueeze(1), y[1][:,0].unsqueeze(1), y[2][:,0].unsqueeze(1)], dim=1)
        print("Timesteps: ", timesteps)
        print("Labels: ", labels)
        print("-----------------")



