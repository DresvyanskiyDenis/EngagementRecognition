from typing import Tuple

import numpy as np
import pandas as pd

def load_data():
    train_pose_scores=pd.read_csv('train_pose_scores.csv')
    train_facial_scores=pd.read_csv('train_facial_scores.csv')
    # take only those filenames, which are in both dataframes
    train_pose_scores= train_pose_scores[train_pose_scores.set_index(['filename']).index.isin(train_facial_scores.set_index(['filename']).index)]
    train_facial_scores= train_facial_scores[train_facial_scores.set_index(['filename']).index.isin(train_pose_scores.set_index(['filename']).index)]
    # sort to have the same order
    train_pose_scores=train_pose_scores.sort_values(by=['filename'])
    train_facial_scores=train_facial_scores.sort_values(by=['filename'])
    # check congruity
    if train_pose_scores.shape[0] != train_facial_scores.shape[0]:
        raise ValueError('Dataframes have different number of rows')
    train_facial_scores.set_index("filename", inplace=True)
    train_facial_scores.reindex(index=train_pose_scores['filename'])
    train_facial_scores.reset_index(inplace=True)
    # now the format is: [filename, score_0, score_1,..., label_0, label_1,...]
    # development
    dev_pose_scores=pd.read_csv('dev_pose_scores.csv')
    dev_facial_scores=pd.read_csv('dev_facial_scores.csv')
    # take only those filenames, which are in both dataframes
    dev_pose_scores= dev_pose_scores[dev_pose_scores.set_index(['filename']).index.isin(dev_facial_scores.set_index(['filename']).index)]
    dev_facial_scores= dev_facial_scores[dev_facial_scores.set_index(['filename']).index.isin(dev_pose_scores.set_index(['filename']).index)]
    # sort to have the same order
    dev_pose_scores=dev_pose_scores.sort_values(by=['filename'])
    dev_facial_scores=dev_facial_scores.sort_values(by=['filename'])
    # check congruity
    if dev_pose_scores.shape[0] != dev_facial_scores.shape[0]:
        raise ValueError('Dataframes have different number of rows')
    dev_facial_scores.set_index("filename", inplace=True)
    dev_facial_scores.reindex(index=dev_pose_scores['filename'])
    dev_facial_scores.reset_index(inplace=True)

    return (train_pose_scores, train_facial_scores), (dev_pose_scores, dev_facial_scores)


def find_best_weights(train:Tuple[pd.DataFrame,...], dev:Tuple[pd.DataFrame,...], num_generations:int=10000)->np.array:
    pass

def train_simple_dnn_model(train:Tuple[pd.DataFrame,...], dev:Tuple[pd.DataFrame,...])->None:
    pass



def main():
    (train_pose, train_facial), (dev_pose, dev_facial) = load_data()



if __name__=='__main__':
    main()