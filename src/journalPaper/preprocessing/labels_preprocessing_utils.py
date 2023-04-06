import sys
sys.path.extend(["/work/home/dsu/datatools/"])
sys.path.extend(["/work/home/dsu/engagement_recognition_project_server/"])
sys.path.extend(["/work/home/dsu/simple-HRNet-master/"])

import gc
import glob
import os
from typing import List, Optional

import pandas as pd
import numpy as np


def turn_class_indices_into_one_hot(class_indices: List[int], number_of_classes: int) -> np.ndarray:
    """
    Turns class indices into one-hot vector.
    :param class_indices: List[int]
            List of indices, which should be turn into one-hot vector.
    :param number_of_classes: int
            Overall number of classes.
    :return: np.ndarray
            One-hot vectors.
    """
    one_hot = np.eye(number_of_classes)[np.array(class_indices).reshape((-1,))]
    return one_hot


def align_labels_with_frames_for_DAiSEE(df_with_frame_paths:pd.DataFrame, df_with_labels: pd.DataFrame) -> pd.DataFrame:
    """
    Aligns labels with frames for DAISEE dataset. The labels will be aligned to image frames.
    param df_with_frame_paths: pd.DataFrame
            Dataframe with paths to frames. The columns are: ['path_to_frame', 'timestamp']
    :param df_with_labels: pd.DataFrame
            Dataframe with labels. The columns are: ['ClipID', 'Boredom', 'Engagement', 'Confusion', 'Frustration']
    :return: pd.DataFrame
            Dataframe with aligned paths to frames and labels. The columns are: ['path_to_frame', 'timestamp', 'engagement']
    """
    df_with_labels['ClipID'] = df_with_labels['ClipID'].astype(str).apply(lambda x: x.split('.')[0])
    df_with_labels = df_with_labels.drop(columns=["Boredom", "Confusion", "Frustration "])
    df_with_labels = df_with_labels.rename(columns={"Engagement": "engagement",
                                                    "ClipID": "filename"})
    df_with_frame_paths['filename'] = df_with_frame_paths['path_to_frame'].astype(str).apply(lambda x: x.split('/')[-1].split('_')[0])

    # merge dataframes upon their filenames
    result_df = pd.merge(df_with_frame_paths, df_with_labels, on='filename', how='left')
    # drop NaN values (the filenames, which did not exist either in df_with_frame_paths or in df_with_labels) and drop the abundunt filename column
    result_df = result_df.drop(columns=['filename']).dropna().reset_index(drop=True)
    return result_df

def prepare_df_for_DAiSEE(df_with_frame_paths:pd.DataFrame, df_with_labels: pd.DataFrame) -> pd.DataFrame:
    """
    Prepares dataframe for DAISEE dataset. The labels will be aligned to image frames.
    The classes will be represented using one-hot encoding.

    :param df_with_frame_paths: pd.DataFrame
            Dataframe with paths to frames. The columns are: ['path_to_frame', 'timestamp', 'detected']
    :param df_with_labels: pd.DataFrame
            Dataframe with labels. The columns are: ['ClipID', 'Boredom', 'Engagement', 'Confusion', 'Frustration']
    :return: pd.DataFrame
            Dataframe with aligned paths to frames and labels and one-hot encoding. THe columns are: ['path_to_frame', 'timestamp', 'engagement']
    """
    # clear df_with_frame_paths from NaN and False values in detected columns
    df_with_frame_paths = df_with_frame_paths[df_with_frame_paths['detected'] == True].reset_index(drop=True)
    df_with_frame_paths = df_with_frame_paths.drop(columns=['detected'])
    df_with_frame_paths = df_with_frame_paths.dropna().reset_index(drop=True)
    # align (match) labels with frames
    aligned_df = align_labels_with_frames_for_DAiSEE(df_with_frame_paths, df_with_labels)
    # turn class indices to one-hot vectors
    one_hot_vectors = turn_class_indices_into_one_hot(class_indices=aligned_df['engagement'].values.astype(int),
                                                      number_of_classes=4)
    one_hot_vectors = pd.DataFrame(one_hot_vectors, columns=['label_0', 'label_1', 'label_2', 'label_3'])
    # drop engagement column and add one-hot vectors instead of it
    aligned_df = aligned_df.drop(columns=['engagement'])
    aligned_df = pd.concat([aligned_df, one_hot_vectors], axis=1)

    return aligned_df



def form_df_with_labels_from_NoXi_labels(path_to_labels:str, FPS:int=25)->pd.DataFrame:
    """
    Forms dataframe with labels from NoXi labels.
    :param path_to_labels: str
            Path to dir with labels.
    :return: pd.DataFrame
            Dataframe with labels. The columns are: ['filename', 'frame', 'timestep', 'label_0', 'label_1', 'label_2', 'label_3', 'label_4']
    """
    filenames = glob.glob(os.path.join(path_to_labels, '*', '*.txt'))
    result_df = pd.DataFrame(columns=['filename', 'frame', 'timestep', 'label_0', 'label_1', 'label_2', 'label_3', 'label_4'])
    for filename in filenames:
        # form final filename from provided path
        result_filename = os.path.join(filename.split('/')[-2], filename.split('/')[-1].split('_')[-1].split('.')[0])
        # load labels
        label_values = np.loadtxt(filename, delimiter=' ')
        # define the number of frames
        frame_num = np.arange(0, label_values.shape[0], 1)
        # calculate timesteps for number of frames
        timesteps = np.round(np.arange(0, frame_num.shape[0])*(1.0/FPS), 2)
        # assemble the final dataframe
        df = pd.DataFrame(columns=['filename', 'frame', 'timestep', 'label_0', 'label_1', 'label_2', 'label_3', 'label_4'])
        df['filename'] = [result_filename]*timesteps.shape[0]
        df['frame'] = timesteps
        df['timestep'] = timesteps
        df['label_0'] = label_values[:, 0]
        df['label_1'] = label_values[:, 1]
        df['label_2'] = label_values[:, 2]
        df['label_3'] = label_values[:, 3]
        df['label_4'] = label_values[:, 4]
        result_df = pd.concat([result_df, df], axis=0)
    return result_df


def align_labels_with_frames_for_NoXi(df_with_frame_paths:pd.DataFrame, df_with_labels: pd.DataFrame) -> pd.DataFrame:
    """
    Aligns labels with frames for NoXi dataset. The labels will be aligned to image frames.
    param df_with_frame_paths: pd.DataFrame
            Dataframe with paths to frames. The columns are: ['path_to_frame', 'timestamp']
    :param df_with_labels: pd.DataFrame
            Dataframe with labels. The columns are: ['filename', 'frame', 'timestep', 'label_0', 'label_1', 'label_2', 'label_3', 'label_4']
    :return: pd.DataFrame
            Dataframe with aligned paths to frames and labels. The columns are: ['path_to_frame', 'timestep', 'label_0', 'label_1', 'label_2', 'label_3', 'label_4']
    """
    # form path_to_frame from 'path_to_frame' and 'timestamp'
    df_with_labels['path_to_frame'] = df_with_labels['filename'].str.split(os.path.sep).str[0]+str(os.path.sep)+\
                                      df_with_labels['filename'].str.split(os.path.sep).str[1].str.capitalize()+'_video'+str(os.path.sep)+\
                                      df_with_labels['filename'].str.split(os.path.sep).str[1].str.capitalize()+\
                                      '_video_'+df_with_labels['timestep'].astype(float).round(2).astype(str).str.replace('.','_')+'.png'

    # make path_to_frame column of df_with_frame_paths to be the same as in df_with_labels
    # the absolute path to all frames is saved and added at the end of the function
    general_path_to_data = os.path.join(*df_with_frame_paths.iloc[0,0].split(os.path.sep)[:-3])
    df_with_frame_paths['path_to_frame'] = df_with_frame_paths['path_to_frame'].apply(lambda x: os.path.join(*x.split(os.path.sep)[-3:]))
    # merge two dataframes based on path_to_frame
    df_with_labels['path_to_frame'] = df_with_labels['path_to_frame'].astype(str)
    df_with_frame_paths['path_to_frame'] = df_with_frame_paths['path_to_frame'].astype(str)
    result_df = pd.merge(df_with_frame_paths, df_with_labels, on='path_to_frame', how='left').dropna()
    # drop unnecessary columns
    result_df = result_df.drop(columns=['filename', 'frame', 'timestamp']).reset_index(drop=True)
    # add absolute path to all frames
    result_df['path_to_frame'] = result_df['path_to_frame'].apply(lambda x: os.path.join(general_path_to_data, x))
    return result_df



def prepare_df_for_NoXi(df_with_frame_paths:pd.DataFrame, path_to_labels: str) -> pd.DataFrame:
    """
    Prepares dataframe for NoXi dataset. The labels will be aligned to image frames.
    :param df_with_frame_paths: pd.DataFrame
            Dataframe with paths to frames. The columns are: ['path_to_frame', 'timestamp', 'detected']
    :param df_with_labels: str
            Path to dir with labels.
    :return: pd.DataFrame
            Dataframe with aligned paths to frames and labels. The columns are: ['path_to_frame', 'timestamp', 'label_0', 'label_1', 'label_2', 'label_3', 'label_4']
    """
    # clear df_with_frame_paths from NaN and False values in detected columns
    df_with_frame_paths = df_with_frame_paths[df_with_frame_paths['detected'] == True].reset_index(drop=True)
    df_with_frame_paths = df_with_frame_paths.drop(columns=['detected'])
    df_with_frame_paths = df_with_frame_paths.dropna().reset_index(drop=True)
    # get labels packed in dataframe from NoXi labels
    df_with_labels = form_df_with_labels_from_NoXi_labels(path_to_labels=path_to_labels, FPS=25)
    # align (match) labels with frames
    aligned_df = align_labels_with_frames_for_NoXi(df_with_frame_paths, df_with_labels)

    return aligned_df


def prepare_df_for_MHHRI(df_with_frame_paths:pd.DataFrame, path_to_labels: str) -> pd.DataFrame:
    """ TODO: check it
    Prepares dataframe for NoXi dataset. The labels will be aligned to image frames.
    :param df_with_frame_paths: pd.DataFrame
            Dataframe with paths to frames. The columns are: ['path_to_frame', 'timestamp', 'detected']
    :param df_with_labels: str
            Path to dir with labels.
    :return: pd.DataFrame
            Dataframe with aligned paths to frames and labels. The columns are: ['path_to_frame', 'timestamp', 'label_0', 'label_1', 'label_2', 'label_3', 'label_4']
    """
    # clear df_with_frame_paths from NaN and False values in detected columns
    df_with_frame_paths = df_with_frame_paths[df_with_frame_paths['detected'] == True].reset_index(drop=True)
    df_with_frame_paths = df_with_frame_paths.drop(columns=['detected'])
    df_with_frame_paths = df_with_frame_paths.dropna().reset_index(drop=True)
    # load labels from csv file
    df_with_labels = load_MHHRI_labels(path_to_labels=path_to_labels)
    # align (match) labels with frames
    aligned_df = align_MHHRI_labels_with_frames(df_with_frame_paths, df_with_labels)

    return aligned_df



def load_MHHRI_labels(path_to_labels:str)->pd.DataFrame:
    """ Loads MHHRI labels from csv file. The MHHRI labels file is presented in format:
    column1:U002,U001, ..., U001 (identifier of the assesed participant)
    column2:S02,S01,...,S02 (identifier of the session)
    column3:5,1,...,4 (label 0)
    etc

    We need label 0 (engagement in the human-human conversation) and label 6 (engagement in the human-robot conversation)
    All labels are in range [1, 10]

    :param path_to_labels: str
            Path to the labels file.
    :return: pd.DataFrame
            Dataframe with labels. The columns are: ['participant_id', 'session_id', 'engagement_hhi', 'engagement_hri']
    """

    labels = pd.read_csv(path_to_labels, header=None)
    # transform dataframe to required format
    labels = labels.T
    labels.columns = ['participant_id', 'session_id', 'engagement_hhi', 'label_1', 'label_2', 'label_3', 'label_4', 'label_5', 'engagement_hri']
    # drop unnecessary columns
    labels = labels.drop(columns=['label_1', 'label_2', 'label_3', 'label_4', 'label_5'])

    return labels


def align_MHHRI_labels_with_frames(df_with_frame_paths:pd.DataFrame, df_with_labels:pd.DataFrame)->pd.DataFrame:
    """ Aligns MHHRI labels with already extracted frames. Basically, since one interaction has only one label values,
    it assigns to all frames from all videos of this interaction the same label value.

    :param df_with_frame_paths: pd.DataFrame
            Dataframe with paths to frames. The columns are: ['path_to_frame', 'timestamp', 'detected']
    :param df_with_labels: pd.DataFrame
            Dataframe with labels. The columns are: ['participant_id', 'session_', 'engagement_hhi', 'engagement_hri']
    :return: pd.DataFrame
            Dataframe with aligned paths to frames and labels. The columns are: ['path_to_frame', 'timestamp', 'engagement_hhi', 'engagement_hri']
    """
    # extract from df_with_frame_paths['path_to_frame'] the id of the second person (the person, which is seen on EGO camera)
    # and the id of the session
    df_with_frame_paths['visible_person'] = df_with_frame_paths['path_to_frame'].apply(lambda x:
                                                                                    os.path.basename(x).split('_')[4])
    df_with_frame_paths['session_id'] = df_with_frame_paths['path_to_frame'].apply(lambda x:
                                                                                    os.path.basename(x).split('_')[0])
    # depending on the id session and id of the second person, we can find the corresponding label
    df_with_frame_paths['engagement_hhi'] = df_with_frame_paths.apply(lambda x:
                df_with_labels[(df_with_labels['participant_id'] == x['visible_person']) &
                (df_with_labels['session_id'] == x['session_id'])]['engagement_hhi'].values[0],
                                                                      axis=1)

    df_with_frame_paths['engagement_hri'] = df_with_frame_paths.apply(lambda x:
                df_with_labels[(df_with_labels['participant_id'] == x['visible_person']) &
                (df_with_labels['session_id'] == x['session_id'])]['engagement_hri'].values[0],
                                                                        axis=1)

    # drop unnecessary columns
    df_with_frame_paths = df_with_frame_paths.drop(columns=['visible_person', 'session_id'])

    return df_with_frame_paths





if __name__=="__main__":
    # process the NoXi dataset, pose frames
    # params
    path_to_labels = '/media/external_hdd_2/NoXi/NoXi_annotations_reliable_gold_standard_classification_with_additional_train_data/'
    path_to_frames = '/media/external_hdd_2/NoXi/prepared_data/poses/'
    path_to_df_with_frame_paths = '/media/external_hdd_2/NoXi/prepared_data/poses/metadata.csv'
    output_path = '/media/external_hdd_2/NoXi/prepared_data/poses/'
    # load dataframe with paths to frames
    df_with_frame_paths = pd.read_csv(path_to_df_with_frame_paths)
    # load and prepare labels
    english_train = prepare_df_for_NoXi(df_with_frame_paths, path_to_labels=os.path.join(path_to_labels, 'English', 'train'))
    english_dev = prepare_df_for_NoXi(df_with_frame_paths, path_to_labels=os.path.join(path_to_labels, 'English', 'dev'))
    english_test = prepare_df_for_NoXi(df_with_frame_paths, path_to_labels=os.path.join(path_to_labels, 'English', 'test'))
    german_train = prepare_df_for_NoXi(df_with_frame_paths, path_to_labels=os.path.join(path_to_labels, 'German', 'train'))
    german_dev = prepare_df_for_NoXi(df_with_frame_paths, path_to_labels=os.path.join(path_to_labels, 'German', 'dev'))
    german_test = prepare_df_for_NoXi(df_with_frame_paths, path_to_labels=os.path.join(path_to_labels, 'German', 'test'))
    french_train = prepare_df_for_NoXi(df_with_frame_paths, path_to_labels=os.path.join(path_to_labels, 'French', 'train'))
    french_dev = prepare_df_for_NoXi(df_with_frame_paths, path_to_labels=os.path.join(path_to_labels, 'French', 'dev'))
    french_test = prepare_df_for_NoXi(df_with_frame_paths, path_to_labels=os.path.join(path_to_labels, 'French', 'test'))
    # concatenate train, dev and test dataframes
    train_NoXi = pd.concat([english_train, german_train, french_train], axis=0).reset_index(drop=True)
    dev_NoXi = pd.concat([english_dev, german_dev, french_dev], axis=0).reset_index(drop=True)
    test_NoXi = pd.concat([english_test, german_test, french_test], axis=0).reset_index(drop=True)
    # save dataframes
    train_NoXi.to_csv(os.path.join(output_path, 'NoXi_pose_train.csv'), index=False)
    dev_NoXi.to_csv(os.path.join(output_path, 'NoXi_pose_dev.csv'), index=False)
    test_NoXi.to_csv(os.path.join(output_path, 'NoXi_pose_test.csv'), index=False)


    # process the NoXi dataset, face frames
    # params
    path_to_labels = '/media/external_hdd_2/NoXi/NoXi_annotations_reliable_gold_standard_classification_with_additional_train_data/'
    path_to_frames = '/media/external_hdd_2/NoXi/prepared_data/faces/'
    path_to_df_with_frame_paths = '/media/external_hdd_2/NoXi/prepared_data/faces/metadata.csv'
    output_path = '/media/external_hdd_2/NoXi/prepared_data/faces/'
    # load dataframe with paths to frames
    df_with_frame_paths = pd.read_csv(path_to_df_with_frame_paths)
    # load and prepare labels
    english_train = prepare_df_for_NoXi(df_with_frame_paths,
                                        path_to_labels=os.path.join(path_to_labels, 'English', 'train'))
    english_dev = prepare_df_for_NoXi(df_with_frame_paths,
                                      path_to_labels=os.path.join(path_to_labels, 'English', 'dev'))
    english_test = prepare_df_for_NoXi(df_with_frame_paths,
                                       path_to_labels=os.path.join(path_to_labels, 'English', 'test'))
    german_train = prepare_df_for_NoXi(df_with_frame_paths,
                                       path_to_labels=os.path.join(path_to_labels, 'German', 'train'))
    german_dev = prepare_df_for_NoXi(df_with_frame_paths, path_to_labels=os.path.join(path_to_labels, 'German', 'dev'))
    german_test = prepare_df_for_NoXi(df_with_frame_paths,
                                      path_to_labels=os.path.join(path_to_labels, 'German', 'test'))
    french_train = prepare_df_for_NoXi(df_with_frame_paths,
                                       path_to_labels=os.path.join(path_to_labels, 'French', 'train'))
    french_dev = prepare_df_for_NoXi(df_with_frame_paths, path_to_labels=os.path.join(path_to_labels, 'French', 'dev'))
    french_test = prepare_df_for_NoXi(df_with_frame_paths,
                                      path_to_labels=os.path.join(path_to_labels, 'French', 'test'))
    # concatenate train, dev and test dataframes
    train_NoXi = pd.concat([english_train, german_train, french_train], axis=0).reset_index(drop=True)
    dev_NoXi = pd.concat([english_dev, german_dev, french_dev], axis=0).reset_index(drop=True)
    test_NoXi = pd.concat([english_test, german_test, french_test], axis=0).reset_index(drop=True)
    # save dataframes
    train_NoXi.to_csv(os.path.join(output_path, 'NoXi_facial_train.csv'), index=False)
    dev_NoXi.to_csv(os.path.join(output_path, 'NoXi_facial_dev.csv'), index=False)
    test_NoXi.to_csv(os.path.join(output_path, 'NoXi_facial_test.csv'), index=False)


    # process DAiSEE dataset (train), pose frames
    # params
    path_to_frames = '/media/external_hdd_2/DAiSEE/prepared_data/poses/'
    path_to_df_with_frame_paths = "/media/external_hdd_2/DAiSEE/prepared_data/poses/metadata.csv"
    path_to_df_with_labels = '/media/external_hdd_2/DAiSEE/DAiSEE/Labels/TrainLabels.csv'
    output_path = '/media/external_hdd_2/DAiSEE/prepared_data/poses/'
    filename = 'DAiSEE_pose_train_labels.csv'
    # load dataframe with paths to frames
    df_with_frame_paths = pd.read_csv(path_to_df_with_frame_paths)
    # load dataframe with labels
    df_with_labels = pd.read_csv(path_to_df_with_labels)
    # prepare dataframe for DAiSEE dataset
    df_with_labels = prepare_df_for_DAiSEE(df_with_frame_paths, df_with_labels)
    # save dataframe
    df_with_labels.to_csv(os.path.join(output_path, filename), index=False)

    # process DAiSEE dataset (dev), pose frames
    # params
    path_to_frames = '/media/external_hdd_2/DAiSEE/prepared_data/poses/'
    path_to_df_with_frame_paths = "/media/external_hdd_2/DAiSEE/prepared_data/poses/metadata.csv"
    path_to_df_with_labels = '/media/external_hdd_2/DAiSEE/DAiSEE/Labels/ValidationLabels.csv'
    output_path = '/media/external_hdd_2/DAiSEE/prepared_data/poses/'
    filename = 'DAiSEE_pose_dev_labels.csv'
    # load dataframe with paths to frames
    df_with_frame_paths = pd.read_csv(path_to_df_with_frame_paths)
    # load dataframe with labels
    df_with_labels = pd.read_csv(path_to_df_with_labels)
    # prepare dataframe for DAiSEE dataset
    df_with_labels = prepare_df_for_DAiSEE(df_with_frame_paths, df_with_labels)
    # save dataframe
    df_with_labels.to_csv(os.path.join(output_path, filename), index=False)

    # process DAiSEE dataset (test), pose frames
    # params
    path_to_frames = '/media/external_hdd_2/DAiSEE/prepared_data/poses/'
    path_to_df_with_frame_paths = "/media/external_hdd_2/DAiSEE/prepared_data/poses/metadata.csv"
    path_to_df_with_labels = '/media/external_hdd_2/DAiSEE/DAiSEE/Labels/TestLabels.csv'
    output_path = '/media/external_hdd_2/DAiSEE/prepared_data/poses/'
    filename = 'DAiSEE_pose_test_labels.csv'
    # load dataframe with paths to frames
    df_with_frame_paths = pd.read_csv(path_to_df_with_frame_paths)
    # load dataframe with labels
    df_with_labels = pd.read_csv(path_to_df_with_labels)
    # prepare dataframe for DAiSEE dataset
    df_with_labels = prepare_df_for_DAiSEE(df_with_frame_paths, df_with_labels)
    # save dataframe
    df_with_labels.to_csv(os.path.join(output_path, filename), index=False)

    # process DAiSEE dataset (train), face frames
    # params
    path_to_frames = '/media/external_hdd_2/DAiSEE/prepared_data/faces/'
    path_to_df_with_frame_paths = "/media/external_hdd_2/DAiSEE/prepared_data/faces/metadata.csv"
    path_to_df_with_labels = '/media/external_hdd_2/DAiSEE/DAiSEE/Labels/TrainLabels.csv'
    output_path = '/media/external_hdd_2/DAiSEE/prepared_data/faces/'
    filename = 'DAiSEE_facial_train_labels.csv'
    # load dataframe with paths to frames
    df_with_frame_paths = pd.read_csv(path_to_df_with_frame_paths)
    # load dataframe with labels
    df_with_labels = pd.read_csv(path_to_df_with_labels)
    # prepare dataframe for DAiSEE dataset
    df_with_labels = prepare_df_for_DAiSEE(df_with_frame_paths, df_with_labels)
    # save dataframe
    df_with_labels.to_csv(os.path.join(output_path, filename), index=False)

    # process DAiSEE dataset (dev), face frames
    # params
    path_to_frames = '/media/external_hdd_2/DAiSEE/prepared_data/faces/'
    path_to_df_with_frame_paths = "/media/external_hdd_2/DAiSEE/prepared_data/faces/metadata.csv"
    path_to_df_with_labels = '/media/external_hdd_2/DAiSEE/DAiSEE/Labels/ValidationLabels.csv'
    output_path = '/media/external_hdd_2/DAiSEE/prepared_data/faces/'
    filename = 'DAiSEE_facial_dev_labels.csv'
    # load dataframe with paths to frames
    df_with_frame_paths = pd.read_csv(path_to_df_with_frame_paths)
    # load dataframe with labels
    df_with_labels = pd.read_csv(path_to_df_with_labels)
    # prepare dataframe for DAiSEE dataset
    df_with_labels = prepare_df_for_DAiSEE(df_with_frame_paths, df_with_labels)
    # save dataframe
    df_with_labels.to_csv(os.path.join(output_path, filename), index=False)

    # process DAiSEE dataset (test), face frames
    # params
    path_to_frames = '/media/external_hdd_2/DAiSEE/prepared_data/faces/'
    path_to_df_with_frame_paths = "/media/external_hdd_2/DAiSEE/prepared_data/faces/metadata.csv"
    path_to_df_with_labels = '/media/external_hdd_2/DAiSEE/DAiSEE/Labels/TestLabels.csv'
    output_path = '/media/external_hdd_2/DAiSEE/prepared_data/faces/'
    filename = 'DAiSEE_facial_test_labels.csv'
    # load dataframe with paths to frames
    df_with_frame_paths = pd.read_csv(path_to_df_with_frame_paths)
    # load dataframe with labels
    df_with_labels = pd.read_csv(path_to_df_with_labels)
    # prepare dataframe for DAiSEE dataset
    df_with_labels = prepare_df_for_DAiSEE(df_with_frame_paths, df_with_labels)
    # save dataframe
    df_with_labels.to_csv(os.path.join(output_path, filename), index=False)


    """# process MHHRI dataset (HHI), face frames
    # params
    path_to_df_with_frame_paths = "/media/external_hdd_2/MHHRI/mhhri/prepared_data/HHI_Ego_Recordings/faces/metadata.csv"
    path_to_df_with_labels = "/media/external_hdd_2/MHHRI/mhhri/dataset/annotations/MHHRI_labels/post_questionnaire.csv"
    output_path = '/media/external_hdd_2/MHHRI/mhhri/prepared_data/HHI_Ego_Recordings/faces/'
    metadata = pd.read_csv(path_to_df_with_frame_paths)
    prepared_labels = prepare_df_for_MHHRI(df_with_frame_paths=metadata, path_to_labels=path_to_df_with_labels)
    prepared_labels.to_csv(os.path.join(output_path, 'MHHRI_facial_labels.csv'), index=False)

    # process MHHRI dataset (HHI), pose frames
    # params
    path_to_df_with_frame_paths = "/media/external_hdd_2/MHHRI/mhhri/prepared_data/HHI_Ego_Recordings/poses/metadata.csv"
    path_to_df_with_labels = "/media/external_hdd_2/MHHRI/mhhri/dataset/annotations/MHHRI_labels/post_questionnaire.csv"
    output_path = '/media/external_hdd_2/MHHRI/mhhri/prepared_data/HHI_Ego_Recordings/poses/'
    metadata = pd.read_csv(path_to_df_with_frame_paths)
    prepared_labels = prepare_df_for_MHHRI(df_with_frame_paths=metadata, path_to_labels=path_to_df_with_labels)
    prepared_labels.to_csv(os.path.join(output_path, 'MHHRI_pose_labels.csv'), index=False)

    # process MHHRI dataset (HRI), face frames
    # params
    path_to_df_with_frame_paths = "/media/external_hdd_2/MHHRI/mhhri/prepared_data/HRI_Ego_Recordings/faces/metadata.csv"
    path_to_df_with_labels = "/media/external_hdd_2/MHHRI/mhhri/dataset/annotations/MHHRI_labels/post_questionnaire.csv"
    output_path = '/media/external_hdd_2/MHHRI/mhhri/prepared_data/HRI_Ego_Recordings/faces/'
    metadata = pd.read_csv(path_to_df_with_frame_paths)
    prepared_labels = prepare_df_for_MHHRI(df_with_frame_paths=metadata, path_to_labels=path_to_df_with_labels)
    prepared_labels.to_csv(os.path.join(output_path, 'MHHRI_facial_labels.csv'), index=False)

    # process MHHRI dataset (HRI), pose frames
    # params
    path_to_df_with_frame_paths = "/media/external_hdd_2/MHHRI/mhhri/prepared_data/HRI_Ego_Recordings/poses/metadata.csv"
    path_to_df_with_labels = "/media/external_hdd_2/MHHRI/mhhri/dataset/annotations/MHHRI_labels/post_questionnaire.csv"
    output_path = '/media/external_hdd_2/MHHRI/mhhri/prepared_data/HRI_Ego_Recordings/poses/'
    metadata = pd.read_csv(path_to_df_with_frame_paths)
    prepared_labels = prepare_df_for_MHHRI(df_with_frame_paths=metadata, path_to_labels=path_to_df_with_labels)
    prepared_labels.to_csv(os.path.join(output_path, 'MHHRI_pose_labels.csv'), index=False)"""







