




import glob
from typing import Dict

import pandas as pd
import numpy as np



def prepare_paths_df(df_with_frame_paths:pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    Prepares metadata file for NoXi dataset.
    """
    # clear df_with_frame_paths from NaN and False values in detected columns
    df_with_frame_paths = df_with_frame_paths[df_with_frame_paths['detected'] == True].reset_index(drop=True)
    df_with_frame_paths = df_with_frame_paths.drop(columns=['detected'])
    df_with_frame_paths = df_with_frame_paths.dropna().reset_index(drop=True)
    # divide the path_to_labels into filenames using dictionary
    unique_filenames = df_with_frame_paths['path_to_frame'].apply(lambda x: '/'.join(x.split('/')[-3:-1])).unique()
    result = {}
    for filename in unique_filenames:
        result[filename] = df_with_frame_paths[df_with_frame_paths['path_to_frame'].str.contains(filename)]
    # change keys to the sessionID_novice/expert format
    for key in list(result.keys()):
        session_id = key.split('/')[-2] + '_' + ('novice' if 'Novice' in key or 'novice' in key else 'expert')
        result[session_id] = result.pop(key)
    return result


def load_NoXi_original_labels(path_to_labels:str):
    """ Loads original NoXi labels.
    in the directory with the path_to_labels, there should be two subdirectories: train and dev.
    Every subdirectory then contains subsubdirectories with the names of the sessions. In them, the labels for novice and expert are stored.
    Every label file for novice contains 'novice' in its name and for expert 'expert'.
    """
    train_files = glob.glob(path_to_labels + '/train/*/*.csv')
    dev_files = glob.glob(path_to_labels + '/dev/**/*.csv')
    # read files using pandas
    train = {}
    for train_file in train_files:
        session_id = train_file.split('/')[-2] + '_' + ('novice' if 'novice' in train_file.split('/')[-1] else 'expert')
        df = pd.read_csv(train_file, sep=';', header=None)
        df.columns = ['timestep', 'engagement','confidence']
        # drop rows with NaN values
        df = df.dropna().reset_index(drop=True)
        df.drop(columns=['confidence'], inplace=True)
        # assign to the dictionary
        train[session_id] = df
    # dev
    dev = {}
    for dev_file in dev_files:
        session_id = dev_file.split('/')[-2] + '_' + ('novice' if 'novice' in dev_file.split('/')[-1] else 'expert')
        df = pd.read_csv(dev_file, sep=';', header=None)
        df.columns = ['timestep', 'engagement','confidence']
        # drop rows with NaN values
        df = df.dropna().reset_index(drop=True)
        df.drop(columns=['confidence'], inplace=True)
        # assign to the dictionary
        dev[session_id] = df
    return train, dev


def align_labels_with_frames(df_with_frame_paths:Dict[str, pd.DataFrame], labels:Dict[str, pd.DataFrame])->Dict[str, pd.DataFrame]:
    """ Aligns labels with the frames. If there are more frames than labels, the residual frames are removed.
    If there are more labels than frames, the residual labels are removed.
    The aligning should be done using the timesteps.

    :param df_with_frame_paths: Dict[str, pd.DataFrame]
        Dictionary with the paths to the frames. The keys are sessionID_novice/expert.
    :param labels: Dict[str, pd.DataFrame]
        Dictionary with the labels. The keys are sessionID_novice/expert.
    :return: Dict[str, pd.DataFrame]
        Dictionary with the aligned labels. The keys are sessionID_novice/expert.
    """
    # take the intersection of the keys
    keys = list(set(df_with_frame_paths.keys()) & set(labels.keys()))
    result = {}
    for key in keys:
        df = df_with_frame_paths[key]
        # change 'timestamp' to 'timestep'
        df.columns = ['path_to_frame','timestep']
        labels_df = labels[key]
        # merge the labels with the frames
        result[key] = pd.merge(df, labels_df, on='timestep', how='inner')
    return result




if __name__ == '__main__':
    path_to_metafile_face = "/nfs/scratch/ddresvya/NoXi/NoXi/prepared_data/faces/metadata.csv"
    path_to_metafile_pose = "/nfs/scratch/ddresvya/NoXi/NoXi/prepared_data/poses/metadata.csv"
    path_to_labels = "/nfs/scratch/ddresvya/NoXi/NoXi/NoXi_annotations_original/"
    output_path = "/nfs/scratch/ddresvya/NoXi/NoXi/prepared_data/"
    # load metadata files
    df_with_frame_paths_face = pd.read_csv(path_to_metafile_face)
    df_with_frame_paths_pose = pd.read_csv(path_to_metafile_pose)
    # prepare paths dataframes
    df_with_frame_paths_face = prepare_paths_df(df_with_frame_paths_face)
    df_with_frame_paths_pose = prepare_paths_df(df_with_frame_paths_pose)
    # load original labels
    train, dev = load_NoXi_original_labels(path_to_labels)
    # align labels with frames
    aligned_labels_train_face = align_labels_with_frames(df_with_frame_paths_face, train)
    aligned_labels_dev_face = align_labels_with_frames(df_with_frame_paths_face, dev)
    aligned_labels_train_pose = align_labels_with_frames(df_with_frame_paths_pose, train)
    aligned_labels_dev_pose = align_labels_with_frames(df_with_frame_paths_pose, dev)
    # save aligned labels. To do so, combine them into one dataframe, separately for train, dev
    aligned_labels_train_face = pd.concat(list(aligned_labels_train_face.values()), ignore_index=True)
    aligned_labels_dev_face = pd.concat(list(aligned_labels_dev_face.values()), ignore_index=True)
    aligned_labels_train_pose = pd.concat(list(aligned_labels_train_pose.values()), ignore_index=True)
    aligned_labels_dev_pose = pd.concat(list(aligned_labels_dev_pose.values()), ignore_index=True)
    # saving as csv file
    aligned_labels_train_face.to_csv(output_path + 'aligned_labels_train_face.csv', index=False)
    aligned_labels_dev_face.to_csv(output_path + 'aligned_labels_dev_face.csv', index=False)
    aligned_labels_train_pose.to_csv(output_path + 'aligned_labels_train_pose.csv', index=False)
    aligned_labels_dev_pose.to_csv(output_path + 'aligned_labels_dev_pose.csv', index=False)






