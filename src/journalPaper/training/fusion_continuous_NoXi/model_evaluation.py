import glob
from typing import List, Dict, Union

import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr
from src.journalPaper.training.fusion_continuous_NoXi.data_preparation import load_dataframes_3_modalities
from src.journalPaper.training.fusion_continuous_NoXi.metrics import np_concordance_correlation_coefficient


def evaluate_model(model, generator, modalities:List[str], device,
                   metrics_name_prefix, print_metrics):
    evaluation_metrics = {'CCC': np_concordance_correlation_coefficient,
                                         'PCC': lambda x,y: pearsonr(x,y)[0],
                                         }
    # change the names of the metrics if needed
    if metrics_name_prefix is not None:
        evaluation_metrics = {metrics_name_prefix + metric_name: metric for metric_name, metric in
                                             evaluation_metrics.items()}
    # create arrays for predictions and ground truth labels
    predictions = []
    ground_truth = []
    # start
    model.eval()
    with torch.no_grad():
        for i, data in enumerate(generator):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            # order of modalities is always [face, pose, emo]. We need to choose the ones that are included in current experiment
            tmp = []
            if 'face' in modalities: tmp.append(inputs[0])
            if 'pose' in modalities: tmp.append(inputs[1])
            if 'emo' in modalities: tmp.append(inputs[2])
            inputs = tmp
            # now we have only the modalities that are included in the experiment
            inputs = [input.float().to(device) for input in inputs]
            # labels shape: List of (batch_size, sequence_length, 1)
            # as all labels are the same, just take the first element of list
            labels = labels[0].float()
            # forward pass
            outputs = model(*inputs)
            # transform output to numpy
            outputs = outputs.cpu().numpy().squeeze()
            labels = labels.cpu().numpy().squeeze()
            # save ground_truth labels and predictions in arrays to calculate metrics afterwards by one time
            predictions.append(outputs)
            ground_truth.append(labels)
        # concatenate all predictions and ground truth labels
        predictions = np.concatenate(predictions, axis=0).flatten()
        ground_truth = np.concatenate(ground_truth, axis=0).flatten()
        # calculate evaluation metrics
        evaluation_metrics = {
            metric: evaluation_metrics[metric](ground_truth, predictions) for metric in evaluation_metrics.keys()}
        if print_metrics:
            print(evaluation_metrics)
        return evaluation_metrics



def __load_NoXi_original_labels(path_to_labels:str):
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

def __synchronize_dataframes(input_dataframes):
    """ Synchronizes passed dataframes based on the several columns (using pandas merge)
    The columns for synchronization are formed from the "path" column that have the following format:
    */Session_id/Identity/Identity_timestep.png
    where Identity can be either 'Expert_video' or 'Novice_video'
    and timestep is the timestep of the video with _ that divides seconds and milliseconds.

    from the path, we will form new unique for every row identificator that will be used for synchronization
    it will be the following: Session_id/Identity_timestep
    """
    # check if the keys are the same for all the dataframes
    keys = list(input_dataframes[0].keys())
    for i in range(1, len(input_dataframes)):
        if keys != list(input_dataframes[i].keys()):
            raise ValueError("The keys of the dataframes are not the same")
    # go over the keys and synchronize the dataframes
    for session in keys:
        # get the dataframes and form in them new key_column
        for i in range(len(input_dataframes)):
            df = input_dataframes[i][session]
            df['key_column'] = df.apply(lambda x: x['path'].split('/')[-3]+'/'+x['path'].split('/')[-1], axis=1)
            input_dataframes[i][session] = df
        # get intersection of the key_columns for all the dataframes
        intersection = input_dataframes[0][session]['key_column']
        for i in range(1, len(input_dataframes)):
            intersection = intersection[intersection.isin(input_dataframes[i][session]['key_column'])]
        # synchronize the dataframes
        for i in range(len(input_dataframes)):
            df = input_dataframes[i][session]
            input_dataframes[i][session] = df[df['key_column'].isin(intersection)]
            input_dataframes[i][session] = input_dataframes[i][session].drop(columns=['key_column'])
    return input_dataframes

def __cut_sequence_on_windows(sequence:pd.DataFrame, window_size:int, stride:int)->Union[List[pd.DataFrame],None]:
        """ Cuts one sequence of values (represented as pd.DataFrame) into windows with fixed size. The stride is used
        to move the window. If there is not enough values to fill the last window, the window starting from
        sequence_end-window_size is added as a last window.

        :param sequence: pd.DataFrame
                Sequence of values represented as pd.DataFrame
        :param window_size: int
                Size of the window in number of values/frames
        :param stride: int
                Stride of the window in number of values/frames
        :return: List[pd.DataFrame]
                List of windows represented as pd.DataFrames
        """
        # check if the sequence is long enough
        # if not, return None and this sequence will be skipped in the __cut_all_data_on_windows method
        if sequence.shape[0] < window_size:
            return None
        windows = []
        # cut sequence on windows using while and shifting the window every step
        window_start = 0
        window_end = window_start + window_size
        while window_end <= len(sequence):
            windows.append(sequence.iloc[window_start:window_end])
            window_start += stride
            window_end += stride
        # add last window if there is not enough values to fill it
        if window_start < len(sequence):
            windows.append(sequence.iloc[-window_size:])
        return windows


def __interpolate_to_100_fps(df:pd.DataFrame)->pd.DataFrame:
    # duplicate prediction for 0.0 if the first timestep is not 0.0
    if df['timestep'].iloc[0] != 0.0:
        df = pd.concat([pd.DataFrame({'timestep': [0.0], 'prediction': [df['prediction'].iloc[0]]}), df], ignore_index=True).reset_index(drop=True)
    # make milliseconds
    df['milliseconds'] = df['timestep'] * 1000
    # set timestep as index
    df = df.set_index('milliseconds')
    # convert index to TimeDeltaIndex
    df.index = pd.to_timedelta(df.index, unit='ms')
    # Resample the DataFrame to 100 FPS (0.01 seconds interval)
    df = df.resample('10ms').asfreq()
    # Interpolate the missing values
    df = df.interpolate(method='linear')
    # Reset the index to get the timesteps back as a column
    df.reset_index(inplace=True)
    df['timestep'] = df['milliseconds'].dt.total_seconds().apply("float64")
    # round timesteps to 2 decimal places
    df['timestep'] = df['timestep'].apply(lambda x: round(x, 2))
    df.drop(columns=['milliseconds'], inplace=True)
    return df



def evaluate_model_one_session(embeddings_dfs: List[pd.DataFrame], origin_labels, modalities,
                               window_size, stride, feature_columns, model, device):
    model.eval()
    # filter out modalities that we do not need (order is always face, pose, emo)
    embeddings = []
    if 'face' in modalities: embeddings.append(embeddings_dfs[0])
    if 'pose' in modalities: embeddings.append(embeddings_dfs[1])
    if 'emo' in modalities: embeddings.append(embeddings_dfs[2])
    # cutting sequence on windows (every modality)
    for idx in range(len(embeddings)):
        embeddings[idx] = __cut_sequence_on_windows(embeddings[idx], window_size*5, stride*5) # FPS is 5
    # make predictions for every sequence, add them to windows
    predictions_session = pd.DataFrame(columns=['timestep', 'prediction'])
    for window_idx in range(len(embeddings[0])):
        inputs = [emb[window_idx] for emb in embeddings]
        inputs = [input[feature_columns] for input in inputs]
        inputs = [torch.Tensor(input.values).unsqueeze(0) for input in inputs]
        inputs = [input.float().to(device) for input in inputs]
        with torch.no_grad():
            prediction = model(*inputs).cpu().numpy().squeeze()
        # extract timesteps
        timesteps = embeddings[0][window_idx]['timestep'].values
        # add to the predictions_session
        predictions_session = pd.concat([predictions_session,
                                         pd.DataFrame({'timestep': timesteps, 'prediction': prediction})],
                                        ignore_index=True)
    # round timesteps to 2 decimal places
    predictions_session['timestep'] = predictions_session['timestep'].apply(lambda x: round(x, 2))
    # average predictions based on the timesteps
    predictions_session = predictions_session.groupby('timestep').mean().reset_index()
    # now we need to interpolate results to the original labels as they have 25 in second
    # and our predictions have 5 in second
    # we will use linear interpolation
    predictions_session = __interpolate_to_100_fps(predictions_session)
    # choose only predictions that have the same timestep as origin labels
    predictions_session = predictions_session[predictions_session['timestep'].isin(origin_labels['timestep'])]
    if origin_labels.shape[0]>predictions_session.shape[0]:
        # cut off last few rows of origin labels
        origin_labels = origin_labels.iloc[:predictions_session.shape[0]]
    return predictions_session['prediction'].values.flatten(), origin_labels['engagement'].values.flatten() # order: predictions, ground truth







def evaluate_model_full(modalities:List[str], model, device,
                        metrics_name_prefix, print_metrics):
    evaluation_metrics = {'CCC': np_concordance_correlation_coefficient,
                          'PCC': lambda x, y: pearsonr(x, y)[0],
                          }
    # for the convenience, we will define all parameters inside the function.
    path_to_original_labels = "/nfs/scratch/ddresvya/NoXi/NoXi/NoXi_annotations_original/"
    paths_to_embeddings = {
        "face_train": "/nfs/scratch/ddresvya/NoXi/NoXi/prepared_data/embeddings/static/NoXi_face_embeddings_train.csv",
        "face_dev": "/nfs/scratch/ddresvya/NoXi/NoXi/prepared_data/embeddings/static/NoXi_face_embeddings_dev.csv",
        "pose_train": "/nfs/scratch/ddresvya/NoXi/NoXi/prepared_data/embeddings/static/NoXi_pose_embeddings_train.csv",
        "pose_dev": "/nfs/scratch/ddresvya/NoXi/NoXi/prepared_data/embeddings/static/NoXi_pose_embeddings_dev.csv",
        "emo_train": "/nfs/scratch/ddresvya/NoXi/NoXi/prepared_data/embeddings/static/NoXi_affective_embeddings_train.csv",
        "emo_dev": "/nfs/scratch/ddresvya/NoXi/NoXi/prepared_data/embeddings/static/NoXi_affective_embeddings_dev.csv",
    }
    # load original labels
    train_origin, dev_origin = __load_NoXi_original_labels(path_to_original_labels)
    # change keys of origin labels to fit embeddings
    tmp = {}
    for key in dev_origin.keys():
        new_key = "_".join(key.split('_')[:-1])+'/'+key.split('_')[-1].capitalize()+'_video'
        tmp[new_key] = dev_origin[key]
    dev_origin = tmp




    # load dev embeddings
    face_dev, pose_dev, emo_dev = load_dataframes_3_modalities(paths_to_embeddings['face_dev'],
                                                               paths_to_embeddings['pose_dev'],
                                                               paths_to_embeddings['emo_dev'])
    # synchronize dataframes
    face_dev, pose_dev, emo_dev = __synchronize_dataframes([face_dev, pose_dev, emo_dev])
    session_ids = list(face_dev.keys())

    # calculate PCC and CCC for every session separately
    results = []
    for session_id in session_ids:
        preds, gts = evaluate_model_one_session([face_dev[session_id], pose_dev[session_id], emo_dev[session_id]],
                                            dev_origin[session_id], modalities, 4, 2,
                                            ['embedding_%i'%i for i in range(256)], model, device)
        results.append((preds, gts))
    # calculate metrics. We do two approaches: average the results and calculate metrics on the concatenated predictions and ground truth labels
    # first calls gen_avg, the second concat
    result = {
        'gen_avg_val_PCC': np.mean([evaluation_metrics['PCC'](preds, gts) for preds, gts in results]),
        'gen_avg_val_CCC': np.mean([evaluation_metrics['CCC'](preds, gts) for preds, gts in results]),
        'concat_val_PCC': evaluation_metrics['PCC'](np.concatenate([preds for preds, gts in results], axis=0),
                                                    np.concatenate([gts for preds, gts in results], axis=0)),
        'concat_val_CCC': evaluation_metrics['CCC'](np.concatenate([preds for preds, gts in results], axis=0),
                                                    np.concatenate([gts for preds, gts in results], axis=0))
    }
    # print metrics if needed
    if print_metrics:
        print(result)
    return result





