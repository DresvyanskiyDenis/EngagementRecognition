import os
from typing import Tuple, List

import tensorflow as tf
import numpy as np
import pandas as pd
from scipy.stats import mode
from tensorflow.python.keras.utils.data_utils import Sequence


class SequenceLoader(Sequence):
    features: pd.DataFrame
    labels:pd.DataFrame
    batch_size: int
    num_classes: int
    scaling:bool
    num_frames_in_seq:int
    proportion_of_intersection:float
    class_label:str

    def __init__(self, features:pd.DataFrame,labels:pd.DataFrame, batch_size:int,
                 num_classes:int, num_frames_in_seq:int,
                 proportion_of_intersection:float, class_label:str,scaling:bool=False):
        self.features=features
        self.batch_size=batch_size
        self.num_classes=num_classes
        self.num_frames_in_seq=num_frames_in_seq
        self.proportion_of_intersection=proportion_of_intersection
        self.labels=labels
        self.scaling=scaling
        self.class_label=class_label

        if 'filename' not in self.features.columns.to_list():
            raise AttributeError('Features dataframe should contain filename column in format filename_frame.')
        if 'filename' not in self.labels.columns.to_list():
            raise AttributeError('Labels dataframe should contain filename column in format filename_frame.')

        self._prepare_dataframe_for_sequence_extraction()

    def _prepare_features_and_labels_dataframes(self):
        self.features[['filename', 'frame_num']] = self.features['filename'].str.rsplit('_', 1, expand=True)
        self.features['frame_num']=self.features['frame_num'].astype('int32')
        self.labels[['filename', 'frame_num']] = self.labels['filename'].str.rsplit('_', 1, expand=True)
        self.labels['frame_num'] = self.labels['frame_num'].astype('int32')

    def _prepare_dataframe_for_sequence_extraction(self) -> None:
        # convert filename_frame format to separate columns
        self._prepare_features_and_labels_dataframes()
        # calculate step of window based on proportion
        step = int(np.ceil(self.num_frames_in_seq * self.proportion_of_intersection))
        # sort dataframe by filename and then by frame_num
        self.features = self.features.sort_values(by=['filename', 'frame_num'])
        self.labels = self.labels.sort_values(by=['filename', 'frame_num'])
        # divide dataframe on sequences
        self.feature_sequences = self._divide_dataframe_on_sequences(self.features, self.num_frames_in_seq, step)
        self.labels_sequences = self._divide_dataframe_on_sequences(self.labels, self.num_frames_in_seq, step)

    def _divide_dataframe_on_sequences(self, dataframe: pd.DataFrame, seq_length: int, step: int) -> List[pd.DataFrame]:
        unique_filenames = np.unique(dataframe['filename'])
        # TODO: implement dividing whole dataframe on sequences
        sequences = []
        for unique_filename in unique_filenames:
            df_to_cut = dataframe[dataframe['filename'] == unique_filename]
            try:
                cut_df = self._divide_dataframe_on_list_of_seq(df_to_cut, seq_length, step)
            except AttributeError:
                continue
            sequences = sequences + cut_df
        return sequences

    def _divide_dataframe_on_list_of_seq(self, dataframe: pd.DataFrame, seq_length: int, step: int) -> List[
        pd.DataFrame]:
        sequences = []
        if dataframe.shape[0] < seq_length:
            # TODO: create your own exception
            raise AttributeError('The length of dataframe is less than seq_length. '
                                 'Dataframe length:%i, seq_length:%i' % (dataframe.shape[0], seq_length))
        num_sequences = int(np.ceil((dataframe.shape[0] - seq_length) / step + 1))
        for num_seq in range(num_sequences - 1):
            start = num_seq * step
            end = start + seq_length
            seq = dataframe.iloc[start:end]
            sequences.append(seq)
        # last sequence is from the end of sequence to end-seq_length
        sequences.append(dataframe.iloc[(dataframe.shape[0] - seq_length):])
        return sequences

    def _sequence_to_one_transformation(self, labels:np.ndarray)->np.ndarray:
        labels=mode(labels, axis=1)[0].reshape((-1,1))
        return labels

    def _one_hot_encoding(self, labels: np.ndarray) -> np.ndarray:
        # one-hot-label encoding
        labels = np.eye(self.num_classes)[labels.reshape((-1,)).astype('int32')]
        return labels

    def _load_and_preprocess_batch(self, index:int)->Tuple[np.ndarray, np.ndarray]:
        # choose needed
        feature_sequences=self.feature_sequences[index*batch_size:(index+1)*batch_size]
        labels_sequences = self.labels_sequences[index * batch_size:(index + 1) * batch_size]
        # transform sequences by dleting columns
        feature_sequences=[np.array(item.drop(columns=['filename', 'frame_num']))[np.newaxis,...] for item in feature_sequences]
        labels_sequences=[np.array(item[self.class_label])[np.newaxis,...] for item in labels_sequences]
        # concat it
        feature_sequences=np.concatenate(feature_sequences, axis=0)
        labels_sequences = np.concatenate(labels_sequences, axis=0)
        """# concat it
        feature_sequences=pd.concat(feature_sequences, ignore_index=True)
        labels_sequences = pd.concat(labels_sequences, ignore_index=True)
        # take np.ndarray
        feature_sequences=np.array(feature_sequences.drop(columns=['filename', 'frame_num']))
        labels_sequences = np.array(labels_sequences[self.class_label]).reshape((-1,1))"""
        return feature_sequences, labels_sequences


    def __getitem__(self, index) -> Tuple[np.ndarray, np.ndarray]:
        feature_sequences, labels_sequences = self._load_and_preprocess_batch(index)
        # turn into sequence-to-one labeling
        labels_sequences=labels_sequences.squeeze()
        labels_sequences=self._sequence_to_one_transformation(labels_sequences)
        # one-hot encoding
        labels_sequences = self._one_hot_encoding(labels_sequences)
        return feature_sequences, labels_sequences

    def __len__(self) -> int:
        num_batches = int(np.ceil(len(self.feature_sequences) / self.batch_size))
        return num_batches





if __name__=="__main__":
    # params
    path_to_train = r'C:\Databases\DAiSEE\train_preprocessed'
    path_to_train_labels = r'C:\Databases\DAiSEE\Labels\TrainLabels.csv'
    path_to_dev = r'C:\Databases\DAiSEE\dev_preprocessed'
    path_to_dev_labels = r'C:\Databases\DAiSEE\Labels\ValidationLabels.csv'
    path_to_test = r'C:\Databases\DAiSEEE\test_preprocessed'
    path_to_test_labels = r'C:\Databases\DAiSEE\Labels\TestLabels.csv'

    path_to_save_model_and_results = '../results'

    num_classes = 4
    batch_size = 64
    epochs = 30
    highest_lr = 0.0005
    lowest_lr = 0.00001
    momentum = 0.9
    weighting_beta = 0.99
    focal_loss_gamma = 2
    output_path = 'results'
    # create output path
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    optimizer = tf.keras.optimizers.Adam(highest_lr, clipnorm=1.)
    # loading features and labels

    #FAU_train=pd.read_csv(os.path.join(path_to_train, "FAU_features_with_labels.csv")).drop(columns=['Unnamed: 0'])
    FAU_dev = pd.read_csv(os.path.join(path_to_dev, "FAU_features_with_labels.csv")).drop(columns=['Unnamed: 0'])
    #FAU_test = pd.read_csv(os.path.join(path_to_test, "FAU_features_with_labels.csv")).drop(columns=['Unnamed: 0'])
    # labels
    #labels_train = FAU_train[["filename", "engagement", "boredom", "frustration", "confusion"]].copy()
    #labels_train=labels_train.dropna()
    labels_dev = FAU_dev[["filename", "engagement", "boredom", "frustration", "confusion"]].copy()
    labels_dev = labels_dev.dropna()
    #labels_test = FAU_test[["filename", "engagement", "boredom", "frustration", "confusion"]].copy()
    #labels_test = labels_test.dropna()
    # EMOVGGFace2 embeddings
    #EMO_deep_emb_train=pd.read_csv(os.path.join(path_to_train,"deep_embeddings_from_EMOVGGFace2.csv" ))
    EMO_deep_emb_dev = pd.read_csv(os.path.join(path_to_dev, "deep_embeddings_from_EMOVGGFace2.csv"))
    #EMO_deep_emb_test = pd.read_csv(os.path.join(path_to_test, "deep_embeddings_from_EMOVGGFace2.csv"))

    # AttVGGFace2 embeddings
    #Att_deep_emb_train=pd.read_csv(os.path.join(path_to_train,"deep_embeddings_from_AttVGGFace2.csv" ))
    Att_deep_emb_dev = pd.read_csv(os.path.join(path_to_dev, "deep_embeddings_from_AttVGGFace2.csv"))
    #Att_deep_emb_test = pd.read_csv(os.path.join(path_to_test, "deep_embeddings_from_AttVGGFace2.csv"))
    print('loaded')
    # split filenames to fit FAU filenames
    #EMO_deep_emb_train['filename'] = EMO_deep_emb_train['filename'].apply(lambda x: x.split('\\')[-1].split('.')[0])
    EMO_deep_emb_dev['filename'] = EMO_deep_emb_dev['filename'].apply(lambda x: x.split('\\')[-1].split('.')[0])
    #EMO_deep_emb_test['filename'] = EMO_deep_emb_test['filename'].apply(lambda x: x.split('\\')[-1].split('.')[0])
    #Att_deep_emb_train['filename'] = Att_deep_emb_train['filename'].apply(lambda x: x.split('\\')[-1].split('.')[0])
    Att_deep_emb_dev['filename'] = Att_deep_emb_dev['filename'].apply(lambda x: x.split('\\')[-1].split('.')[0])
    #Att_deep_emb_test['filename'] = Att_deep_emb_test['filename'].apply(lambda x: x.split('\\')[-1].split('.')[0])

    # prepare features for concatenation
    #FAU_train.set_index('filename', inplace=True)
    FAU_dev.set_index('filename', inplace=True)
    #FAU_test.set_index('filename', inplace=True)
    #EMO_deep_emb_train.set_index('filename', inplace=True)
    EMO_deep_emb_dev.set_index('filename', inplace=True)
    #EMO_deep_emb_test.set_index('filename', inplace=True)
    #Att_deep_emb_train.set_index('filename', inplace=True)
    Att_deep_emb_dev.set_index('filename', inplace=True)
    #Att_deep_emb_test.set_index('filename', inplace=True)

    # concatenate it
    #concatenated_train=pd.merge(FAU_train, EMO_deep_emb_train,
    #                            left_index=True, right_index=True).merge(Att_deep_emb_train, left_index=True, right_index=True)
    concatenated_dev=pd.merge(FAU_dev, EMO_deep_emb_dev,
                                left_index=True, right_index=True).merge(Att_deep_emb_dev, left_index=True, right_index=True)
    #concatenated_test=pd.merge(FAU_test, EMO_deep_emb_test,
    #                            left_index=True, right_index=True).merge(Att_deep_emb_test, left_index=True, right_index=True)
    print('merged')
    # prepare labels
    #labels_train.set_index('filename', inplace=True)
    labels_dev.set_index('filename', inplace=True)
    #labels_test.set_index('filename', inplace=True)

    # delete rows which is not in labels
    #concatenated_train=concatenated_train.iloc[concatenated_train.index.isin(labels_train.index)]
    concatenated_dev = concatenated_dev.iloc[concatenated_dev.index.isin(labels_dev.index)]
    #concatenated_test = concatenated_test.iloc[concatenated_test.index.isin(labels_test.index)]


    # sort features and labels to make them in the same order
    #labels_train.sort_index(inplace=True)
    labels_dev.sort_index(inplace=True)
    #labels_test.sort_index(inplace=True)
    #concatenated_train.sort_index(inplace=True)
    concatenated_dev.sort_index(inplace=True)
    #concatenated_test.sort_index(inplace=True)

    # prepare dataframes before loader
    concatenated_dev=concatenated_dev.reset_index()
    labels_dev=labels_dev.reset_index()
    concatenated_dev=concatenated_dev.iloc[:10000]
    labels_dev = labels_dev.iloc[:10000]


    dev_gen=SequenceLoader(features=concatenated_dev,labels=labels_dev, batch_size=batch_size,
                 num_classes=num_classes, num_frames_in_seq=20,
                 proportion_of_intersection=0.5, class_label=['engagement'],scaling=True)


    # one-hot encoding for labels
    labels_dev=tf.keras.utils.to_categorical(np.array(labels_dev['engagement']).reshape((-1,)))

    model=tf.keras.Sequential()
    model.add(tf.keras.layers.LSTM(100, input_shape=(20, 1575,), activation='relu'))
    model.add(tf.keras.layers.Dense(4, activation='softmax'))
    model.compile(optimizer="Adam", loss='categorical_crossentropy', metrics=[tf.keras.metrics.CategoricalAccuracy()])
    model.fit(dev_gen, epochs=100)


