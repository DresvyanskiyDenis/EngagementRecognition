import os

import tensorflow as tf
import numpy as np
import pandas as pd





if __name__=="__main__":
    # params
    path_to_train = r'E:\Databases\DAiSEE\DAiSEE\train_preprocessed'
    path_to_train_labels = r'E:\Databases\DAiSEE\DAiSEE\Labels\TrainLabels.csv'
    path_to_dev = r'E:\Databases\DAiSEE\DAiSEE\dev_preprocessed'
    path_to_dev_labels = r'E:\Databases\DAiSEE\DAiSEE\Labels\ValidationLabels.csv'
    path_to_test = r'E:\Databases\DAiSEE\DAiSEE\test_preprocessed'
    path_to_test_labels = r'E:\Databases\DAiSEE\DAiSEE\Labels\TestLabels.csv'

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

    FAU_train=pd.read_csv(os.path.join(path_to_train, "FAU_features_with_labels.csv"))
    FAU_dev = pd.read_csv(os.path.join(path_to_dev, "FAU_features_with_labels.csv"))
    FAU_test = pd.read_csv(os.path.join(path_to_test, "FAU_features_with_labels.csv"))
    # labels
    labels_train = FAU_train[["filename", "engagement", "boredom", "frustration", "confusion"]].copy()
    labels_train=labels_train.dropna()
    labels_dev = FAU_dev[["filename", "engagement", "boredom", "frustration", "confusion"]].copy()
    labels_dev = labels_dev.dropna()
    labels_test = FAU_test[["filename", "engagement", "boredom", "frustration", "confusion"]].copy()
    labels_test = labels_test.dropna()
    # EMOVGGFace2 embeddings
    EMO_deep_emb_train=pd.read_csv(os.path.join(path_to_train,"deep_embeddings_from_EMOVGGFace2.csv" ))
    EMO_deep_emb_dev = pd.read_csv(os.path.join(path_to_dev, "deep_embeddings_from_EMOVGGFace2.csv"))
    EMO_deep_emb_test = pd.read_csv(os.path.join(path_to_test, "deep_embeddings_from_EMOVGGFace2.csv"))

    # AttVGGFace2 embeddings
    Att_deep_emb_train=pd.read_csv(os.path.join(path_to_train,"deep_embeddings_from_AttVGGFace2.csv" ))
    Att_deep_emb_dev = pd.read_csv(os.path.join(path_to_dev, "deep_embeddings_from_AttVGGFace2.csv"))
    Att_deep_emb_test = pd.read_csv(os.path.join(path_to_test, "deep_embeddings_from_AttVGGFace2.csv"))
    print('loaded')
    # split filenames to fit FAU filenames
    EMO_deep_emb_train['filename'] = EMO_deep_emb_train['filename'].apply(lambda x: x.split('\\')[-1].split('.')[0])
    EMO_deep_emb_dev['filename'] = EMO_deep_emb_dev['filename'].apply(lambda x: x.split('\\')[-1].split('.')[0])
    EMO_deep_emb_test['filename'] = EMO_deep_emb_test['filename'].apply(lambda x: x.split('\\')[-1].split('.')[0])
    Att_deep_emb_train['filename'] = Att_deep_emb_train['filename'].apply(lambda x: x.split('\\')[-1].split('.')[0])
    Att_deep_emb_dev['filename'] = Att_deep_emb_dev['filename'].apply(lambda x: x.split('\\')[-1].split('.')[0])
    Att_deep_emb_test['filename'] = Att_deep_emb_test['filename'].apply(lambda x: x.split('\\')[-1].split('.')[0])

    # prepare features for concatenation
    FAU_train.set_index('filename', inplace=True)
    FAU_dev.set_index('filename', inplace=True)
    FAU_test.set_index('filename', inplace=True)
    EMO_deep_emb_train.set_index('filename', inplace=True)
    EMO_deep_emb_dev.set_index('filename', inplace=True)
    EMO_deep_emb_test.set_index('filename', inplace=True)
    Att_deep_emb_train.set_index('filename', inplace=True)
    Att_deep_emb_dev.set_index('filename', inplace=True)
    Att_deep_emb_test.set_index('filename', inplace=True)

    # concatenate it
    concatenated_train=pd.merge(FAU_train, EMO_deep_emb_train,
                                left_index=True, right_index=True).merge(Att_deep_emb_train, left_index=True, right_index=True)
    concatenated_dev=pd.merge(FAU_dev, EMO_deep_emb_dev,
                                left_index=True, right_index=True).merge(Att_deep_emb_dev, left_index=True, right_index=True)
    concatenated_test=pd.merge(FAU_test, EMO_deep_emb_test,
                                left_index=True, right_index=True).merge(Att_deep_emb_test, left_index=True, right_index=True)
    print('merged')
    # prepare labels
    labels_train.set_index('filename', inplace=True)
    labels_dev.set_index('filename', inplace=True)
    labels_test.set_index('filename', inplace=True)

    # delete rows which is not in labels
    concatenated_train=concatenated_train.iloc[concatenated_train.index.isin(labels_train.index)]
    concatenated_dev = concatenated_dev.iloc[concatenated_dev.index.isin(labels_dev.index)]
    concatenated_test = concatenated_test.iloc[concatenated_test.index.isin(labels_test.index)]


    # sort features and labels to make them in the same order
    labels_train.sort_index(inplace=True)
    labels_dev.sort_index(inplace=True)
    labels_test.sort_index(inplace=True)
    concatenated_train.sort_index(inplace=True)
    concatenated_dev.sort_index(inplace=True)
    concatenated_test.sort_index(inplace=True)

    # one-hot encoding for labels
    labels_dev=tf.keras.utils.to_categorical(np.array(labels_dev['engagement']).reshape((-1,)))

    model=tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(100, input_shape=(concatenated_dev.shape[1],), activation='relu'))
    model.add(tf.keras.layers.Dense(4, activation='softmax'))
    model.compile(optimizer="Adam", loss='categorical_crossentropy', metrics=[tf.keras.metrics.CategoricalAccuracy()])
    model.fit(concatenated_train, labels_train, epochs=100, validation_data=(concatenated_dev, labels_dev))


