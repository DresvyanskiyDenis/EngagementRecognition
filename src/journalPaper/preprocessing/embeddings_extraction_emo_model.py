import os
from functools import partial

import pandas as pd
import torch

from feature_extraction.embeddings_extraction_torch import EmbeddingsExtractor
from pytorch_utils.models.CNN_models import Modified_EfficientNet_B1
from pytorch_utils.models.input_preprocessing import resize_image_saving_aspect_ratio, EfficientNet_image_preprocessor


def load_embeddings_extractor_model(weights_path: str) -> torch.nn.Module:
    """ Creates and loads the emotion recognition model for embeddings extraction. To do so, the cutting off some last
    layers of the model is required.
    """
    model = Modified_EfficientNet_B1(embeddings_layer_neurons=256, num_classes=8, num_regression_neurons=2)
    model.load_state_dict(torch.load(weights_path))
    # cut off last two layers responsible for classification and regression
    model = torch.nn.Sequential(*list(model.children())[:-2])
    return model


def extract_emo_embeddings_NoXi(extractor: EmbeddingsExtractor, output_path: str) -> None:
    path_to_train_file = "/nfs/home/ddresvya/Data/NoXi/prepared_data/faces/NoXi_face_train.csv"
    path_to_dev_file = "/nfs/home/ddresvya/Data/NoXi/prepared_data/faces/NoXi_face_dev.csv"
    path_to_test_file = "/nfs/home/ddresvya/Data/NoXi/prepared_data/faces/NoXi_face_test.csv"
    # check if output path exists
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    # load dataframes
    train_df = pd.read_csv(path_to_train_file)
    dev_df = pd.read_csv(path_to_dev_file)
    test_df = pd.read_csv(path_to_test_file)
    # clear from NaNs
    train_df = train_df.dropna()
    dev_df = dev_df.dropna()
    test_df = test_df.dropna()
    # change path_to_frame column name to path
    train_df = train_df.rename(columns={"path_to_frame": "path"})
    dev_df = dev_df.rename(columns={"path_to_frame": "path"})
    test_df = test_df.rename(columns={"path_to_frame": "path"})
    # map 5-class labels to 3-class labels for NoXi dataset. 5 classes were: highly disengaged, disengaged, neutral, engaged, highly engaged.
    # Now it would be 3 classes: disengaged, neutral, engaged
    # Remember that they are presented as one-hot vectors
    # we can add two first columns that represent disengagemend state (it would be then disengagement class in 3-class classification)
    # then, add two last columns that represent engagement state (it would be then engagement class in 3-class classification)
    # then, drop old columns and rename new columns to the template 'label_0', 'label_1', 'label_2'
    train_df['new_label_0'] = train_df['label_0'] + train_df['label_1']
    train_df['new_label_1'] = train_df['label_2']
    train_df['new_label_2'] = train_df['label_3'] + train_df['label_4']
    train_df = train_df.drop(columns=['label_0', 'label_1', 'label_2', 'label_3', 'label_4'])
    train_df = train_df.rename(
        columns={"new_label_0": "label_0", "new_label_1": "label_1", "new_label_2": "label_2"})

    dev_df['new_label_0'] = dev_df['label_0'] + dev_df['label_1']
    dev_df['new_label_1'] = dev_df['label_2']
    dev_df['new_label_2'] = dev_df['label_3'] + dev_df['label_4']
    dev_df = dev_df.drop(columns=['label_0', 'label_1', 'label_2', 'label_3', 'label_4'])
    dev_df = dev_df.rename(columns={"new_label_0": "label_0", "new_label_1": "label_1", "new_label_2": "label_2"})

    test_df['new_label_0'] = test_df['label_0'] + test_df['label_1']
    test_df['new_label_1'] = test_df['label_2']
    test_df['new_label_2'] = test_df['label_3'] + test_df['label_4']
    test_df = test_df.drop(columns=['label_0', 'label_1', 'label_2', 'label_3', 'label_4'])
    test_df = test_df.rename(columns={"new_label_0": "label_0", "new_label_1": "label_1", "new_label_2": "label_2"})
    # change paths from 'media/external_hdd_2/*/prepared_data/faces' to provided data paths
    # NoXi
    train_df['path'] = train_df['path'].apply(
        lambda x: x.replace('media/external_hdd_2/NoXi/prepared_data/faces/',
                            '/nfs/home/ddresvya/Data/NoXi/prepared_data/faces/'))
    dev_df['path'] = dev_df['path'].apply(
        lambda x: x.replace('media/external_hdd_2/NoXi/prepared_data/faces/',
                            '/nfs/home/ddresvya/Data/NoXi/prepared_data/faces/'))
    test_df['path'] = test_df['path'].apply(
        lambda x: x.replace('media/external_hdd_2/NoXi/prepared_data/faces/',
                            '/nfs/home/ddresvya/Data/NoXi/prepared_data/faces/'))
    # extract embeddings using the EmbeddingsExtractor function extract_embeddings()
    extractor.extract_embeddings(data=train_df, batch_size=64, num_workers=8,
                                 output_path=os.path.join(output_path, "NoXi_emo_embeddings_train.csv"), verbose=True)
    extractor.extract_embeddings(data=dev_df, batch_size=64, num_workers=8,
                                    output_path=os.path.join(output_path, "NoXi_emo_embeddings_dev.csv"), verbose=True)
    extractor.extract_embeddings(data=test_df, batch_size=64, num_workers=8,
                                    output_path=os.path.join(output_path, "NoXi_emo_embeddings_test.csv"), verbose=True)




def extract_emo_embeddings_DAiSEE(extractor: EmbeddingsExtractor, output_path: str) -> None:
    path_to_train_file = "/nfs/home/ddresvya/Data/DAiSEE/prepared_data/faces/DAiSEE_face_train_labels.csv"
    path_to_dev_file = "/nfs/home/ddresvya/Data/DAiSEE/prepared_data/faces/DAiSEE_face_dev_labels.csv"
    path_to_test_file = "/nfs/home/ddresvya/Data/DAiSEE/prepared_data/faces/DAiSEE_face_test_labels.csv"
    # check if output path exists
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    # load dataframes
    train_df = pd.read_csv(path_to_train_file)
    dev_df = pd.read_csv(path_to_dev_file)
    test_df = pd.read_csv(path_to_test_file)
    # clear from Nans
    train_df = train_df.dropna()
    dev_df = dev_df.dropna()
    test_df = test_df.dropna()
    # change path_to_frame column name to path
    train_df = train_df.rename(columns={"path_to_frame": "path"})
    dev_df = dev_df.rename(columns={"path_to_frame": "path"})
    test_df = test_df.rename(columns={"path_to_frame": "path"})
    # map 4-class labels to 3-class labels for DAiSEE. 4 classes were: highly disengaged, disengaged, engaged, highly engaged.
    # Now it would be 3 classes: disengaged, neutral, engaged
    # Remember that they are presented as one-hot vectors
    # we can simply add two columns that represent middle classes, while keep the other two columns as they are
    # then, drop old columns and rename new columns to the template 'label_0', 'label_1', 'label_2',
    train_df['new_label_0'] = train_df['label_0']
    train_df['new_label_1'] = train_df['label_1'] + train_df['label_2']
    train_df['new_label_2'] = train_df['label_3']
    train_df = train_df.drop(columns=['label_0', 'label_1', 'label_2', 'label_3'])
    train_df = train_df.rename(
        columns={"new_label_0": "label_0", "new_label_1": "label_1", "new_label_2": "label_2"})

    dev_df['new_label_0'] = dev_df['label_0']
    dev_df['new_label_1'] = dev_df['label_1'] + dev_df['label_2']
    dev_df['new_label_2'] = dev_df['label_3']
    dev_df = dev_df.drop(columns=['label_0', 'label_1', 'label_2', 'label_3'])
    dev_df = dev_df.rename(
        columns={"new_label_0": "label_0", "new_label_1": "label_1", "new_label_2": "label_2"})

    test_df['new_label_0'] = test_df['label_0']
    test_df['new_label_1'] = test_df['label_1'] + test_df['label_2']
    test_df['new_label_2'] = test_df['label_3']
    test_df = test_df.drop(columns=['label_0', 'label_1', 'label_2', 'label_3'])
    test_df = test_df.rename(
        columns={"new_label_0": "label_0", "new_label_1": "label_1", "new_label_2": "label_2"})
    # DAiSEE
    train_df['path'] = train_df['path'].apply(
        lambda x: x.replace('media/external_hdd_2/DAiSEE/prepared_data/faces/',
                            '/nfs/home/ddresvya/Data/DAiSEE/prepared_data/faces/'))
    dev_df['path'] = dev_df['path'].apply(
        lambda x: x.replace('media/external_hdd_2/DAiSEE/prepared_data/faces/',
                            '/nfs/home/ddresvya/Data/DAiSEE/prepared_data/faces/'))
    test_df['path'] = test_df['path'].apply(
        lambda x: x.replace('media/external_hdd_2/DAiSEE/prepared_data/faces/',
                            '/nfs/home/ddresvya/Data/DAiSEE/prepared_data/faces/'))
    # extract embeddings using the EmbeddingsExtractor function extract_embeddings()
    extractor.extract_embeddings(data=train_df, batch_size=64, num_workers=8,
                                 output_path=os.path.join(output_path, "DAiSEE_emo_embeddings_train.csv"), verbose=True)
    extractor.extract_embeddings(data=dev_df, batch_size=64, num_workers=8,
                                 output_path=os.path.join(output_path, "DAiSEE_emo_embeddings_dev.csv"), verbose=True)
    extractor.extract_embeddings(data=test_df, batch_size=64, num_workers=8,
                                 output_path=os.path.join(output_path, "DAiSEE_emo_embeddings_test.csv"), verbose=True)




def extract_emo_embeddings_MHHRI():
    pass


if __name__ == "__main__":
    # define preprocessing functions
    preprocessing_functions = [partial(resize_image_saving_aspect_ratio, expected_size=240),
                               EfficientNet_image_preprocessor()]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # load model
    path_to_weights = '/nfs/home/ddresvya/scripts/emotion_recognition_project/best_models/facial/radiant_fog_160.pth'
    output_path = '/nfs/home/ddresvya/Data/extracted_embeddings/'
    model = load_embeddings_extractor_model(path_to_weights)
    # create embeddings extractor
    extractor = EmbeddingsExtractor(model,device = device,
                 preprocessing_functions = preprocessing_functions)
    # extract embeddings
    extract_emo_embeddings_NoXi(extractor, os.path.join(output_path, "NoXi/face/emo/"))
    extract_emo_embeddings_DAiSEE(extractor, os.path.join(output_path, "DAiSEE/face/emo/"))
    #extract_emo_embeddings_MHHRI(extractor, os.path.join(output_path, "MHHRI/face/emo/"))



