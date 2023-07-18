import sys

sys.path.append("/work/home/dsu/engagement_recognition_project_server/")
sys.path.append("/work/home/dsu/datatools/")
sys.path.append("/work/home/dsu/simple-HRNet-master/")


from functools import partial
from typing import List, Callable

import torch
import torchvision.transforms as T
import numpy as np
import pandas as pd
import os


from feature_extraction.embeddings_extraction_torch import EmbeddingsExtractor
from pytorch_utils.data_preprocessing import convert_image_to_float_and_scale
from pytorch_utils.models.CNN_models import Modified_EfficientNet_B1, Modified_EfficientNet_B4
from pytorch_utils.models.Pose_estimation.HRNet import Modified_HRNet
from pytorch_utils.models.input_preprocessing import resize_image_saving_aspect_ratio, EfficientNet_image_preprocessor
from src.journalPaper.feature_extraction.data_loader import load_NoXi_and_DAiSEE_dataframes


def create_and_load_model(model_type: str, path_to_weights: str) -> torch.nn.Module:
    """Create and load model from weights

    :param model_type: str
        type of model to create
    :param path_to_weights: str
        path to weights
    :return: torch.nn.Module
        PyTorch model
    """
    # create model
    if model_type == "EfficientNet-B1":
        model = Modified_EfficientNet_B1(embeddings_layer_neurons=256, num_classes=3,
                                         num_regression_neurons=None)
    elif model_type == "EfficientNet-B4":
        model = Modified_EfficientNet_B4(embeddings_layer_neurons=256, num_classes=3,
                                         num_regression_neurons=None)
    elif model_type == "Modified_HRNet":
        model = Modified_HRNet(pretrained=True,
                               path_to_weights="/work/home/dsu/simple-HRNet-master/pose_hrnet_w32_256x192.pth",
                               embeddings_layer_neurons=256, num_classes=3,
                               num_regression_neurons=None,
                               consider_only_upper_body=True)
    else:
        raise ValueError("Unknown model type: %s" % model_type)
    # load weights
    model.load_state_dict(torch.load(path_to_weights))
    # cut off last layer
    model.classifier = torch.nn.Identity()
    # freeze model
    for param in model.parameters():
        param.requires_grad = False
    # set model to evaluation mode
    model.eval()
    return model


def get_preprocessing_functions(model_type: str) -> List[Callable]:
    """Outputs preprocessing functions for a given model type

    :param model_type: str
        type of model
    :return: List[Callable]
        list of preprocessing functions
    """

    if model_type == 'EfficientNet-B1':
        preprocessing_functions = [partial(resize_image_saving_aspect_ratio, expected_size=240),
                                   EfficientNet_image_preprocessor()]
    elif model_type == 'EfficientNet-B4':
        preprocessing_functions = [partial(resize_image_saving_aspect_ratio, expected_size=380),
                                   EfficientNet_image_preprocessor()]
    elif model_type == 'Modified_HRNet':
        preprocessing_functions = [partial(resize_image_saving_aspect_ratio, expected_size=256),
                                   convert_image_to_float_and_scale,
                                   T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                   ]  # From HRNet
    else:
        raise ValueError(f'The model type should be either "EfficientNet-B1", "EfficientNet-B4", or "Modified_HRNet".'
                         f'Got {model_type} instead.')
    return preprocessing_functions


if __name__ == '__main__':
    # params
    model_weights = "/work/home/dsu/tmp/fresh-bush-43.pth"
    model_type = 'Modified_HRNet'
    data_type = 'pose'
    path_to_data_NoXi = f'/work/home/dsu/Datasets/NoXi/prepared_data/{data_type}'
    path_to_data_DAiSEE = f'/work/home/dsu/Datasets/DAiSEE/prepared_data/{data_type}'
    output_path = '/work/home/dsu/Datasets/Embeddings/'
    # create and load model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = create_and_load_model(model_type=model_type, path_to_weights=model_weights)
    model = model.to(device)
    # get preprocessing functions
    preprocessing_functions = get_preprocessing_functions(model_type=model_type)
    # load data
    train, dev, test = load_NoXi_and_DAiSEE_dataframes(path_to_data_NoXi=path_to_data_NoXi,
                                                       path_to_data_DAiSEE=path_to_data_DAiSEE, data_type=data_type)
    # separate dataframes into DAiSEE and NoXi
    DAiSEE_train = train[train['path'].str.contains('DAiSEE')]
    NoXi_train = train[train['path'].str.contains('NoXi')]
    DAiSEE_dev = dev[dev['path'].str.contains('DAiSEE')]
    NoXi_dev = dev[dev['path'].str.contains('NoXi')]
    DAiSEE_test = test[test['path'].str.contains('DAiSEE')]
    NoXi_test = test[test['path'].str.contains('NoXi')]
    # create feature extractor class
    extractor = EmbeddingsExtractor(model=model, device=device, preprocessing_functions=preprocessing_functions,
                                    output_shape = 256)
    # extract embeddings NoXi
    extractor.extract_embeddings(data=NoXi_train, batch_size=64, num_workers=8,
                                 output_path=os.path.join(output_path, f"NoXi_{data_type}_embeddings_train.csv"), verbose=True)
    extractor.extract_embeddings(data=NoXi_dev, batch_size=64, num_workers=8,
                                 output_path=os.path.join(output_path, f"NoXi_{data_type}_embeddings_dev.csv"), verbose=True)
    extractor.extract_embeddings(data=NoXi_test, batch_size=64, num_workers=8,
                                    output_path=os.path.join(output_path, f"NoXi_{data_type}_embeddings_test.csv"), verbose=True)
    # extract embeddings DAiSEE
    extractor.extract_embeddings(data=DAiSEE_train, batch_size=64, num_workers=8,
                                    output_path=os.path.join(output_path, f"DAiSEE_{data_type}_embeddings_train.csv"), verbose=True)
    extractor.extract_embeddings(data=DAiSEE_dev, batch_size=64, num_workers=8,
                                    output_path=os.path.join(output_path, f"DAiSEE_{data_type}_embeddings_dev.csv"), verbose=True)
    extractor.extract_embeddings(data=DAiSEE_test, batch_size=64, num_workers=8,
                                    output_path=os.path.join(output_path, f"DAiSEE_{data_type}_embeddings_test.csv"), verbose=True)



