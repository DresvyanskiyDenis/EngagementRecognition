import sys
sys.path.append("/nfs/home/ddresvya/scripts/EngagementRecognition/")
sys.path.append("/nfs/home/ddresvya/scripts/datatools/")
sys.path.append("/nfs/home/ddresvya/scripts/simple-HRNet-master/")


from functools import partial
from typing import List, Callable

import torch
import torchvision.transforms as T
import numpy as np
import pandas as pd
import os


from pytorch_utils.data_preprocessing import convert_image_to_float_and_scale
from pytorch_utils.models.CNN_models import Modified_EfficientNet_B1, Modified_EfficientNet_B4
from pytorch_utils.models.Pose_estimation.HRNet import Modified_HRNet
from pytorch_utils.models.input_preprocessing import resize_image_saving_aspect_ratio, EfficientNet_image_preprocessor
from feature_extraction.pytorch_based.embeddings_extraction_torch import EmbeddingsExtractor


def load_embeddings_extractor_affective_model(weights_path: str) -> torch.nn.Module:
    """ Creates and loads the emotion recognition model for embeddings extraction. To do so, the cutting off some last
    layers of the model is required.
    """
    model = Modified_EfficientNet_B1(embeddings_layer_neurons=256, num_classes=8, num_regression_neurons=2)
    model.load_state_dict(torch.load(weights_path))
    # cut off last two layers responsible for classification and regression
    model = torch.nn.Sequential(*list(model.children())[:-2])
    # freeze model
    for param in model.parameters():
        param.requires_grad = False
    # set model to evaluation mode
    model.eval()
    return model




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
                               path_to_weights="/nfs/home/ddresvya/scripts/simple-HRNet-master/pose_hrnet_w32_256x192.pth",
                               embeddings_layer_neurons=256, num_classes=3,
                               num_regression_neurons=None,
                               consider_only_upper_body=True)
    elif model_type == 'affective':
        model = load_embeddings_extractor_affective_model(path_to_weights)
        return model
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
    elif model_type == 'affective':
        preprocessing_functions = [partial(resize_image_saving_aspect_ratio, expected_size=240),
                                   EfficientNet_image_preprocessor()]
    else:
        raise ValueError(f'The model type should be either "EfficientNet-B1", "EfficientNet-B4", "Modified_HRNet", or "affective".'
                         f'Got {model_type} instead.')
    return preprocessing_functions

def main():
    # params
    output_path = '/nfs/scratch/ddresvya/NoXi/NoXi/prepared_data/embeddings/static/'
    if os.path.exists(output_path) is False:
        os.makedirs(output_path, exist_ok=True)
    # load paths with labels
    train_face = pd.read_csv("/nfs/scratch/ddresvya/NoXi/NoXi/prepared_data/aligned_labels_train_face.csv").rename(columns={"path_to_frame": "path"})
    dev_face = pd.read_csv("/nfs/scratch/ddresvya/NoXi/NoXi/prepared_data/aligned_labels_dev_face.csv").rename(columns={"path_to_frame": "path"})
    train_pose = pd.read_csv("/nfs/scratch/ddresvya/NoXi/NoXi/prepared_data/aligned_labels_train_pose.csv").rename(columns={"path_to_frame": "path"})
    dev_pose = pd.read_csv("/nfs/scratch/ddresvya/NoXi/NoXi/prepared_data/aligned_labels_dev_pose.csv").rename(columns={"path_to_frame": "path"})
    train_affective = pd.read_csv("/nfs/scratch/ddresvya/NoXi/NoXi/prepared_data/aligned_labels_train_face.csv").rename(columns={"path_to_frame": "path"})
    dev_affective = pd.read_csv("/nfs/scratch/ddresvya/NoXi/NoXi/prepared_data/aligned_labels_dev_face.csv").rename(columns={"path_to_frame": "path"})

    # extract embeddings (facial)
        # create and load model
    model_weights = "/nfs/home/ddresvya/scripts/weights_best_models/Engagement/static/facial_engagement_static_efficientNet_b1.pth"
    model_type = 'EfficientNet-B1'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = create_and_load_model(model_type=model_type, path_to_weights=model_weights)
    model = model.to(device)
        # get preprocessing functions
    preprocessing_functions = get_preprocessing_functions(model_type=model_type)
    extractor = EmbeddingsExtractor(model=model, device=device, preprocessing_functions=preprocessing_functions,
                                    output_shape=256)
    extractor.extract_embeddings(data=train_face, batch_size=64, num_workers=8, labels_columns=["timestep","engagement"],
                                 output_path=os.path.join(output_path, f"NoXi_face_embeddings_train.csv"),
                                 verbose=True)
    extractor.extract_embeddings(data=dev_face, batch_size=64, num_workers=8, labels_columns=["timestep","engagement"],
                                    output_path=os.path.join(output_path, f"NoXi_face_embeddings_dev.csv"),
                                    verbose=True)

    # extract embeddings (pose)
        # create and load model
    model_weights = "/nfs/home/ddresvya/scripts/weights_best_models/Engagement/static/kinesics_engagement_static_hrnet.pth"
    model_type = 'Modified_HRNet'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = create_and_load_model(model_type=model_type, path_to_weights=model_weights)
    model = model.to(device)
        # get preprocessing functions
    preprocessing_functions = get_preprocessing_functions(model_type=model_type)
    extractor = EmbeddingsExtractor(model=model, device=device, preprocessing_functions=preprocessing_functions,
                                    output_shape=256)
    extractor.extract_embeddings(data=train_pose, batch_size=64, num_workers=8, labels_columns=["timestep","engagement"],
                                    output_path=os.path.join(output_path, f"NoXi_pose_embeddings_train.csv"),
                                    verbose=True)
    extractor.extract_embeddings(data=dev_pose, batch_size=64, num_workers=8, labels_columns=["timestep","engagement"],
                                    output_path=os.path.join(output_path, f"NoXi_pose_embeddings_dev.csv"),
                                    verbose=True)

    # extract embeddings (affective)
        # create and load model
    model_weights = "/nfs/home/ddresvya/scripts/weights_best_models/Engagement/static/affective_static_efficientNet_b1.pth"
    model_type = 'affective'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = create_and_load_model(model_type=model_type, path_to_weights=model_weights)
    model = model.to(device)
        # get preprocessing functions
    preprocessing_functions = get_preprocessing_functions(model_type=model_type)
    extractor = EmbeddingsExtractor(model=model, device=device, preprocessing_functions=preprocessing_functions,
                                    output_shape=256)
    extractor.extract_embeddings(data=train_affective, batch_size=64, num_workers=8, labels_columns=["timestep","engagement"],
                                    output_path=os.path.join(output_path, f"NoXi_affective_embeddings_train.csv"),
                                    verbose=True)
    extractor.extract_embeddings(data=dev_affective, batch_size=64, num_workers=8, labels_columns=["timestep","engagement"],
                                    output_path=os.path.join(output_path, f"NoXi_affective_embeddings_dev.csv"),
                                    verbose=True)



if __name__=="__main__":
    main()
