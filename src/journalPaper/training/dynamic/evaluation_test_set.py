import gc
import sys
from typing import Optional

from pytorch_utils.models.CNN_models import Modified_EfficientNet_B1
from src.journalPaper.training.dynamic import training_config
from src.journalPaper.training.dynamic.data_preparation import load_data_and_construct_dataloaders

sys.path.append('/nfs/home/ddresvya/scripts/EngagementRecognition/')
sys.path.append('/nfs/home/ddresvya/scripts/datatools/')
sys.path.append('/nfs/home/ddresvya/scripts/simple-HRNet-master/')

import os.path
import pandas as pd
import torch
import wandb
from torchinfo import summary

from pytorch_utils.models.Pose_estimation.HRNet import Modified_HRNet
from src.journalPaper.training.dynamic.model_evaluation import evaluate_model, draw_confusion_matrix
from src.journalPaper.training.dynamic.models.unimodal_engagement_recognition_model import \
    Pose_engagement_recognition_model, Facial_engagement_recognition_model


def construct_model_seq2one_facial(base_model: torch.nn.Module, cut_n_last_layers: int, num_classes: int,
                    num_timesteps: int, pretrained: Optional[str] = None) -> torch.nn.Module:
    if pretrained is not None:
        base_model.load_state_dict(torch.load(pretrained))
    # cut off last layers
    base_model = torch.nn.Sequential(*list(base_model.children())[:cut_n_last_layers])
    # freeze base_model
    for param in base_model.parameters():
        param.requires_grad = False
    # construct sequence_to_one model
    model = Facial_engagement_recognition_model(facial_model=base_model,
                                                embeddings_layer_neurons=256,
                                                num_classes=num_classes,
                                                transformer_num_heads=4,
                                                num_timesteps=num_timesteps)
    return model

def construct_model_seq2one_pose(base_model: torch.nn.Module, cut_n_last_layers: int, num_classes: int,
                    num_timesteps: int, pretrained: Optional[str] = None) -> torch.nn.Module:
    if pretrained is not None:
        base_model.load_state_dict(torch.load(pretrained))
    # cut off last layers
    base_model.classifier = torch.nn.Identity()
    # freeze base_model
    for param in base_model.parameters():
        param.requires_grad = False
    # construct sequence_to_one model
    model = Pose_engagement_recognition_model(pose_model=base_model,
                                                embeddings_layer_neurons=256,
                                                num_classes=num_classes,
                                                transformer_num_heads=4,
                                                num_timesteps=num_timesteps)
    return model



def test_model(model: torch.nn.Module, generator: torch.utils.data.DataLoader, device: torch.device,
               metrics_name_prefix: str = 'test_', print_metrics: bool = True,
               draw_cm:Optional[bool]=False, output_path_cm:Optional[str]=None):
    test_metrics = evaluate_model(model, generator, device,
                                  metrics_name_prefix=metrics_name_prefix, print_metrics=print_metrics)
    # draw confusion matrix if needed
    if draw_cm:
        draw_confusion_matrix(model=model, generator=generator, device=device, output_path = output_path_cm)
    return test_metrics


def get_info_and_download_models_weights_from_project(entity: str, project_name: str, output_path: str) -> pd.DataFrame:
    """ Extracts info about run models from the project and downloads the models weights to the output_path.
        The extracted information will be stored as pd.DataFrame with the columns:
        ['ID', 'model_type', 'window_size', 'stride' 'loss_multiplication_factor', 'best_val_recall']

    :param entity: str
            The name of the WandB entity. (usually account name)
    :param project_name: str
            The name of the WandB project.
    :param output_path: str
            The path to the folder where the models weights will be downloaded.
    :return: pd.DataFrame
            The extracted information about the models.
    """
    # get api
    api = wandb.Api()
    # establish the entity and project name
    entity, project = entity, project_name
    # get runs from the project
    runs = api.runs(f"{entity}/{project}")
    # extract info about the runs
    info = pd.DataFrame(columns=['ID', 'model_type', 'window_size', 'stride',
                                 'loss_multiplication_factor', 'best_val_recall'])
    for run in runs:
        ID = run.name
        model_type = run.config['MODEL_TYPE']
        window_size = run.config['window_size']
        stride = run.config['stride']
        loss_multiplication_factor = run.config['loss_multiplication_factor']
        best_val_recall = run.config['best_val_recall_classification']
        info = pd.concat([info,
                          pd.DataFrame.from_dict(
                              {'ID': [ID], 'model_type': [model_type],
                               'window_size': [window_size],
                               'stride': [stride],
                               'loss_multiplication_factor': [loss_multiplication_factor],
                               'best_val_recall': [best_val_recall]}
                          )
                          ]
                         )
        # download the model weights
        final_output_path = os.path.join(output_path, ID)
        run.file('best_model_recall.pth').download(final_output_path, replace=True)
        # move the file out of dir and rename file for convenience
        os.rename(os.path.join(final_output_path, 'best_model_recall.pth'),
                  final_output_path + '.pth')
        # delete the dir
        os.rmdir(final_output_path)

    return info


if __name__ == "__main__":
    # params
    batch_size = 16
    project_name = 'engagement_recognition_seq2one'
    entity = 'denisdresvyanskiy'
    output_path_for_models_weights = "/" + os.path.join(*os.path.abspath(__file__).split(os.path.sep)[:-6],
                                                        'weights_best_models/sequence_to_one/')
    tested_model_type = 'EfficientNet-B1' # this is a shortcut, since i can change the DATA_TYPE and PATH_TO_DATA variables
    # in the training_config.py only manually and before the start of the script. Of course, it can all these scripts
    # can be rewritten in a more convenient way, but, unfortunately, I don't have time for this right now.
    # so, to run this script, you need to change both the tested_model_type and the DATA_TYPE and PATH_TO_DATA variables
    # in the training_config.py file manually.

    if not os.path.exists(output_path_for_models_weights):
        os.makedirs(output_path_for_models_weights)

    # get info about all runs (training sessions) and download the models weights
    info = get_info_and_download_models_weights_from_project(entity=entity, project_name=project_name,
                                                             output_path=output_path_for_models_weights)

    # test all models on the test set
    info['test_accuracy'] = -100
    info['test_precision'] = -100
    info['test_recall'] = -100
    info['test_f1'] = -100
    info.reset_index(drop=True, inplace=True)
    for i in range(len(info)):
        print("Testing model %d / %s" % (i + 1, info['model_type'].iloc[i]))
        # get model type
        model_type = info['model_type'].iloc[i]
        if model_type != tested_model_type:
            continue
        # create data loaded based on the model type
        # TODO: with current implementation, you would need to change every time the DATA_TYPE in the training_config.py
        #  file to the corresponding data type of the model you are testing. This is, of course, not very convenient, and
        #  will not work if you want to test models with different data types at the same time (the case I have right now)
        #  You need to change it somehow.
        train_generator, dev_generator, test_generator = load_data_and_construct_dataloaders(model_type=model_type,
                                        batch_size=batch_size,
                                        window_size=info['window_size'].iloc[i], stride=info['stride'].iloc[i],
                                        consider_timestamps=True,
                                        return_class_weights = False,
                                        need_test_set=True)
        # get one iteration of train generator to get sequence length
        inputs, labels = next(iter(train_generator))
        sequence_length = inputs.shape[1]

        if model_type == "Modified_HRNet":
            model = Modified_HRNet(pretrained=True,
                                   path_to_weights=training_config.PATH_TO_WEIGHTS_HRNET,
                                   embeddings_layer_neurons=256, num_classes=training_config.NUM_CLASSES,
                                   num_regression_neurons=None,
                                   consider_only_upper_body=True)
            model = construct_model_seq2one_pose(base_model=model, cut_n_last_layers=-1, num_classes=training_config.NUM_CLASSES,
                                    num_timesteps=sequence_length, pretrained=None)

        elif model_type == "EfficientNet-B1":
            model = Modified_EfficientNet_B1(embeddings_layer_neurons=256, num_classes=training_config.NUM_CLASSES,
                                             num_regression_neurons=None)
            model = construct_model_seq2one_facial(base_model=model, cut_n_last_layers=-1, num_classes=training_config.NUM_CLASSES,
                                    num_timesteps=sequence_length, pretrained=None)
        else:
            raise ValueError("Unknown model type: %s" % model_type)

        # load model weights
        path_to_weights = os.path.join(output_path_for_models_weights, info['ID'].iloc[i] + '.pth')
        model.load_state_dict(torch.load(path_to_weights))
        # define device
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = model.to(device)

        # test model
        test_metrics = evaluate_model(model, test_generator, device, metrics_name_prefix='test_', print_metrics=True)
        # draw confusion matrix
        if not os.path.exists(os.path.join(output_path_for_models_weights, 'confusion_matrices')):
            os.makedirs(os.path.join(output_path_for_models_weights, 'confusion_matrices'))
        draw_confusion_matrix(model=model, generator=test_generator, device=device,
                              output_path = os.path.join(output_path_for_models_weights, 'confusion_matrices'),
                              filename = info['ID'].iloc[i] + '.png')

        # save test metrics
        info.loc[i, 'test_accuracy'] = test_metrics['test_accuracy_classification']
        info.loc[i, 'test_precision'] = test_metrics['test_precision_classification']
        info.loc[i, 'test_recall'] = test_metrics['test_recall_classification']
        info.loc[i, 'test_f1'] = test_metrics['test_f1_classification']

        # save info
        info.to_csv(os.path.join(output_path_for_models_weights, 'info.csv'), index=False)

        # clear RAM and GPU memory
        del model, train_generator, dev_generator, test_generator
        torch.cuda.empty_cache()
        gc.collect()
