import gc
import sys
from typing import Optional

sys.path.append('/nfs/home/ddresvya/scripts/EngagementRecognition/')
sys.path.append('/nfs/home/ddresvya/scripts/datatools/')
sys.path.append('/nfs/home/ddresvya/scripts/simple-HRNet-master/')

import os.path

import pandas as pd
import torch
import wandb
from torchinfo import summary

from pytorch_utils.models.CNN_models import Modified_EfficientNet_B1, Modified_EfficientNet_B4
from src.journalPaper.training.static import training_config
from src.journalPaper.training.static.data_preparation import load_data_and_construct_dataloaders
from src.journalPaper.training.static.model_evaluation import evaluate_model, draw_confusion_matrix


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
        ['ID', 'model_type', 'discriminative_learning', 'gradual_unfreezing', 'loss_multiplication_factor', 'best_val_recall']

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
    info = pd.DataFrame(columns=['ID', 'model_type', 'discriminative_learning', 'gradual_unfreezing',
                                 'loss_multiplication_factor', 'best_val_recall'])
    for run in runs:
        ID = run.name
        model_type = run.config['MODEL_TYPE']
        discriminative_learning = run.config['DISCRIMINATIVE_LEARNING']
        gradual_unfreezing = run.config['GRADUAL_UNFREEZING']
        loss_multiplication_factor = run.config['loss_multiplication_factor']
        best_val_recall = run.config['best_val_recall_classification']
        info = pd.concat([info,
                          pd.DataFrame.from_dict(
                              {'ID': [ID], 'model_type': [model_type],
                               'discriminative_learning': [discriminative_learning],
                               'gradual_unfreezing': [gradual_unfreezing],
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
    batch_size = 32
    project_name = 'Engagement_recognition_F2F'
    entity = 'denisdresvyanskiy'
    output_path_for_models_weights = "/" + os.path.join(*os.path.abspath(__file__).split(os.path.sep)[:-6],
                                                        'weights_best_models/')

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
        # create model
        model_type = info['model_type'].iloc[i]
        if model_type == "EfficientNet-B1":
            model = Modified_EfficientNet_B1(embeddings_layer_neurons=256, num_classes=training_config.NUM_CLASSES,
                                             num_regression_neurons=None)
        elif model_type == "EfficientNet-B4":
            model = Modified_EfficientNet_B4(embeddings_layer_neurons=256, num_classes=training_config.NUM_CLASSES,
                                             num_regression_neurons=None)
        else:
            raise ValueError("Unknown model type: %s" % model_type)
        # load model weights
        path_to_weights = os.path.join(output_path_for_models_weights, info['ID'].iloc[i] + '.pth')
        model.load_state_dict(torch.load(path_to_weights))
        # define device
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        # get generators, including test generator
        (train_generator, dev_generator, test_generator), class_weights = load_data_and_construct_dataloaders(
            path_to_data_NoXi=training_config.NOXI_DATA_PATH,
            path_to_data_DAiSEE=training_config.DAISEE_DATA_PATH,
            model_type=model_type,
            batch_size=batch_size,
            return_class_weights=True)

        # test model
        test_metrics = evaluate_model(model, test_generator, device, metrics_name_prefix='test_', print_metrics=True)
        # draw confusion matrix
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
        del model, train_generator, dev_generator, test_generator, class_weights
        torch.cuda.empty_cache()
        gc.collect()
