import gc
from functools import partial
from typing import Tuple, Dict

import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error, \
    mean_absolute_error

from pytorch_utils.models.CNN_models import Modified_MobileNetV3_large
from src.training.facial import training_config
from src.training.facial.data_preparation import load_all_dataframes, resize_image_to_224_saving_aspect_ratio, \
    preprocess_image_MobileNetV3, construct_data_loaders


def validate_model_modified_MobileNetV3(model:torch.nn.Module, data_loader:torch.utils.data.DataLoader,
                                        device:torch.device)->Tuple[Dict[str,float],...]:
    """Validates model with the data provided by data_loader using the given metrics.

    :param model: torch.nn.Module
            Model to validate.
    :param data_loader: torch.utils.data.DataLoader
            Data loader to use for validation.
    :param metrics: Tuple[Callable]
            Tuple of metrics to use for validation. Each metric should take model output and labels and return a scalar.
    :param device: torch.device
            Device to use for validation.
    :return: Dict[str,float]
            Dictionary with metrics names as keys and metrics values as values.
    """
    evaluation_metrics_classification = {'val_accuracy_classification': accuracy_score,
                                         'val_precision_classification': partial(precision_score, average='macro'),
                                         'val_recall_classification': partial(recall_score, average='macro'),
                                         'val_f1_classification': partial(f1_score, average='macro')
                                         }

    evaluation_metric_arousal = {'val_arousal_mse': mean_squared_error,
                                 'val_arousal_mae': mean_absolute_error
                                 }

    evaluation_metric_valence = {'val_valence_mse': mean_squared_error,
                                 'val_valence_mae': mean_absolute_error
                                 }
    # create arrays for predictions and ground truth labels
    predictions_classifier, predictions_arousal, predictions_valence = [], [], []
    ground_truth_classifier, ground_truth_arousal, ground_truth_valence = [], [], []

    # start evaluation
    model.eval()
    with torch.no_grad():
        for i, data in enumerate(data_loader):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs = inputs.float()
            inputs = inputs.to(device)

            # forward pass
            outputs = model(inputs)
            regression_output = [outputs[1][:, 0], outputs[1][:, 1]]
            classification_output = outputs[0]

            # transform classification output to fit labels
            classification_output = torch.softmax(classification_output, dim=-1)
            classification_output = classification_output.cpu().numpy().squeeze()
            classification_output = np.argmax(classification_output, axis=-1)
            # transform regression output to fit labels
            regression_output = [regression_output[0].cpu().numpy().squeeze(),
                                 regression_output[1].cpu().numpy().squeeze()]


            # transform ground truth labels to fit predictions and sklearn metrics
            classification_ground_truth = labels[:, 2].cpu().numpy().squeeze()
            regression_ground_truth = [labels[:, 0].cpu().numpy().squeeze(), labels[:, 1].cpu().numpy().squeeze()]

            # save ground_truth labels and predictions in arrays to calculate metrics afterwards by one time
            predictions_arousal.append(regression_output[0])
            predictions_valence.append(regression_output[1])
            predictions_classifier.append(classification_output)
            ground_truth_arousal.append(regression_ground_truth[0])
            ground_truth_valence.append(regression_ground_truth[1])
            ground_truth_classifier.append(classification_ground_truth)

        # concatenate all predictions and ground truth labels
        predictions_arousal = np.concatenate(predictions_arousal, axis=0)
        predictions_valence = np.concatenate(predictions_valence, axis=0)
        predictions_classifier = np.concatenate(predictions_classifier, axis=0)
        ground_truth_arousal = np.concatenate(ground_truth_arousal, axis=0)
        ground_truth_valence = np.concatenate(ground_truth_valence, axis=0)
        ground_truth_classifier = np.concatenate(ground_truth_classifier, axis=0)

        # create mask for all NaN values to remove them from evaluation
        mask_arousal = ~np.isnan(ground_truth_arousal)
        mask_valence = ~np.isnan(ground_truth_valence)
        mask_classifier = ~np.isnan(ground_truth_classifier)
        # remove NaN values from arrays
        predictions_arousal = predictions_arousal[mask_arousal]
        predictions_valence = predictions_valence[mask_valence]
        predictions_classifier = predictions_classifier[mask_classifier]
        ground_truth_arousal = ground_truth_arousal[mask_arousal]
        ground_truth_valence = ground_truth_valence[mask_valence]
        ground_truth_classifier = ground_truth_classifier[mask_classifier]

        # calculate evaluation metrics
        evaluation_metrics_arousal = {
            metric: evaluation_metric_arousal[metric](ground_truth_arousal, predictions_arousal) for metric in
            evaluation_metric_arousal}
        evaluation_metrics_valence = {
            metric: evaluation_metric_valence[metric](ground_truth_valence, predictions_valence) for metric in
            evaluation_metric_valence}
        evaluation_metrics_classifier = {
            metric: evaluation_metrics_classification[metric](ground_truth_classifier, predictions_classifier) for
            metric in evaluation_metrics_classification}
        # print evaluation metrics
        print('Evaluation metrics for arousal:')
        for metric_name, metric_value in evaluation_metrics_arousal.items():
            print("%s: %.4f" % (metric_name, metric_value))
        print('Evaluation metrics for valence:')
        for metric_name, metric_value in evaluation_metrics_valence.items():
            print("%s: %.4f" % (metric_name, metric_value))
        print('Evaluation metrics for classifier:')
        for metric_name, metric_value in evaluation_metrics_classifier.items():
            print("%s: %.4f" % (metric_name, metric_value))
        general_val_metric = ((1.-evaluation_metrics_arousal['val_arousal_mae']) + (1.-evaluation_metrics_valence[
            'val_valence_mae']) + evaluation_metrics_classifier['val_recall_classification'])/3.
        print('General validation metric: %.4f' % general_val_metric)
    return evaluation_metrics_arousal, evaluation_metrics_valence, evaluation_metrics_classifier, general_val_metric




def validate_modified_MobileNetV3_on_AffectNet():
    # TODO: check this function and the validation one
    # load pd.DataFrames
    train, dev, test = load_all_dataframes(training_config.splitting_seed)
    # define preprocessing functions
    preprocessing_functions = [resize_image_to_224_saving_aspect_ratio,
                               preprocess_image_MobileNetV3]
    # take only AffectNet data
    #train = train[train['path'].str.contains('AffectNet')]
    #dev = dev[dev['path'].str.contains('AffectNet')]
    # construct data loaders
    train_dataloader, dev_dataloader, test_dataloader = construct_data_loaders(train, dev, test,
                                                                               preprocessing_functions,
                                                                               augmentation_functions=None,
                                                                               num_workers=16)
    # clear RAM
    del test, test_dataloader
    del train, train_dataloader
    gc.collect()

    # load model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    path_to_weights = '/work/home/dsu/emotion_recognition_project/best_model_metric.pth'
    model = Modified_MobileNetV3_large(embeddings_layer_neurons=256, num_classes=training_config.NUM_CLASSES,
                                       num_regression_neurons=training_config.NUM_REGRESSION_NEURONS)
    model.load_state_dict(torch.load(path_to_weights))
    model = model.to(device)
    model.eval()
    # evaluate model
    metrics_arousal, metrics_valence, metrics_classifier, general_val_metric = validate_model_modified_MobileNetV3(model, dev_dataloader, device)


if __name__ == '__main__':
    validate_modified_MobileNetV3_on_AffectNet()






