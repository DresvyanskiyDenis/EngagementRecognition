from functools import partial
from typing import Dict, Optional

import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from visualization.ConfusionMatrixVisualization import plot_and_save_confusion_matrix


def evaluate_model(model: torch.nn.Module, generator: torch.utils.data.DataLoader, device: torch.device,
                   metrics_name_prefix:Optional[str]=None, print_metrics:Optional[bool]=None) -> Dict[object, float]:
    evaluation_metrics_classification = {'accuracy_classification': accuracy_score,
                                         'precision_classification': partial(precision_score, average='macro'),
                                         'recall_classification': partial(recall_score, average='macro'),
                                         'f1_classification': partial(f1_score, average='macro'),
                                         }

    # create arrays for predictions and ground truth labels
    predictions_classifier= []
    ground_truth_classifier = []

    # start evaluation
    model.eval()
    with torch.no_grad():
        for i, data in enumerate(generator):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs = inputs.float()
            inputs = inputs.to(device)

            # forward pass
            classification_output = model(inputs)

            # transform classification output to fit labels
            classification_output = torch.softmax(classification_output, dim=-1)
            classification_output = classification_output.cpu().numpy().squeeze()
            classification_output = np.argmax(classification_output, axis=-1)

            # transform ground truth labels to fit predictions and sklearn metrics
            classification_ground_truth = labels.cpu().numpy().squeeze()
            classification_ground_truth = np.argmax(classification_ground_truth, axis=-1)

            # save ground_truth labels and predictions in arrays to calculate metrics afterwards by one time
            predictions_classifier.append(classification_output)
            ground_truth_classifier.append(classification_ground_truth)

        # concatenate all predictions and ground truth labels
        predictions_classifier = np.concatenate(predictions_classifier, axis=0)
        ground_truth_classifier = np.concatenate(ground_truth_classifier, axis=0)

        # calculate evaluation metrics
        evaluation_metrics_classifier = {
            metric: evaluation_metrics_classification[metric](ground_truth_classifier, predictions_classifier)
            for metric in evaluation_metrics_classification
        }
        # print evaluation metrics
        if print_metrics:
            for metric_name, metric_value in evaluation_metrics_classifier.items():
                print("%s: %.4f" % (metric_name, metric_value))
        # change the name of the metrics if needed
        if metrics_name_prefix:
            evaluation_metrics_classifier = {metrics_name_prefix + metric_name: metric_value
                                             for metric_name, metric_value in evaluation_metrics_classifier.items()}
    # clear RAM from unused variables
    del inputs, labels, classification_output, classification_ground_truth
    torch.cuda.empty_cache()
    return evaluation_metrics_classifier


def draw_confusion_matrix(model: torch.nn.Module, generator: torch.utils.data.DataLoader, device: torch.device,
                          output_path:Optional[str]='confusion_matrix', filename:Optional[str]='cm.png')->None:

    # create arrays for predictions and ground truth labels
    predictions_classifier = []
    ground_truth_classifier = []

    # start evaluation
    model.eval()
    with torch.no_grad():
        for i, data in enumerate(generator):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs = inputs.float()
            inputs = inputs.to(device)

            # forward pass
            classification_output = model(inputs)

            # transform classification output to fit labels
            classification_output = torch.softmax(classification_output, dim=-1)
            classification_output = classification_output.cpu().numpy().squeeze()
            classification_output = np.argmax(classification_output, axis=-1)

            # transform ground truth labels to fit predictions and sklearn metrics
            classification_ground_truth = labels.cpu().numpy().squeeze()
            classification_ground_truth = np.argmax(classification_ground_truth, axis=-1)

            # save ground_truth labels and predictions in arrays to calculate metrics afterwards by one time
            predictions_classifier.append(classification_output)
            ground_truth_classifier.append(classification_ground_truth)

        # concatenate all predictions and ground truth labels
        predictions_classifier = np.concatenate(predictions_classifier, axis=0)
        ground_truth_classifier = np.concatenate(ground_truth_classifier, axis=0)

        # draw confusion matrix
        plot_and_save_confusion_matrix(y_true=ground_truth_classifier, y_pred=predictions_classifier,
                                       name_labels=['disengaged', 'neutral', 'engaged'],
        path_to_save=output_path, name_filename = filename, title = 'Confusion matrix')

    # clear RAM from unused variables
    del inputs, labels, classification_output, classification_ground_truth
    torch.cuda.empty_cache()
