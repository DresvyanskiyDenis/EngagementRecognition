from functools import partial
from typing import Optional, Dict

import numpy as np
import torch
from sklearn.metrics import precision_score, recall_score, f1_score
from tqdm import tqdm

from visualization.ConfusionMatrixVisualization import plot_and_save_confusion_matrix




def evaluate_model(model: torch.nn.Module, generator: torch.utils.data.DataLoader, device: torch.device,
                   metrics_name_prefix:Optional[str]=None, print_metrics:Optional[bool]=None) -> Dict[
    object, float]:
    from sklearn.metrics import accuracy_score
    evaluation_metrics_classification = {'accuracy_classification': accuracy_score,
                                         'precision_classification': partial(precision_score, average='macro'),
                                         'recall_classification': partial(recall_score, average='macro'),
                                         'f1_classification': partial(f1_score, average='macro')
                                         }
    # change the names of the metrics if needed
    if metrics_name_prefix is not None:
        evaluation_metrics_classification = {metrics_name_prefix + metric_name: metric for metric_name, metric in
                                             evaluation_metrics_classification.items()}

    # create arrays for predictions and ground truth labels
    predictions_classifier = []
    ground_truth_classifier = []

    # start evaluation
    model.eval()
    with torch.no_grad():
        for i, data in enumerate(tqdm(generator)):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            # inputs are list of tensors with the shape (batch_size, sequence_length, num_features)
            inputs = [inp.float().to(device) for inp in inputs]
            # the labels originally are the sequence of values, since we have several images per one instance
            # however, we transform it to the sequence-to-one problem. Thus, we take the mode of the sequence, but
            # firstly we should get rid of the one-hot encoding
            labels = torch.argmax(labels, dim=-1)
            # then we take the mode
            labels = torch.mode(labels, dim=-1)[0]

            # forward pass
            outputs = model(*inputs)
            classification_output = outputs

            # transform classification output to fit labels
            classification_output = torch.softmax(classification_output, dim=-1)
            classification_output = classification_output.cpu().numpy().squeeze()
            classification_output = np.argmax(classification_output, axis=-1)

            # transform ground truth labels to fit predictions and sklearn metrics
            classification_ground_truth = labels.cpu().numpy().squeeze()

            # save ground_truth labels and predictions in arrays to calculate metrics afterwards by one time
            predictions_classifier.append(classification_output)
            ground_truth_classifier.append(classification_ground_truth)

        # concatenate all predictions and ground truth labels
        predictions_classifier = np.concatenate(predictions_classifier, axis=0)
        ground_truth_classifier = np.concatenate(ground_truth_classifier, axis=0)

        # calculate evaluation metrics
        evaluation_metrics_classifier = {
            metric: evaluation_metrics_classification[metric](ground_truth_classifier, predictions_classifier) for
            metric in evaluation_metrics_classification}
        # print evaluation metrics if needed
        if print_metrics:
            print('Evaluation metrics:')
            for metric_name, metric_value in evaluation_metrics_classifier.items():
                print("%s: %.4f" % (metric_name, metric_value))
    # clear RAM from unused variables
    del inputs, labels, outputs, classification_output, classification_ground_truth
    torch.cuda.empty_cache()
    return evaluation_metrics_classifier


