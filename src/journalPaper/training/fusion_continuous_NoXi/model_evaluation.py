from typing import List

import numpy as np
import torch
from scipy.stats import pearsonr

from src.journalPaper.training.fusion_continuous_NoXi.metrics import np_concordance_correlation_coefficient


def evaluate_model(model, generator, modalities:List[str], device,
                   metrics_name_prefix, print_metrics):
    evaluation_metrics = {'CCC': np_concordance_correlation_coefficient,
                                         'PCC': lambda x,y: pearsonr(x,y)[0],
                                         }
    # change the names of the metrics if needed
    if metrics_name_prefix is not None:
        evaluation_metrics = {metrics_name_prefix + metric_name: metric for metric_name, metric in
                                             evaluation_metrics.items()}
    # create arrays for predictions and ground truth labels
    predictions = []
    ground_truth = []
    # start
    model.eval()
    with torch.no_grad():
        for i, data in enumerate(generator):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            # order of modalities is always [face, pose, emo]. We need to choose the ones that are included in current experiment
            tmp = []
            if 'face' in modalities: tmp.append(inputs[0])
            if 'pose' in modalities: tmp.append(inputs[1])
            if 'emo' in modalities: tmp.append(inputs[2])
            inputs = tmp
            # now we have only the modalities that are included in the experiment
            inputs = [input.float().to(device) for input in inputs]
            # labels shape: List of (batch_size, sequence_length, 1)
            # as all labels are the same, just take the first element of list
            labels = labels[0].float()
            # forward pass
            outputs = model(*inputs)
            # transform output to numpy
            outputs = outputs.cpu().numpy().squeeze()
            labels = labels.cpu().numpy().squeeze()
            # save ground_truth labels and predictions in arrays to calculate metrics afterwards by one time
            predictions.append(outputs)
            ground_truth.append(labels)
        # concatenate all predictions and ground truth labels
        predictions = np.concatenate(predictions, axis=0).flatten()
        ground_truth = np.concatenate(ground_truth, axis=0).flatten()
        # calculate evaluation metrics
        evaluation_metrics = {
            metric: evaluation_metrics[metric](ground_truth, predictions) for metric in evaluation_metrics.keys()}
        if print_metrics:
            print(evaluation_metrics)
        return evaluation_metrics

