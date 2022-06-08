import sys
sys.path.extend(["/work/home/dsu/datatools/"])
sys.path.extend(["/work/home/dsu/engagement_recognition_project_server/"])

import gc
import numpy as np
import pandas as pd
import tensorflow as tf
import wandb
from typing import Optional, Tuple

from keras.callbacks import EarlyStopping
from wandb.integration.keras import WandbCallback
from functools import partial
from sklearn.metrics import recall_score, precision_score, f1_score, accuracy_score, confusion_matrix
from sklearn.utils import compute_class_weight

from src.NoXi.visual_subsystem.sequence_models.sequence_data_loader import create_generator_from_pd_dictionary, \
    load_data
from tensorflow_utils.models.sequence_to_one_models import create_simple_RNN_network
from tensorflow_utils.Losses import categorical_focal_loss
from tensorflow_utils.wandb_callbacks import WandB_LR_log_callback, WandB_val_metrics_callback
from tensorflow_utils.callbacks import get_annealing_LRreduce_callback, get_reduceLRonPlateau_callback


def create_sequence_model(input_shape:Tuple[int,...], neurons_on_layer:Tuple[int,...],
                          num_classes: Optional[int] = 5) -> tf.keras.Model:

    sequence_model=create_simple_RNN_network(input_shape=input_shape, num_output_neurons=num_classes,
                              neurons_on_layer = neurons_on_layer,
                              rnn_type = 'LSTM',
                              dnn_layers=(128,),
                              need_regularization = True,
                              dropout = True)
    return sequence_model



def validate_model_on_generator(model:tf.keras.Model, generator):
    """Validates provided model using generator (preferable tf.Dataset).
       Validation is done applying the following metrics: [accuracy, precision, recall, f1_score, confusion matrix]
    :param model: tf.Model
                Tensorflow model
    :param generator: Iterator
                Any generator that produces the output as a tuple: (features, labels)
    :return:
    """

    metrics = {
        'accuracy': accuracy_score,
        'recall': partial(recall_score, average='macro'),
        'precision': partial(precision_score, average='macro'),
        'f1_score': partial(f1_score, average='macro'),
        'confusion_matrix': confusion_matrix,
    }
    # make predictions for data from generator and save ground truth labels
    total_predictions = np.zeros((0,))
    total_ground_truth = np.zeros((0,))
    for x, y in generator.as_numpy_iterator():
        predictions = model.predict(x, batch_size=64)
        predictions = predictions.squeeze().argmax(axis=-1).reshape((-1,))
        total_predictions = np.append(total_predictions, predictions)
        total_ground_truth = np.append(total_ground_truth, y.squeeze().argmax(axis=-1).reshape((-1,)))
    # calculate all provided metrics and save them as dict object
    # as Dict[metric_name->value]
    results = {}
    for key in metrics.keys():
        results[key] = metrics[key](total_ground_truth, total_predictions)
    # clear RAM
    del total_predictions, total_ground_truth
    gc.collect()
    conf_m=results.pop('confusion_matrix')
    # print results
    for key, value in results.items():
        print("Metric: %s, result:%f"%(key, value))
    # print confusion matrix separately one more time for better vision
    print("Confusion matrix")
    print(conf_m)
    return results




def validate_model(dev, test, weights_path, window_length, num_neurons, num_layers):
    # model params
    num_classes=5
    num_embeddings=256
    window_stride=1
    batch_size=16
    type_of_labels = 'sequence_to_one'
    loss='categorical_crossentropy'
    optimizer='Adam'
    metrics=None



    # model initialization
    model = create_sequence_model(num_classes=num_classes, neurons_on_layer=tuple(num_neurons for i in range(num_layers)),
                          input_shape=(window_length, num_embeddings))

    # load weights
    model.load_weights(weights_path)

    # model compilation
    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
    model.summary()

    # transform labels in dev data to one-hot encodings
    dev_data_loader = create_generator_from_pd_dictionary(embeddings_dict=dev, num_classes=num_classes,
                                                            type_of_labels=type_of_labels,
                                        window_length=window_length, window_shift=window_length,
                                        window_stride=window_stride,batch_size=batch_size, shuffle=False,
                                        preprocessing_function=None,
                                        clip_values=None,
                                        cache_loaded_seq= None
                                        )

    # transform labels in dev data to one-hot encodings
    test_data_loader = create_generator_from_pd_dictionary(embeddings_dict=test, num_classes=num_classes,
                                                            type_of_labels=type_of_labels,
                                        window_length=window_length, window_shift=window_length,
                                        window_stride=window_stride,batch_size=batch_size, shuffle=False,
                                        preprocessing_function=None,
                                        clip_values=None,
                                        cache_loaded_seq= None
                                        )

    print("00000000000000000000000000000000000000000000000000000000")
    print(weights_path)
    dev_results=validate_model_on_generator(model, dev_data_loader)
    print('--------------------test-------------------')
    test_results=validate_model_on_generator(model, test_data_loader)
    print("00000000000000000000000000000000000000000000000000000000")
    return (dev_results, test_results)


def main():
    train, dev, test = load_data()
    dir_with_models="/work/home/dsu/weights_of_best_models/sequence_to_one_experiments/models_to_check/"
    params_for_testing_models=pd.read_csv("/work/home/dsu/weights_of_best_models/sequence_to_one_experiments/models_to_check/params_of_testing.csv",
                                          delimiter=';')
    params_for_testing_models["weights_path"]=params_for_testing_models.apply(lambda x:
                                                                              "window_"+str(x["window_length"])+"_sweep_"+str(x["sweep_id"])+".h5",
                                                                              axis=1)


    results=pd.DataFrame(columns=["weights_path","val_recall", "val_precision", "val_f1_score", "val_accuracy",
                                  "test_recall", "test_precision", "test_f1_score", "test_accuracy" ])
    # calculate results
    for index, row in params_for_testing_models.iterrows():
        dev_res, test_res=validate_model(dev, test, dir_with_models+row["weights_path"], row["window_length"], row["num_neurons"], row["num_layers"])
        results=results.append({"weights_path":row["weights_path"],
                                "val_recall":dev_res["recall"], "val_precision":dev_res["precision"],
                                "val_f1_score":dev_res["f1_score"], "val_accuracy":dev_res["accuracy"],
                                "test_recall":test_res["recall"], "test_precision":test_res["precision"],
                                "test_f1_score":test_res["f1_score"], "test_accuracy":test_res["accuracy"]},
                                ignore_index=True)
    # write results to the csv file
    results.to_csv("/work/home/dsu/weights_of_best_models/sequence_to_one_experiments/models_to_check/results_of_testing.csv")


if __name__ == '__main__':
    main()
