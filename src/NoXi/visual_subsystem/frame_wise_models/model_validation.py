import sys

from src.NoXi.visual_subsystem.frame_wise_models.Xception_training import create_Xception_model

sys.path.extend(["/work/home/dsu/datatools/"])
sys.path.extend(["/work/home/dsu/engagement_recognition_project_server/"])


from src.NoXi.visual_subsystem.frame_wise_models.MobileNetv3_training import create_MobileNetv3_model
from tensorflow_utils.tensorflow_datagenerators.ImageDataLoader_tf2 import get_tensorflow_generator
from tensorflow_utils.tensorflow_datagenerators.tensorflow_image_preprocessing import preprocess_data_MobileNetv3, \
    preprocess_image_VGGFace2, preprocess_data_Xception

from src.NoXi.visual_subsystem.frame_wise_models.VGGFace2_training import load_and_preprocess_data, \
    create_VGGFace2_model
from functools import partial
from sklearn.metrics import recall_score, precision_score, f1_score, confusion_matrix, accuracy_score
import gc
import numpy as np
import pandas as pd

from preprocessing.data_normalizing_utils import VGGFace2_normalization


def validate_model(model, generator):

    metrics = {
        'val_accuracy': accuracy_score,
        'val_recall': partial(recall_score, average='macro'),
        'val_precision': partial(precision_score, average='macro'),
        'val_f1_score:': partial(f1_score, average='macro'),
        'confusion_matrix': confusion_matrix,
    }
    # make predictions for data from generator and save ground truth labels
    total_predictions = np.zeros((0,))
    total_ground_truth = np.zeros((0,))
    for x, y in generator.as_numpy_iterator():
        predictions = model.predict(x, batch_size=64)
        predictions = predictions.argmax(axis=-1).reshape((-1,))
        total_predictions = np.append(total_predictions, predictions)
        total_ground_truth = np.append(total_ground_truth, y.argmax(axis=-1).reshape((-1,)))
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



def main():
    print("234")
    # params
    frame_step = 5
    path_to_data = "/Noxi_extracted/NoXi/extracted_faces/"
    path_to_model_weights = "/work/home/dsu/weights_of_best_models/ID_9_weeep_30.h5"

    # loading data
    # french data
    path_to_labels_french = "/media/external_hdd_1/NoXi_annotations_reliable_gold_standard_classification_with_additional_train_data/French"
    train_french, dev_french, test_french = load_and_preprocess_data(path_to_data, path_to_labels_french,
                                                                     frame_step)
    # english data
    path_to_labels_german = "/media/external_hdd_1/NoXi_annotations_reliable_gold_standard_classification_with_additional_train_data/German"
    train_german, dev_german, test_german = load_and_preprocess_data(path_to_data, path_to_labels_german,
                                                                     frame_step)
    # german data
    path_to_labels_english = "/media/external_hdd_1/NoXi_annotations_reliable_gold_standard_classification_with_additional_train_data/English"
    train_english, dev_english, test_english = load_and_preprocess_data(path_to_data, path_to_labels_english,
                                                                        frame_step)
    # all data
    train = pd.concat([train_french, train_german, train_english], axis=0)
    dev = pd.concat([dev_french, dev_german, dev_english], axis=0)
    test = pd.concat([test_french, test_german, test_english], axis=0)
    # shuffle one more time train data
    train = train.sample(frac=1).reset_index(drop=True)
    # clear RAM
    del train_english, train_french, train_german
    del dev_english, dev_french, dev_german
    del test_english, test_german, test_french
    gc.collect()

    # transform labels in dev data to one-hot encodings
    dev = dev.__deepcopy__()
    dev = pd.concat([dev, pd.get_dummies(dev['class'], dtype="float32")], axis=1).drop(columns=['class'])

    dev_data_loader = get_tensorflow_generator(paths_and_labels=dev, batch_size=128,
                                                 augmentation=False,
                                                 augmentation_methods=None,
                                                 preprocessing_function=preprocess_image_VGGFace2,
                                                 clip_values=None,
                                                 cache_loaded_images=False)

    # transform labels in dev data to one-hot encodings
    test = test.__deepcopy__()
    test = pd.concat([test, pd.get_dummies(test['class'], dtype="float32")], axis=1).drop(columns=['class'])

    test_data_loader = get_tensorflow_generator(paths_and_labels=test, batch_size=128,
                                               augmentation=False,
                                               augmentation_methods=None,
                                               preprocessing_function=preprocess_image_VGGFace2,
                                               clip_values=None,
                                               cache_loaded_images=False)

    # model initialization
    model = create_VGGFace2_model(path_to_weights="/work/home/dsu/VGG_model_weights/resnet50_softmax_dim512/weights.h5",
                                  num_classes=5)
    # load weights of trained model
    model.load_weights(path_to_model_weights)
    # model compilation
    model.compile(loss='categorical_crossentropy', optimizer='Adam')
    model.summary()

    validate_model(model, dev_data_loader)
    print("----------------------------test--------------------------------")
    validate_model(model, test_data_loader)

if __name__ == '__main__':
    main()