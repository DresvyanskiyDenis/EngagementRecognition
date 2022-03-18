import sys
sys.path.extend(["/work/home/dsu/datatools/"])
sys.path.extend(["/work/home/dsu/engagement_recognition_project_server/"])


from src.NoXi.visual_subsystem.frame_wise_models.VGGFace2_training import load_and_preprocess_data, \
    create_VGGFace2_model
from functools import partial
from sklearn.metrics import recall_score, precision_score, f1_score
import gc
import numpy as np
import pandas as pd

from preprocessing.data_normalizing_utils import VGGFace2_normalization
from tensorflow_utils.keras_datagenerators.ImageDataLoader import ImageDataLoader


def validate_model(model, generator):

    metrics = {
        'val_recall': partial(recall_score, average='macro'),
        'val_precision': partial(precision_score, average='macro'),
        'val_f1_score:': partial(f1_score, average='macro')
    }
    # make predictions for data from generator and save ground truth labels
    total_predictions = np.zeros((0,))
    total_ground_truth = np.zeros((0,))
    for x, y in generator:
        predictions = model.predict(x, batch_size=32)
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
    # print results
    for key, value in results.items():
        print("Metric: %s, result:%f"%(key, value))



def main():
    # loading data
    frame_step = 5
    path_to_data = "/Noxi_extracted/NoXi/extracted_faces/"
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
    del test, test_english, test_german, test_french
    gc.collect()
    pass

    dev_data_loader = ImageDataLoader(paths_with_labels=dev, batch_size=32,
                                      preprocess_function=VGGFace2_normalization,
                                      num_classes=5,
                                      horizontal_flip=0, vertical_flip=0,
                                      shift=0,
                                      brightness=0, shearing=0, zooming=0,
                                      random_cropping_out=0, rotation=0,
                                      scaling=0,
                                      channel_random_noise=0, bluring=0,
                                      worse_quality=0,
                                      mixup=None,
                                      prob_factors_for_each_class=None,
                                      pool_workers=16)


    # model initialization
    model = create_VGGFace2_model(path_to_weights='/work/home/dsu/VGG_model_weights/resnet50_softmax_dim512/weights.h5',
                                  num_classes=5)
    # load weights of trained model
    model.load_weights('/work/home/dsu/model_best_val_recall.h5')
    # model compilation
    model.compile(loss='categorical_crossentropy', optimizer='Adam')
    model.summary()

    validate_model(model, dev_data_loader)

if __name__ == '__main__':
    main()