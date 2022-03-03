import sys
sys.path.extend(["/work/home/dsu/datatools/"])
sys.path.extend(["/work/home/dsu/engagement_recognition_project_server/"])


from functools import partial
from sklearn.metrics import recall_score, precision_score, f1_score
import tensorflow as tf
from tensorflow_utils.wandb_callbacks import WandB_val_metrics_callback
import objgraph
from preprocessing.data_normalizing_utils import VGGFace2_normalization
from src.NoXi.visual_subsystem.frame_wise_models.VGGFace2_training import load_and_preprocess_data, \
    create_VGGFace2_model
from tensorflow_utils.keras_datagenerators.ImageDataLoader import ImageDataLoader



import gc

import pandas as pd


# loading data
def main():
    metaparams = {
        "optimizer": "Adam",  # SGD, Nadam
        "learning_rate_max": 0.001,  # up to 0.0001
        "learning_rate_min": 0.00001,  # up to 0.000001
        "lr_scheduller": "Cyclic",  # "reduceLRonPlateau"
        "annealing_period": 5,
        "epochs": 30,
        "batch_size": 80,
        "augmentation_rate": 0.1,  # 0.2, 0.3
        "architecture": "VGGFace2_full_training",
        "dataset": "NoXi_english",
        "num_classes": 5
    }
    print("start----------------------------------------------")
    objgraph.show_growth()

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

    print("data loaded----------------------------------------------")
    objgraph.show_growth()

    # create DataLoaders (DataGenerator)
    train_data_loader = ImageDataLoader(paths_with_labels=train, batch_size=metaparams["batch_size"],
                                        already_one_hot_encoded=True,
                                        preprocess_function=VGGFace2_normalization,
                                        num_classes=metaparams["num_classes"],
                                        horizontal_flip=metaparams["augmentation_rate"],
                                        vertical_flip=metaparams["augmentation_rate"],
                                        shift=metaparams["augmentation_rate"],
                                        brightness=metaparams["augmentation_rate"], shearing=metaparams["augmentation_rate"],
                                        zooming=metaparams["augmentation_rate"],
                                        random_cropping_out=metaparams["augmentation_rate"], rotation=metaparams["augmentation_rate"],
                                        scaling=metaparams["augmentation_rate"],
                                        channel_random_noise=metaparams["augmentation_rate"], bluring=metaparams["augmentation_rate"],
                                        worse_quality=metaparams["augmentation_rate"],
                                        mixup=None,
                                        prob_factors_for_each_class=None,
                                        pool_workers=24)

    dev_data_loader = ImageDataLoader(paths_with_labels=dev, batch_size=metaparams["batch_size"],
                                      preprocess_function=VGGFace2_normalization,
                                      num_classes=metaparams["num_classes"],
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

    print("data loaders are created----------------------------------------------")
    objgraph.show_growth()

    # model initialization
    model = create_VGGFace2_model(path_to_weights='/work/home/dsu/VGG_model_weights/resnet50_softmax_dim512/weights.h5',
                                  num_classes=metaparams["num_classes"])
    model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])
    model.summary()

    print("--------------------")

    val_metrics = {
        'val_recall': partial(recall_score, average='macro'),
        'val_precision': partial(precision_score, average='macro'),
        'val_f1_score:': partial(f1_score, average='macro')
    }
    val_metrics_callback = WandB_val_metrics_callback(dev_data_loader, val_metrics)


    model.fit(train_data_loader, epochs=100,
              callbacks=[val_metrics_callback])
    # clear RAM
    del train_data_loader, dev_data_loader
    del model
    gc.collect()
    tf.keras.backend.clear_session()


if __name__ == '__main__':
    main()