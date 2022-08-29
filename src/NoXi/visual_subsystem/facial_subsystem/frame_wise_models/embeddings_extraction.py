#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Contains the script for extracting the embeddings from the facial images taken from NoXi dataset.

"""

__author__ = "Denis Dresvyanskiy"
__copyright__ = "Copyright 2022"
__credits__ = ["Denis Dresvyanskiy"]
__maintainer__ = "Denis Dresvyanskiy"
__email__ = "denis.dresvyanskiy@uni-ulm.de"

import os
import sys

from src.NoXi.visual_subsystem.facial_subsystem.frame_wise_models.Cross_language_training import create_Xception_model
from tensorflow_utils.tensorflow_datagenerators.tensorflow_image_preprocessing import preprocess_data_Xception

sys.path.extend(["/work/home/dsu/datatools/"])
sys.path.extend(["/work/home/dsu/engagement_recognition_project_server/"])

from functools import partial
import tensorflow as tf

from feature_extraction.embeddings_extraction import extract_deep_embeddings_from_images_in_df
from preprocessing.data_normalizing_utils import VGGFace2_normalization
from src.NoXi.visual_subsystem.facial_subsystem.frame_wise_models.utils import load_NoXi_data_all_languages, \
    load_NoXi_data_cross_corpus
from tensorflow_utils.models.CNN_models import get_EmoVGGFace2_embeddings_extractor



def main():
    print("Start... embeddings extraction.")
    # parameters for embeddings extraction
    output_dir="/work/home/dsu/NoXi/NoXi_embeddings/Cross-corpus/English"
    model_weights_path=""
    #model_creation_function=partial(get_EmoVGGFace2_embeddings_extractor, path_to_weights="/work/home/dsu/VGG_model_weights/EmoVGGFace2/weights_0_66_37_affectnet_cat.h5")
    model_creation_function = create_Xception_model

    # load the data and labels. The data is a dataframes with the filename (full path) as a first column - that is
    # what we need
    train, dev, test = load_NoXi_data_cross_corpus(test_corpus='english')

    # create the model and load its weights
    model=model_creation_function()
    model.load_weights(model_weights_path)
    # cut off the last layer of the model
    model=tf.keras.Model(inputs=model.inputs, outputs=model.layers[-2].output)
    model.compile(optimizer="Adam", loss="categorical_crossentropy")
    # TODO: check whether the weights are really loaded ot not
    model.summary()
    # extract the embeddings
    extract_deep_embeddings_from_images_in_df(paths_to_images=train, extractor=model, output_dir=os.path.join(output_dir, "train"),
                                              batch_size = 32,
                                              preprocessing_functions= (preprocess_data_Xception,), include_labels=True)

    extract_deep_embeddings_from_images_in_df(paths_to_images=dev, extractor=model, output_dir=os.path.join(output_dir, "dev"),
                                              batch_size=32,
                                              preprocessing_functions=(preprocess_data_Xception,), include_labels=True)

    extract_deep_embeddings_from_images_in_df(paths_to_images=test, extractor=model, output_dir=os.path.join(output_dir, "test"),
                                              batch_size=32,
                                              preprocessing_functions=(preprocess_data_Xception,), include_labels=True)
    # should be done
    # TODO: Check it



if __name__ == '__main__':
    main()