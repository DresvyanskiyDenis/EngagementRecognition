import sys

from tensorflow_utils.models.CNN_models import get_EmoVGGFace2_embeddings_extractor

sys.path.extend(["/work/home/dsu/datatools/"])
sys.path.extend(["/work/home/dsu/engagement_recognition_project_server/"])

from functools import partial

import tensorflow as tf

from feature_extraction.embeddings_extraction import extract_deep_embeddings_from_images_in_df
from preprocessing.data_normalizing_utils import Xception_normalization, VGGFace2_normalization
from src.NoXi.visual_subsystem.frame_wise_models.Xception_training import create_Xception_model
from src.NoXi.visual_subsystem.frame_wise_models.utils import load_NoXi_data_all_languages




def main():
    print("2222")
    # parameters for embeddings extraction
    output_dir="/work/home/dsu/NoXi_embeddings/All_languages/EmoVGGFace2/"
    #model_weights_path="/work/home/dsu/weights_of_best_models/ID_6.h5"
    model_creation_function=partial(get_EmoVGGFace2_embeddings_extractor, path_to_weights="/work/home/dsu/VGG_model_weights/EmoVGGFace2/weights_0_66_37_affectnet_cat.h5")

    # load the data and labels. The data is a dataframes with the filename (full path) as a first column - that is
    # what we need
    train, dev, test = load_NoXi_data_all_languages(labels_as_categories=False)

    # create the model and load its weights
    model=model_creation_function()
    #model.load_weights(model_weights_path)
    # cut off the last layer of the model
    #model=tf.keras.Model(inputs=model.inputs, outputs=model.layers[-2].output)
    model.compile(optimizer="Adam", loss="categorical_crossentropy")
    # TODO: check whether the weights are really loaded ot not
    model.summary()
    # extract the embeddings
    extract_deep_embeddings_from_images_in_df(paths_to_images=test, extractor=model, output_dir=output_dir, batch_size = 32,
                      preprocessing_functions= (VGGFace2_normalization,), include_labels=True)
    # should be done
    # TODO: Check it



if __name__ == '__main__':
    main()