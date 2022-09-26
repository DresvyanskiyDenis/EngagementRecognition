import torch
import tensorflow as tf
import pandas as pd
import numpy as np
import torchvision.transforms as T

from feature_extraction.class_scores_extraction import extract_scores_from_images_in_df_torch, \
    extract_scores_from_images_in_df_tf
from src.NoXi.visual_subsystem.facial_subsystem.frame_wise_models.VGGFace2_training import create_VGGFace2_model
from src.NoXi.visual_subsystem.facial_subsystem.frame_wise_models.Xception_training import create_Xception_model
from src.NoXi.visual_subsystem.pose_subsystem.frame_wise_model.HRNet import load_HRNet_model, modified_HRNet
from src.NoXi.visual_subsystem.pose_subsystem.frame_wise_model.utils import load_NoXi_data_all_languages, \
    convert_image_to_float_and_scale
from tensorflow_utils.tensorflow_datagenerators.tensorflow_image_preprocessing import preprocess_image_VGGFace2, \
    preprocess_data_Xception


def extract_scores_for_pose_model():
    # params
    model_path = "/work/home/dsu/Model_weights/weights_of_best_models/frame_to_frame_experiments/Pose_model/All_languages/crossentropy_6_2.pt"
    HRNet_weights = "/work/home/dsu/simpleHRNet/pose_hrnet_w32_256x192.pth"
    output_path = "/work/home/dsu/NoXi/extracted_scores/All_languages/pose_model/"
    preprocessing_functions = [T.Resize(size=(256, 256)),convert_image_to_float_and_scale,
                               T.Normalize( mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                               ]
    num_classes = 5


    train, dev, test = load_NoXi_data_all_languages(train_labels_as_categories=False,
                                                    dev_labels_as_categories=False,
                                                    test_labels_as_categories=False)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # load model
    model = load_HRNet_model(device='cuda:0' if torch.cuda.is_available() else 'cpu',
                             path_to_weights=HRNet_weights)
    model = modified_HRNet(model, num_classes=num_classes)
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()
    # train data
    extract_scores_from_images_in_df_torch(df=train, model=model, device=device, output_path=output_path,
        output_filename="train_scores.csv",
        preprocessing_functions=preprocessing_functions, num_neurons_last_layer=num_classes,
        apply_softmax=True,
        include_labels = True, batch_size= 64)
    # dev data
    extract_scores_from_images_in_df_torch(df=dev, model=model, device=device, output_path=output_path,
        output_filename="dev_scores.csv",
        preprocessing_functions=preprocessing_functions, num_neurons_last_layer=num_classes,
        apply_softmax=True,
        include_labels = True, batch_size= 64)
    # test data
    extract_scores_from_images_in_df_torch(df=test, model=model, device=device, output_path=output_path,
        output_filename="test_scores.csv",
        preprocessing_functions=preprocessing_functions, num_neurons_last_layer=num_classes,
        apply_softmax=True,
        include_labels = True, batch_size= 64)


def extract_scores_for_Xception():
    # params
    path_to_model_weights ="/work/home/dsu/Model_weights/weights_of_best_models/frame_to_frame_experiments/Facial_model/All_languages/ID_6.h5"
    output_path = "/work/home/dsu/NoXi/extracted_scores/All_languages/facial_model/"
    num_classes = 5
    preprocess_function = preprocess_data_Xception

    # load the data and labels
    train, dev, test = load_NoXi_data_all_languages()

    model = create_Xception_model(num_classes=num_classes)
    model.load_weights(path_to_model_weights)
    model.compile(loss='categorical_crossentropy', optimizer='Adam')
    model.summary()

    # train
    extract_scores_from_images_in_df_tf(paths_to_images=train, model=model, output_dir=output_path,
                                        output_filename="train_scores.csv",
                                        batch_size = 64, preprocessing_functions = (preprocess_function,),
                                        include_labels = True)

    # dev
    extract_scores_from_images_in_df_tf(paths_to_images=dev, model=model, output_dir=output_path,
                                        output_filename="dev_scores.csv",
                                        batch_size = 64, preprocessing_functions = (preprocess_function,),
                                        include_labels = True)

    # test
    extract_scores_from_images_in_df_tf(paths_to_images=test, model=model, output_dir=output_path,
                                        output_filename="test_scores.csv",
                                        batch_size = 64, preprocessing_functions = (preprocess_function,),
                                        include_labels = True)




def main():
    extract_scores_for_pose_model()
    #extract_scores_for_Xception()



if __name__ == "__main__":
    main()