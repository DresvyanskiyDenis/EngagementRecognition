import gc
import sys
from typing import List, Tuple, Callable

sys.path.extend(["/work/home/dsu/datatools/"])
sys.path.extend(["/work/home/dsu/engagement_recognition_project_server/"])
sys.path.extend(["/work/home/dsu/simpleHRNet/"])

import torch
import numpy as np
import pandas as pd
import os

import torchvision.transforms as T
import torchvision.io
from torchinfo import summary

from src.NoXi.visual_subsystem.pose_subsystem.frame_wise_model.HRNet import load_HRNet_model, modified_HRNet
from src.NoXi.visual_subsystem.pose_subsystem.frame_wise_model.utils import convert_image_to_float_and_scale, \
    load_NoXi_data_all_languages


def load_and_preprocess_batch_images(paths: List[str], preprocess_functions: Tuple[Callable]) -> torch.Tensor:
    # read batch of images
    images = [torchvision.io.read_image(path) for path in paths]
    for i in range(len(images)):
        for preprocess_function in preprocess_functions:
            images[i] = preprocess_function(images[i])
    batch = torch.stack(images)
    return batch


def load_embedding_extractor(path_to_weights: str, num_classes: int) -> torch.nn.Module:
    # create model and load its weights
    HRNet = load_HRNet_model(device="cuda" if torch.cuda.is_available() else "cpu",
                             path_to_weights="/work/home/dsu/simpleHRNet/pose_hrnet_w32_256x192.pth")
    model = modified_HRNet(HRNet, num_classes=num_classes)
    model.load_state_dict(torch.load(path_to_weights))
    return model


def extract_embeddings_from_df_with_paths(df: pd.DataFrame, extractor: torch.nn.Module, device, output_path: str,
                                          output_filename: str,
                                          preprocessing_functions, num_neurons_last_layer: int = 256,
                                          include_labels: bool = False,
                                          batch_size: int = 64) -> None:
    """

    :param df: DataFrame with columns "filename", "label_0", "label_1", ... , "label_n"
    :param extractor:
    :param device:
    :param output_path:
    :param preprocessing_functions:
    :param num_neurons_last_layer:
    :return:
    """
    if not os.path.exists(output_path):
        os.makedirs(output_path, exist_ok=True)
    # create a DataFrame file for embeddings
    num_embeddings = num_neurons_last_layer
    columns = ['filename'] + ["embedding_" + str(i) for i in range(num_embeddings)]
    if include_labels:
        columns += ["label_" + str(i) for i in range(df.shape[1] - 1)]
    # create dataframe for saving features
    extracted_deep_embeddings = pd.DataFrame(columns=columns)
    # save the "template" csv file to append to it in future
    extracted_deep_embeddings.to_csv(os.path.join(output_path, 'extracted_deep_embeddings.csv'), index=False)
    # load batch_size images and then predict them
    with torch.no_grad():
        for extraction_idx, filename_idx in enumerate(range(0, df.shape[0], batch_size)):
            batch_filenames = df.iloc[filename_idx:(filename_idx + batch_size), 0].values.flatten()
            # load images and send them to device for calculation
            loaded_images = load_and_preprocess_batch_images(batch_filenames,
                                                             preprocess_functions=preprocessing_functions)
            loaded_images = loaded_images.to(device)
            # extract embeddings
            embeddings = extractor(loaded_images)
            embeddings = embeddings.cpu().numpy()
            # if we want to include labels in the resulting csv file
            if include_labels:
                labels = df.iloc[filename_idx:(filename_idx + batch_size), 1:].values
                embeddings = np.concatenate([embeddings, labels], axis=1)
            # append the filenames as a first column and convert to DataFrame
            embeddings = np.concatenate([np.array(batch_filenames).reshape((-1, 1)), embeddings], axis=1)
            embeddings = pd.DataFrame(data=embeddings, columns=columns)
            # append them to the already extracted ones
            extracted_deep_embeddings = pd.concat([extracted_deep_embeddings, embeddings], axis=0, ignore_index=True)
            # dump the extracted data to the file
            if extraction_idx % 1000 == 0:
                extracted_deep_embeddings.to_csv(os.path.join(output_path, output_filename), index=False,
                                                 header=False, mode="a")
                # clear RAM
                extracted_deep_embeddings = pd.DataFrame(columns=columns)
                gc.collect()
            del loaded_images
            gc.collect()
        # dump remaining data to the file
        extracted_deep_embeddings.to_csv(os.path.join(output_path, output_filename), index=False,
                                         header=False, mode="a")


def main():
    # params
    BATCH_SIZE = 64
    NUM_CLASSES = 5
    path_to_weights = "/work/home/dsu/Model_weights/weights_of_best_models/frame_to_frame_experiments/Pose_model/All_languages/crossentropy_6_2.pt"
    output_path = "/work/home/dsu/NoXi/NoXi_embeddings/All_languages/Pose_model/"
    preprocessing_functions = [T.Resize(size=(256, 256)), convert_image_to_float_and_scale, T.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
                               ]
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # load extractor
    extractor = load_embedding_extractor(path_to_weights, NUM_CLASSES)
    extractor.to(device)
    # cut the last layer from extractor
    extractor = torch.nn.Sequential(list(extractor.children())[0], list(extractor.children())[1][:-1])
    summary(extractor, input_size=(BATCH_SIZE, 3, 256, 256))
    # turn the extractor to the inference mode
    extractor.eval()
    # load data
    train, dev, test = load_NoXi_data_all_languages(train_labels_as_categories=False,
                                                    dev_labels_as_categories=False,
                                                    test_labels_as_categories=False)
    # extract embeddings from train set
    extract_embeddings_from_df_with_paths(df=train, extractor=extractor, device=device, output_path=output_path,
                                          output_filename="embeddings_train.csv",
                                          preprocessing_functions=preprocessing_functions,
                                          num_neurons_last_layer=256, include_labels=True, batch_size=64)

    # extract embeddings from dev set
    extract_embeddings_from_df_with_paths(df=dev, extractor=extractor, device=device, output_path=output_path,
                                          output_filename="embeddings_dev.csv",
                                          preprocessing_functions=preprocessing_functions,
                                          num_neurons_last_layer=256, include_labels=True, batch_size=64)

    # extract embeddings from test set
    extract_embeddings_from_df_with_paths(df=test, extractor=extractor, device=device, output_path=output_path,
                                          output_filename="embeddings_test.csv",
                                          preprocessing_functions=preprocessing_functions,
                                          num_neurons_last_layer=256, include_labels=True, batch_size=64)


if __name__ == "__main__":
    main()
