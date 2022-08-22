import sys

import torchvision.io
from torchinfo import summary

sys.path.extend(["/work/home/dsu/datatools/"])
sys.path.extend(["/work/home/dsu/engagement_recognition_project_server/"])


import torch
import numpy as np
import pandas as pd
import os
import cv2

import torchvision.transforms as T

from src.NoXi.visual_subsystem.pose_subsystem.frame_wise_model.HRNet import load_HRNet_model, modified_HRNet
from src.NoXi.visual_subsystem.pose_subsystem.frame_wise_model.utils import convert_image_to_float_and_scale


def load_embedding_extractor(path_to_weights:str, num_classes:int)->torch.nn.Module:
    # create model and load its weights
    HRNet = load_HRNet_model(device="cuda" if torch.cuda.is_available() else "cpu",
                             path_to_weights="/work/home/dsu/simpleHRNet/pose_hrnet_w32_256x192.pth")
    model = modified_HRNet(HRNet, num_classes=num_classes)
    model.load_state_dict(torch.load(path_to_weights))
    return model

def extract_embeddings_from_images_in_dir(path_to_dir:str, extractor:torch.nn.Module, device, output_path:str,
                                      preprocessing_functions, num_neurons_last_layer:int=256)->None:
    # TODO: CHECK IT
    if not os.path.exists(output_path):
        os.makedirs(output_path, exist_ok=True)
    # create a DataFrame file for embeddings
    columns = ['frame_num']+['embedding_'+str(i) for i in range(num_neurons_last_layer)]
    result_df = pd.DataFrame(columns=columns)
    # go through all images in directory
    with torch.no_grad():
        for filename in os.listdir(path_to_dir):
            if filename.endswith(".png"):
                # load image
                frame = torchvision.io.read_image(os.path.join(path_to_dir, filename))
                # preprocess image
                for preprocessing_function in preprocessing_functions:
                    frame = preprocessing_function(frame)
                # extract embeddings
                frame = torch.unsqueeze(frame, 0)
                embeddings = extractor(frame)
                embeddings = torch.squeeze(embeddings).cpu().numpy()
                # save embeddings to DataFrame
                frame_num = int(filename.split('.')[0])
                frame_num = np.array(frame_num)[..., np.newaxis]
                embeddings = pd.DataFrame(columns=columns, data= np.concatenate((frame_num, embeddings), axis=1))
                result_df = pd.concat([result_df, embeddings], ignore_index=True, axis=0)
    # save embeddings to file
    result_filename = path_to_dir.split(os.path.sep)[-1] + '.csv'
    result_df.to_csv(os.path.join(output_path, result_filename), index=False)

def extract_embeddings_from_subdirs(path_to_dir:str, extractor:torch.nn.Module, device, output_path:str,
                                      preprocessing_functions, num_neurons_last_layer:int=256)->None:
    # TODO: CHECK IT
    if not os.path.exists(output_path):
        os.makedirs(output_path, exist_ok=True)
    # go throught all subdirectories in path_to_dir
    for subdir in os.listdir(path_to_dir):
        if os.path.isdir(os.path.join(path_to_dir, subdir)):
            # extract embeddings from images in subdirectory
            extract_embeddings_from_images_in_dir(os.path.join(path_to_dir, subdir), extractor, device,
                                                  os.path.join(output_path, subdir), preprocessing_functions,
                                                  num_neurons_last_layer)








def main():
    # TODO: CHECK IT
    # params
    BATCH_SIZE = 64
    NUM_CLASSES = 5
    path_to_weights = ""
    preprocessing_functions = [T.Resize(size=(256, 256)),convert_image_to_float_and_scale,T.Normalize(
                                                                                           mean=[0.485, 0.456, 0.406],
                                                                                           std=[0.229, 0.224, 0.225]
                                                                                                     )
                               ]
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # load extractor
    extractor = load_embedding_extractor(path_to_weights, NUM_CLASSES)
    extractor.to(device)
    # cut the last layer from extractor
    extractor = torch.nn.Sequential(*list(extractor.children())[:-1]) # another one solution would be to interchange it with Identity() layer
    summary(extractor, input_size=(BATCH_SIZE, 3, 256, 256))
    # turn the extractor to the inference mode
    extractor.eval()

if __name__ == "__main__":
    main()