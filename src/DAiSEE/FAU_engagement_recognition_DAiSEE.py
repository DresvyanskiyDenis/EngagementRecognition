import numpy as np
import os
import pandas as pd
import shutil

from preprocessing.data_preprocessing.openFace_utils import extract_openface_FAU_from_images_in_dir


def extract_FAU_features_from_all_subdirectories(path_to_dirs:str, path_to_extractor:str)->pd.DataFrame:
    # find all subdirectories in provided directory
    subdirectories=os.listdir(path_to_dirs)
    # create empty list to collect then all extracted features
    extracted_features=[]
    counter=0
    for subdirectory in subdirectories:
        # construct full path to subdirectory
        path_to_images=os.path.join(path_to_dirs, subdirectory)
        # extract features by OpenFace
        features=extract_openface_FAU_from_images_in_dir(path_to_images, path_to_extractor)
        if features is not None:
            extracted_features.append(features)
        print('----------------------------------%i, %i'%(counter, len(subdirectories)))
        counter+=1
    # concat all extracted features
    extracted_features=pd.concat(extracted_features, axis=0)
    return extracted_features



if __name__=="__main__":
    path_to_dir = r'E:\Databases\DAiSEE\DAiSEE\train_preprocessed\extracted_faces'
    path_to_extractor = r'C:\Users\Denis\PycharmProjects\OpenFace\FaceLandmarkImg.exe'
    features = extract_FAU_features_from_all_subdirectories(path_to_dir, path_to_extractor)
    features.to_csv(r'E:\Databases\DAiSEE\DAiSEE\dev_processed\FAU_features.csv', index=False)
    a = 1 + 2