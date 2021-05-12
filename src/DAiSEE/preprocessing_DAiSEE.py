#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
TODO: add description
"""
import os
import time
from typing import Optional, Tuple, Callable

import tensorflow as tf
import numpy as np

from feature_extraction.embeddings_extraction import extract_deep_embeddings_from_images_in_dir
from preprocessing.data_normalizing_utils import VGGFace2_normalization
from preprocessing.data_preprocessing.image_preprocessing_utils import load_image, resize_image, \
    save_image
from preprocessing.face_recognition_utils import recognize_the_most_confident_person_retinaFace, \
    extract_face_according_bbox, load_and_prepare_detector_retinaFace
from tensorflow_utils.models.CNN_models import get_EMO_VGGFace2


def extract_faces_from_dir(path_to_dir:str, output_dir:str,detector:object, resize:Optional[Tuple[int,int]])->None:
    # TODO: add description
    # get filenames of all images in dir
    image_filenames=os.listdir(path_to_dir)
    # create, if necessary, output dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    # extract faces from each image
    for image_filename in image_filenames:
        # load image by PIL.Image
        img=load_image(os.path.join(path_to_dir,image_filename))
        # calculate bounding boxes for the most confident person via provided detector
        bbox=recognize_the_most_confident_person_retinaFace(img, detector)
        bbox=tuple(max(0, _) for _ in bbox)
        if bbox is None or len(bbox)==0:
            continue
        # extract face from image according bbox
        face_img=extract_face_according_bbox(img, bbox) # it is in RGB format now
        # resize if needed
        if not resize is None:
            face_img=resize_image(face_img, resize)
        # save
        save_image(face_img, path_to_output=os.path.join(output_dir, image_filename))


def extract_faces_from_all_subdirectories_in_directory(path_to_dir:str, path_to_output:str,
                                                       resize:Optional[Tuple[int,int]])-> None:
    # TODO: add description
    # get subdirectories from provided dir
    subdirectories=os.listdir(path_to_dir)
    # load retinaFace face detector
    detector=load_and_prepare_detector_retinaFace('retinaface_mnet025_v2')
    # create output path, if it is not existed
    if not os.path.exists(path_to_output):
        os.makedirs(path_to_output, exist_ok=True)
    # go over all subdirectories
    counter=0
    overall=len(subdirectories)
    for subdirectory in subdirectories:
        # inside every subdirectory the subdirectory_frames subsubdirectory is localed. There are frames extracted
        # from video and we need to extract faces from every image
        # construct subdirectory_with_frames variable, which is name of subsubdirectory
        subdirectory_with_frames=subdirectory+'_frames'
        # create name of directory and the directory itself, in which the faces of images will be extracted
        output_dir_name=os.path.join(path_to_output, subdirectory)
        if not os.path.exists(output_dir_name):
            os.makedirs(output_dir_name, exist_ok=True)
        else:
            counter += 1
            continue
        # extract faces from defined subdirectory
        start=time.time()
        extract_faces_from_dir(path_to_dir=os.path.join(path_to_dir, subdirectory, subdirectory_with_frames),
                               output_dir=output_dir_name, detector=detector, resize=resize)
        print('processed:%i, overall:%i, time:%f'%(counter, overall, time.time()-start))
        counter+=1

def extract_deep_embeddings_from_all_dirs(path_to_dirs:str, extractor:tf.keras.Model,
                                          preprocessing_functions:Tuple[Callable[[np.ndarray], np.ndarray], ...]=None,
                                          output_path:str='results')->None:
    # TODO: add description
    # check if output path is existed
    if not os.path.exists(output_path):
        os.makedirs(output_path, exist_ok=True)
    subdirs=os.listdir(path_to_dirs)
    # generate first df for appending other ones to it then
    result_df=extract_deep_embeddings_from_images_in_dir(path_to_dir=os.path.join(path_to_dirs, subdirs[0]),
                                                         extractor=extractor,
                                                         preprocessing_functions=preprocessing_functions)
    # iterate through all subdirectories
    counter=0
    for subdir_idx in range(1, len(subdirs)):
        start_time = time.time()
        curr_df=extract_deep_embeddings_from_images_in_dir(path_to_dir=os.path.join(path_to_dirs, subdirs[subdir_idx]),
                                                         extractor=extractor,
                                                         preprocessing_functions=preprocessing_functions)
        result_df.append(curr_df)
        end_time=time.time()
        print('Subdirectory %s is processed. Time: %f. Remains:%i'%(subdirs[subdir_idx], end_time-start_time, len(subdirs)-1-counter))
        counter+=1
    # save obtained dataframe
    result_df.to_csv(os.path.join(output_path,'deep_embeddings_from_EMOVGGFace2.csv'), index=False)




if __name__=='__main__':
    # just for testing
    path_to_images=r'D:\Databases\DAiSEE\DAiSEE\train_preprocessed\extracted_faces'
    # create model
    model=get_EMO_VGGFace2(path=r'C:\Users\Dresvyanskiy\Desktop\Projects\EMOVGGFace_model\weights_0_66_37_affectnet_cat.h5')
    emb_layer=model.get_layer('dense')
    model=tf.keras.Model(inputs=model.inputs, outputs=[emb_layer.output])
    model.compile()
    extract_deep_embeddings_from_all_dirs(path_to_dirs=path_to_images, extractor=model,
                                          preprocessing_functions=(VGGFace2_normalization,),
                                          output_path=r'D:\Databases\DAiSEE\DAiSEE')