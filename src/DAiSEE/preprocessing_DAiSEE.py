#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
TODO: add description
"""
import os
import re
import time
from typing import Optional, Tuple, Callable

import tensorflow as tf
import numpy as np
import cv2
from PIL import Image

from feature_extraction.embeddings_extraction import extract_deep_embeddings_from_images_in_dir
from preprocessing.data_normalizing_utils import VGGFace2_normalization
from preprocessing.data_preprocessing.image_preprocessing_utils import load_image, resize_image, \
    save_image
from preprocessing.face_recognition_utils import recognize_the_most_confident_person_retinaFace, \
    extract_face_according_bbox, load_and_prepare_detector_retinaFace
from tensorflow_utils.models.CNN_models import get_EMO_VGGFace2


def extract_faces_from_video(path_to_video:str, path_to_output:str,
                             detector:object, every_n_frame:int=1,
                             resize_face:Tuple[int, int]=(224,224))->None:
    # TODO: TEST IT
    # TODO: write description
    # check if output directory exists
    if not os.path.exists(path_to_output):
        os.makedirs(path_to_output, exist_ok=True)
    # open videofile
    videofile = cv2.VideoCapture(path_to_video)
    # counter for frames. It equals to -1, because we will start right from 0 (see below)
    currentframe=-1
    filename=re.split(r'\\|/',path_to_video)[-1].split('.')[0]
    while (True):
        # reading from frame
        ret, frame = videofile.read()
        currentframe += 1
        # if currentframe is not integer divisible by every_frame, skip it
        if not currentframe % every_n_frame == 0:
            continue
        if ret:
            # convert to RGB, because opencv reads in BGR format
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # if video is still left continue detect face and save it
            frame=np.array(frame)
            # recognize the face (as the most confident on frame) and get bounding box for it
            bbox=recognize_the_most_confident_person_retinaFace(frame, detector)
            # if no face was found
            if len(bbox)==0:
                continue
            # extract face from image according to boundeng box
            frame=extract_face_according_bbox(frame, bbox)
            # resize if needed
            if resize_face and resize_face[0]!=frame.shape[0] and resize_face[1]!=frame.shape[1]:
                frame=Image.fromarray(frame).resize(resize_face)
            # save extracted face in png format
            full_path_for_saving = os.path.join(path_to_output, filename, '%s_%i.png'%(filename,currentframe))
            frame.save(full_path_for_saving)
        else:
            break




def extract_faces_from_all_subdirectories_in_directory(path_to_dir:str, path_to_output:str,
                                                       resize:Optional[Tuple[int,int]])-> None:
    if not os.path.exists(path_to_output):
        os.makedirs(path_to_output)
    # create face detector
    detector = load_and_prepare_detector_retinaFace()
    subdirectories=os.listdir(path_to_dir)
    counter=0
    for subdir in subdirectories:
        subsubdirs=os.listdir(os.path.join(path_to_dir, subdir))
        # iterate through subsubdirs
        for subsubdir in subsubdirs:
            full_path_to_videofilename=os.path.join(path_to_dir, subdir, subsubdir, subsubdir+'.avi')
            full_output_path=os.path.join(path_to_output, subsubdir)
            if not os.path.exists(full_output_path):
                os.makedirs(full_output_path, exist_ok=True)
            extract_faces_from_video(path_to_video=full_path_to_videofilename, path_to_output=path_to_output,
            detector=detector, every_n_frame=5, resize_face=resize)
        print('subdirectory number %i processed. Remains:%i.'%(counter, len(subdirectories)-counter))
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
    path_to_data=r"E:\Databases\DAiSEE\DAiSEE\DataSet\Test"
    path_to_output=r"E:\Databases\DAiSEE\DAiSEE\test_preprocessed"
    extract_faces_from_all_subdirectories_in_directory(path_to_dir=path_to_data, path_to_output=path_to_output, resize=(224,224))
    '''# just for testing
    path_to_images=r'D:\\Databases\\DAiSEE\\DAiSEE\\train_preprocessed\extracted_faces'
    # create model
    model=get_EMO_VGGFace2(path='C:\\Users\\Dresvyanskiy\\Desktop\\Projects\\EMOVGGFace_model\\weights_0_66_37_affectnet_cat.h5')
    emb_layer=model.get_layer('dense')
    model=tf.keras.Model(inputs=model.inputs, outputs=[emb_layer.output])
    model.compile()
    extract_deep_embeddings_from_all_dirs(path_to_dirs=path_to_images, extractor=model,
                                          preprocessing_functions=(VGGFace2_normalization,),
                                          output_path='D:\\Databases\\DAiSEE\\DAiSEE')
    '''
    pass