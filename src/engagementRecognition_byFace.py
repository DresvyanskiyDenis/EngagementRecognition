#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
TODO: write description of module
"""

from typing import Optional, Tuple

import numpy as np
import pandas as pd
import cv2
import os

from preprocessing.data_preprocessing.image_preprocessing_utils import load_image, save_image, resize_image
from preprocessing.face_recognition_utils import recognize_the_most_confident_person_retinaFace, \
    extract_face_according_bbox, load_and_prepare_detector_retinaFace


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
    detector=load_and_prepare_detector_retinaFace()
    # create output path, if it is not existed
    if not os.path.exists(path_to_output):
        os.makedirs(path_to_output, exist_ok=True)
    # go over all subdirectories
    for subdirectory in subdirectories:
        # inside every subdirectory the subdirectory_frames subsubdirectory is localed. There are frames extracted
        # from video and we need to extract faces from every image
        # construct subdirectory_with_frames variable, which is name of subsubdirectory
        subdirectory_with_frames=subdirectory+'_frames'
        # create name of directory and the directory itself, in which the faces of images will be extracted
        output_dir_name=os.path.join(path_to_output, subdirectory)
        if not os.path.exists(output_dir_name):
            os.makedirs(output_dir_name, exist_ok=True)
        # extract faces from defined subdirectory
        extract_faces_from_dir(path_to_dir=os.path.join(path_to_dir, subdirectory, subdirectory_with_frames),
                               output_dir=output_dir_name, detector=detector, resize=resize)













if __name__ == '__main__':
    path_to_directory_with_frames=r'D:\Databases\DAiSEE\frames'
    path_to_output_directory=r'D:\Databases\DAiSEE\extracted_faces'
    resize=(224,224)
    extract_faces_from_all_subdirectories_in_directory(path_to_directory_with_frames, path_to_output_directory, resize)