#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
TODO: write description of module
"""
import math
import shutil
import time
from functools import partial
from typing import Optional, Tuple, Dict, NamedTuple, Iterable, List

import numpy as np
import pandas as pd
import os
import cv2
from PIL.Image import Image

from preprocessing.data_preprocessing.image_preprocessing_utils import save_image
from preprocessing.data_preprocessing.video_preprocessing_utils import extract_frames_from_videofile
from preprocessing.face_recognition_utils import recognize_the_most_confident_person_retinaFace, \
    extract_face_according_bbox, load_and_prepare_detector_retinaFace


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
            # extract face from image according to boundeng box
            frame=extract_face_according_bbox(frame, bbox)
            # resize if needed
            if resize_face and resize_face[0]!=frame.shape[0] and resize_face[1]!=frame.shape[1]:
                frame=Image.fromarray(frame).resize(resize_face)
            # save extracted face in png format
            full_path_for_saving = os.path.join(path_to_output, 'frame_%i.png'%str(currentframe))
            save_image(frame, path_to_output=full_path_for_saving)
        else:
            break

def extract_faces_from_all_videos_by_paths(path_to_data:str,relative_paths:Tuple[str], output_path:str)->None:
    # TODO: TEST IT
    # check if output directory exists
    if not os.path.exists(output_path):
        os.makedirs(output_path, exist_ok=True)
    # create face detector
    detector=load_and_prepare_detector_retinaFace()
    for input_path in relative_paths:
        full_output_path=os.path.join(output_path,input_path)
        full_input_path=os.path.join(path_to_data, input_path)
        extract_faces_from_video(path_to_video=full_input_path, path_to_output=full_output_path,
        detector=detector, every_n_frame = 5, resize_face= (224, 224))



if __name__=='__main__':
    # TODO: TEST IT
    path_to_data=''
    # form relative paths
    relative_paths=[]
    output_path=[]