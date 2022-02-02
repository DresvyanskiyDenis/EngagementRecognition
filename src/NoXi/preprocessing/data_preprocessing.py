#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
TODO: write description of module
"""

__author__ = "Denis Dresvyanskiy"
__copyright__ = "Copyright 2021"
__credits__ = ["Denis Dresvyanskiy"]
__maintainer__ = "Denis Dresvyanskiy"
__email__ = "denis.dresvyanskiy@uni-ulm.de"

import glob
from typing import Tuple

import numpy as np
import pandas as pd
import os
import cv2
from PIL import Image

"""from feature_extraction.face_recognition_utils import recognize_the_most_confident_person_retinaFace, \
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
            full_path_for_saving = os.path.join(path_to_output, 'frame_%i.png'%currentframe)
            frame.save(full_path_for_saving)
        else:
            break"""

def extract_faces_from_all_videos_by_paths(path_to_data:str,relative_paths:Tuple[str,...], output_path:str)->None:
    # check if output directory exists
    if not os.path.exists(output_path):
        os.makedirs(output_path, exist_ok=True)
    # create face detector
    detector=load_and_prepare_detector_retinaFace()
    for input_path in relative_paths:
        full_output_path=os.path.join(output_path,input_path.split('.')[0])
        full_input_path=os.path.join(path_to_data, input_path)
        extract_faces_from_video(path_to_video=full_input_path, path_to_output=full_output_path,
        detector=detector, every_n_frame = 5, resize_face= (224, 224))

def generate_rel_paths_to_images_in_all_dirs(path: str, image_format: str = "jpg") -> pd.DataFrame:
    """Generates relative paths to all images with specified format.
       Returns it as a DataFrame

    :param path: str
            path where all images should be found
    :return: pd.DataFrame
            relative paths to images (including filename)
    """
    # define pattern for search (in every dir and subdir the image with specified format)
    pattern = path + "/**/**/*." + image_format
    # searching via this pattern
    abs_paths = glob.glob(pattern)
    # find a relative path to it
    rel_paths = [os.path.relpath(item, path) for item in abs_paths]
    # create from it a DataFrame
    paths_to_images = pd.DataFrame(columns=['rel_path'], data=np.array(rel_paths)[..., np.newaxis])
    # sort procedure to arrange frames in ascending order within one video
    paths_to_images['frame_num']=paths_to_images['rel_path'].apply(lambda x: int(x.split(os.path.sep)[-1].split('.')[0].split('_')[-1]))
    paths_to_images['rel_path']=paths_to_images['rel_path'].apply(lambda x:x[:x.rfind(os.path.sep)])
    paths_to_images=paths_to_images.sort_values(['rel_path','frame_num'], ascending=(True, True))
    paths_to_images['rel_path'] = paths_to_images.apply(lambda x: os.path.join(x['rel_path'],"frame_%i.%s"%(x['frame_num'], image_format)), axis=1)
    paths_to_images=paths_to_images.reset_index(drop=True)
    paths_to_images.drop(columns=['frame_num'], inplace=True)
    # done
    return paths_to_images





if __name__=='__main__':
    # TODO: TEST IT
    path_to_data=r'D:\Noxi_extracted\NoXi\Sessions'
    # form relative paths
    relative_paths_expert=os.listdir(path_to_data)
    relative_paths_expert=tuple(os.path.join(x, 'Expert_video.mp4') for x in relative_paths_expert)
    relative_paths_novice=os.listdir(path_to_data)
    relative_paths_novice=tuple(os.path.join(x, 'Novice_video.mp4') for x in relative_paths_novice)
    relative_paths=relative_paths_expert+relative_paths_novice
    output_path=r'D:\Noxi_extracted\NoXi\extracted_faces'
    extract_faces_from_all_videos_by_paths(path_to_data, relative_paths, output_path)