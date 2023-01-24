#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Contains the functions for the IWSDS2023 data preprocessing.

"""

__author__ = "Denis Dresvyanskiy"
__copyright__ = "Copyright 2022"
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


def extract_faces_from_all_videos_by_paths(path_to_data:str,relative_paths:Tuple[str,...], output_path:str)->None:
    """ Extracts faces from all video using provided path_to_data (path where all videos are) and relative paths to each video

    :param path_to_data: str
                General (common) path to the directory, where there are all videos
    :param relative_paths: Tuple[str,...]
                Tuple of relative paths, while the "starting point" is the path_to_data
    :param output_path: str
                Path to the output directory
    :return: None
    """
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
    path_to_data=r'D:\Noxi_extracted\NoXi\Sessions'
    # form relative paths
    relative_paths_expert=os.listdir(path_to_data)
    relative_paths_expert=tuple(os.path.join(x, 'Expert_video.mp4') for x in relative_paths_expert)
    relative_paths_novice=os.listdir(path_to_data)
    relative_paths_novice=tuple(os.path.join(x, 'Novice_video.mp4') for x in relative_paths_novice)
    relative_paths=relative_paths_expert+relative_paths_novice
    output_path=r'D:\Noxi_extracted\NoXi\extracted_faces'
    extract_faces_from_all_videos_by_paths(path_to_data, relative_paths, output_path)