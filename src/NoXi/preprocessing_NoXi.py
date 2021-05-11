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

from preprocessing.data_preprocessing.video_preprocessing_utils import extract_frames_from_videofile


def extract_faces_from_video(path_to_video:str, path_to_output:str, detector:object)->None:
    # check if output directory exists

    extract_frames_from_videofile