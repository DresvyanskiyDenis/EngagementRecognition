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

import subprocess

def extract_audio_from_videofile(path:str, output_path:str, sample_rate:int=16000)->None:
    """Extract audio in waveform from videofile specified by path.

    :param path: str
            Path to videofile
    :param output_path: str
            Path for saving audiofile
    :param sample_rate: int
            Sample rate of audiofile
    :return:
    """
    command = "ffmpeg -i %s -ab 160k -ac 2 -ar %s -vn %s"%(path, str(sample_rate), output_path)
    subprocess.call(command, shell=True)

