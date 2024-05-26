from collections import OrderedDict
from typing import Callable, List, Dict, Union, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset




class MultiModalTemporalDataLoader(Dataset):

    def __init__(self, embeddings_with_labels:List[Dict[str, pd.DataFrame]], feature_columns:List[str], label_columns:List[str],
                 window_size:Union[int, float], stride:Union[int, float],
                 consider_timesteps:Optional[bool]=False, timesteps_column:Optional[str]=None,
                 preprocessing_functions:List[Callable]=None, shuffle:bool=False):
        super().__init__()
        self.embeddings_with_labels = embeddings_with_labels
        self.feature_columns = feature_columns
        self.label_columns = label_columns
        self.window_size = window_size
        self.stride = stride
        self.consider_timesteps = consider_timesteps
        self.timesteps_column = timesteps_column
        self.preprocessing_functions = preprocessing_functions
        self.shuffle = shuffle


    def unite_dataframes(self):
        """

        """
