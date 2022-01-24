import numpy as np
import pandas as pd
import tensorflow as tf
import os
import glob



def generate_rel_paths_to_images_in_all_dirs(path:str, image_format:str="jpg")->pd.DataFrame:
    """Generates relative paths to all images with specified format.
       Returns it as a DataFrame

    :param path: str
            path where all images should be found
    :return: pd.DataFrame
            relative paths to images (including filename)
    """
    # define pattern for search (in every dir and subdir the image with specified format)
    pattern = path+r"\**\*."+image_format
    # searching via this pattern
    abs_paths=glob.glob(pattern)
    # find a relative path to it
    rel_paths=[os.path.relpath(item, path) for item in abs_paths]
    # create from it a DataFrame
    paths_to_images = pd.DataFrame(columns=['rel_path'], data=np.array(rel_paths))
    return paths_to_images




def main():
    path_to_data=""
    path_to_labels=""

    pass




if __name__ == '__main__':
    print(generate_rel_paths_to_images_in_all_dirs('D:\Downloads'))