import gc
import glob
from typing import Optional, Tuple

import pandas as pd
import numpy as np
import os

import torch

from src.IWSDS2023.preprocessing.labels_preprocessing import generate_paths_to_labels, load_all_labels_by_paths, \
    combine_path_to_images_with_labels_many_videos


def generate_rel_paths_to_images_in_all_dirs(path: str, image_format: str = "jpg") -> pd.DataFrame:
    """Generates relative paths to all images with specified format.
       Returns it as a DataFrame

    :param path: str
            path where all images should be found
    :return: pd.DataFrame
            relative paths to images (including filename)
    """
    # define pattern for search (in every dir and subdir the image with specified format)
    pattern = path + "**/**/*." + image_format
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



def load_data_one_language(path_to_data: str, path_to_labels: str, frame_step: int, shuffle_train:Optional[bool]=False,
                             train_labels_as_categories:bool=False, dev_labels_as_categories:bool=False, test_labels_as_categories:bool=False):
    # generate paths to images (data)
    paths_to_images = generate_rel_paths_to_images_in_all_dirs(path_to_data, image_format="png")
    # generate paths to train/dev/test labels
    paths_train_labels = generate_paths_to_labels(os.path.join(path_to_labels, "train"))
    paths_dev_labels = generate_paths_to_labels(os.path.join(path_to_labels, "dev"))
    paths_test_labels = generate_paths_to_labels(os.path.join(path_to_labels, "test"))
    # load labels
    train_labels = load_all_labels_by_paths(paths_train_labels)
    dev_labels = load_all_labels_by_paths(paths_dev_labels)
    test_labels = load_all_labels_by_paths(paths_test_labels)
    del paths_train_labels, paths_dev_labels, paths_test_labels
    # change the keys of train_labels/dev_labels/test_labels to have only the name with pattern name_of_video/novice_or_expert
    for key in list(train_labels.keys()):
        new_key = key.split(os.path.sep)[-2] + '/'
        new_key = new_key + 'expert' if 'expert' in key.split(os.path.sep)[-1] else new_key + 'novice'
        train_labels[new_key] = train_labels.pop(key)
    for key in list(dev_labels.keys()):
        new_key = key.split(os.path.sep)[-2] + '/'
        new_key = new_key + 'expert' if 'expert' in key.split(os.path.sep)[-1] else new_key + 'novice'
        dev_labels[new_key] = dev_labels.pop(key)
    for key in list(test_labels.keys()):
        new_key = key.split(os.path.sep)[-2] + '/'
        new_key = new_key + 'expert' if 'expert' in key.split(os.path.sep)[-1] else new_key + 'novice'
        test_labels[new_key] = test_labels.pop(key)
    # combine paths to images (data) with labels
    train_image_paths_and_labels = combine_path_to_images_with_labels_many_videos(paths_with_images=paths_to_images,
                                                                                  labels=train_labels,
                                                                                  sample_rate_annotations=25,
                                                                                  frame_step=frame_step)
    dev_image_paths_and_labels = combine_path_to_images_with_labels_many_videos(paths_with_images=paths_to_images,
                                                                                labels=dev_labels,
                                                                                sample_rate_annotations=25,
                                                                                frame_step=5)
    test_image_paths_and_labels = combine_path_to_images_with_labels_many_videos(paths_with_images=paths_to_images,
                                                                                 labels=test_labels,
                                                                                 sample_rate_annotations=25,
                                                                                 frame_step=5)
    # shuffle train data
    if shuffle_train:
        train_image_paths_and_labels = train_image_paths_and_labels.sample(frac=1).reset_index(drop=True)
    # convert dev and test labels to the categories (it is easier to process them like this)
    if train_labels_as_categories:
        train_labels = np.argmax(train_image_paths_and_labels.iloc[:, 1:].values, axis=1, keepdims=True)
        train_image_paths_and_labels = train_image_paths_and_labels.iloc[:, :1]
        train_image_paths_and_labels['class'] = train_labels

    if dev_labels_as_categories:
        dev_labels = np.argmax(dev_image_paths_and_labels.iloc[:, 1:].values, axis=1, keepdims=True)
        dev_image_paths_and_labels = dev_image_paths_and_labels.iloc[:, :1]
        dev_image_paths_and_labels['class'] = dev_labels

    if test_labels_as_categories:
        test_labels = np.argmax(test_image_paths_and_labels.iloc[:, 1:].values, axis=1, keepdims=True)
        test_image_paths_and_labels = test_image_paths_and_labels.iloc[:, :1]
        test_image_paths_and_labels['class'] = test_labels
    # create abs path for all paths instead of relative (needed for generator)
    train_image_paths_and_labels['filename'] = train_image_paths_and_labels['filename'].apply(
        lambda x: os.path.join(path_to_data, x))
    dev_image_paths_and_labels['filename'] = dev_image_paths_and_labels['filename'].apply(
        lambda x: os.path.join(path_to_data, x))
    test_image_paths_and_labels['filename'] = test_image_paths_and_labels['filename'].apply(
        lambda x: os.path.join(path_to_data, x))
    # done
    return (train_image_paths_and_labels, dev_image_paths_and_labels, test_image_paths_and_labels)


def load_NoXi_data_all_languages(train_labels_as_categories:bool=False, dev_labels_as_categories:bool=False,
                                 test_labels_as_categories:bool=False)->Tuple[
    pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """ Exploits the load_and_preprocess_data() function to form DataFrames for every language in the IWSDS2023 dataset
        (French, German, English)


    :param labels_as_categories: bool
            convert labels to the categories represented by one number or keep it in the one-hot encoding format
    :return: Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
                Tuple of three DataFrames (train, dev, test), with the first column equalled to paths to frames (images)
                and labels as a last column (or columns if labels_as_categories equals True)
    """
    # loading data
    frame_step = 5
    path_to_data = "/Pose_frames_256/"
    path_to_labels_french = "/media/external_hdd_1/NoXi_annotations_reliable_gold_standard_classification_with_additional_train_data/French"
    path_to_labels_german = "/media/external_hdd_1/NoXi_annotations_reliable_gold_standard_classification_with_additional_train_data/German"
    path_to_labels_english = "/media/external_hdd_1/NoXi_annotations_reliable_gold_standard_classification_with_additional_train_data/English"
    # load french data
    train_french, dev_french, test_french = load_data_one_language(path_to_data, path_to_labels_french, frame_step,
                                                                   train_labels_as_categories=train_labels_as_categories,
                                                                   dev_labels_as_categories=dev_labels_as_categories,
                                                                   test_labels_as_categories=test_labels_as_categories)
    # load english data
    train_german, dev_german, test_german = load_data_one_language(path_to_data, path_to_labels_german, frame_step,
                                                                   train_labels_as_categories=train_labels_as_categories,
                                                                   dev_labels_as_categories=dev_labels_as_categories,
                                                                   test_labels_as_categories=test_labels_as_categories)
    # load german data
    train_english, dev_english, test_english = load_data_one_language(path_to_data, path_to_labels_english, frame_step,
                                                                   train_labels_as_categories=train_labels_as_categories,
                                                                   dev_labels_as_categories=dev_labels_as_categories,
                                                                   test_labels_as_categories=test_labels_as_categories)
    # concatenate all data
    train = pd.concat([train_french, train_german, train_english], axis=0)
    dev = pd.concat([dev_french, dev_german, dev_english], axis=0)
    test = pd.concat([test_french, test_german, test_english], axis=0)
    # clear RAM
    del train_english, train_french, train_german
    del dev_english, dev_french, dev_german
    del test_english, test_german, test_french
    gc.collect()

    return (train, dev, test)

def load_NoXi_data_cross_corpus(test_corpus:str, train_labels_as_categories:bool=False, dev_labels_as_categories:bool=False,
                                 test_labels_as_categories:bool=False)->Tuple[
    pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # loading data
    frame_step = 5
    path_to_data = "/Pose_frames_256/"
    path_to_labels_french = "/media/external_hdd_1/NoXi_annotations_reliable_gold_standard_classification_with_additional_train_data/French"
    path_to_labels_german = "/media/external_hdd_1/NoXi_annotations_reliable_gold_standard_classification_with_additional_train_data/German"
    path_to_labels_english = "/media/external_hdd_1/NoXi_annotations_reliable_gold_standard_classification_with_additional_train_data/English"
    # load french data
    train_french, dev_french, test_french = load_data_one_language(path_to_data, path_to_labels_french, frame_step,
                                                                   train_labels_as_categories=train_labels_as_categories,
                                                                   dev_labels_as_categories=dev_labels_as_categories,
                                                                   test_labels_as_categories=test_labels_as_categories)
    # load english data
    train_german, dev_german, test_german = load_data_one_language(path_to_data, path_to_labels_german, frame_step,
                                                                   train_labels_as_categories=train_labels_as_categories,
                                                                   dev_labels_as_categories=dev_labels_as_categories,
                                                                   test_labels_as_categories=test_labels_as_categories)
    # load german data
    train_english, dev_english, test_english = load_data_one_language(path_to_data, path_to_labels_english, frame_step,
                                                                      train_labels_as_categories=train_labels_as_categories,
                                                                      dev_labels_as_categories=dev_labels_as_categories,
                                                                      test_labels_as_categories=test_labels_as_categories)


    french = {'train': train_french, 'dev': dev_french, 'test': test_french}
    german = {'train': train_german, 'dev': dev_german, 'test': test_german}
    english = {'train': train_english, 'dev': dev_english, 'test': test_english}
    all_languages = {'french': french, 'german': german, 'english': english}

    # pick the language for test (leave-one-out procedure)
    test = all_languages.pop(test_corpus)
    test = pd.concat(list(test.values()), axis=0)
    # take as a dev set - developments sets of other corpora
    dev = [item.pop('dev') for item in all_languages.values()]
    dev = pd.concat(list(dev), axis=0)
    # take everything else as a training set
    train = [language['train'] for language in all_languages.values()] + [language['test'] for language in
                                                                      all_languages.values()]
    train = pd.concat(train, axis=0)
    # clear RAM
    del train_english, train_french, train_german
    del dev_english, dev_french, dev_german
    del test_english, test_german, test_french
    gc.collect()

    return train, dev, test




def convert_image_to_float_and_scale(image:torch.Tensor)->torch.Tensor:
    image = image.float() / 255.
    return image



if __name__ == "__main__":
    print("Loading data...")
    train, dev, test = load_NoXi_data_cross_corpus(test_corpus='english', train_labels_as_categories=False,
                                                    dev_labels_as_categories=False,
                                                    test_labels_as_categories=False)
    print(train.shape)
    print(dev.shape)
    print(test.shape)
