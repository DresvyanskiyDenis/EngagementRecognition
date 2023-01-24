import sys
sys.path.extend(["/work/home/dsu/datatools/"])
sys.path.extend(["/work/home/dsu/engagement_recognition_project_server/"])
sys.path.extend(["/work/home/dsu/simpleHigherHRNet/"])


import PIL


from SimpleHigherHRNet import SimpleHigherHRNet
from decorators.common_decorators import timer
import glob
from typing import Optional, Union, Tuple, List

import numpy as np
import cv2
import os



def check_bbox_length(bbox)->bool:
    if bbox[3]-bbox[1]>800:
        return True
    else:
        return False

def apply_bbox_to_frame(frame, bbox)->np.ndarray:
    return frame[bbox[1]:bbox[3], bbox[0]:bbox[2]]


def cut_frame_to_pose(extractor, frame:np.ndarray, return_bbox:bool=False)->Union[Tuple[np.ndarray, List[int]],
                                                                                  np.ndarray,
                                                                                  None]:
    height, width, _ = frame.shape
    prediction = extractor.predict(frame)
    if prediction is None or len(prediction[0]) == 0:
        return None
    bbox = prediction[0][0]
    # expand bbox so that it will cover all human with some space
    # height
    bbox[1] -= 125
    bbox[3] +=125
    # width
    bbox[0] -= 100
    bbox[2] += 100
    # check if we are still in the frame
    if bbox[1] < 0:
        bbox[1] = 0
    if bbox[3] > height:
        bbox[3] = height
    if bbox[0] < 0:
        bbox[0] = 0
    if bbox[2] > width:
        bbox[2] = width
    # cut frame
    bbox = [int(x) for x in bbox]
    cut_frame = apply_bbox_to_frame(frame, bbox)
    if return_bbox:
        return cut_frame, bbox
    return cut_frame

@timer
def extract_frames_with_poses_from_one_video(path_to_videofile:str, extractor, output_path:str, every_n_frame:int=5)->None:

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    video_filename = os.path.basename(path_to_videofile)
    video_filename = os.path.splitext(video_filename)[0]

    if not os.path.exists(os.path.join(output_path, video_filename)):
        os.makedirs(os.path.join(output_path, video_filename))

    reader = cv2.VideoCapture(path_to_videofile)
    num_frame=0

    bbox = None
    while (reader.isOpened()):
        ret, frame = reader.read()
        if frame is None:
            break

        if num_frame % every_n_frame == 0:
            previous_bbox = None if bbox is None else bbox
            result = cut_frame_to_pose(extractor, frame, return_bbox=True)
            # check if the model has found anything. THe function cut_frame_to_pose() produces None if nothing has been found.
            if result is None: continue
            else: cut_frame, bbox = result
            # check if the width of the bbox is too large (this can happen, the model sometimes produces wrong bboxes for the whole frame)
            # if it is so, apply previous bbox (from previous frame) to the current frame
            if check_bbox_length(bbox):
                if previous_bbox is not None:
                    cut_frame = apply_bbox_to_frame(frame, previous_bbox)
                else:
                    continue
            # construct output filename for cut frame
            output_filename = os.path.join(output_path, video_filename, str(num_frame) + '.png')
            # save the image (frame)
            cv2.imwrite(output_filename, cut_frame)
        # print the progress every 100 processed frames
        if num_frame % 100 == 0:
            print('Number of preprocessed frames: %i, remaining: %i' % (num_frame, reader.get(cv2.CAP_PROP_FRAME_COUNT) - num_frame))
        num_frame += 1

    reader.release()
    cv2.destroyAllWindows()

def extract_frames_with_poses_from_all_videos(path_to_dir:str, extractor, output_path:str, every_n_frame:int=5)->None:

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    all_videos_in_dir = glob.glob(os.path.join(path_to_dir, '*', '*.mp4'))
    for video_file in all_videos_in_dir:
        print("Processing %s file..."%video_file)
        full_output_path = os.path.join(output_path,
                                        video_file.split(os.path.sep)[-2])
        # check if this video file was already preprocessed before
        if os.path.exists(os.path.join(full_output_path, os.path.basename(video_file).split('.')[0])):
            print("Video file %s was already preprocessed, skipping..."%video_file)
            continue
        extract_frames_with_poses_from_one_video(video_file, extractor, full_output_path, every_n_frame)

def rename_files_to_align_to_other_data(path_to_dir:str)->None:
    all_files_in_dir = glob.glob(os.path.join(path_to_dir, "*.png"))
    for file in all_files_in_dir:
        if "frame_" in file:
            continue
        new_filename = file.split(os.path.sep)[-1]
        new_filename = "frame_"+new_filename
        new_filename = os.path.join(path_to_dir, new_filename)
        os.rename(file, new_filename)

def rename_files_in_all_dirs(path_to_data:str):
    directories = glob.glob(os.path.join(path_to_data, '*', '*'))
    for directory in directories:
        rename_files_to_align_to_other_data(directory)

def load_resize_and_save_image(path_to_image:str, output_path:str, new_size:Tuple[int, int]=(256,256))->None:
    image = PIL.Image.open(path_to_image)
    image = image.resize(new_size, PIL.Image.ANTIALIAS)
    image.save(output_path)

@timer
def load_resize_and_save_images_in_directory(path_to_dir:str, output_path:str, new_size:Tuple[int, int]=(256,256))->None:
    if not os.path.exists(output_path):
        os.makedirs(output_path, exist_ok=True)
    all_files_in_dir = glob.glob(os.path.join(path_to_dir, "*.png"))
    for filename in all_files_in_dir:
        output_filename = os.path.join(output_path, os.path.basename(filename))
        load_resize_and_save_image(filename, output_filename, new_size)


def load_resize_and_save_images_in_all_dirs(path_to_data:str, output_dir:str, new_size:Tuple[int, int]=(256,256))->None:
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    directories = glob.glob(os.path.join(path_to_data, '*', "*"))
    for num, directory in enumerate(directories):
        print("Processing %i/%i directory... name:%s"%(num+1, len(directories), directory))
        output_path = os.path.join(output_dir, *directory.split(os.path.sep)[-2:])
        if not os.path.exists(output_path):
            os.makedirs(output_path, exist_ok=True)
        load_resize_and_save_images_in_directory(directory, output_path, new_size)





if __name__== '__main__':
    print("start of the main sector111...")
    #model = SimpleHigherHRNet(c=32, nof_joints=17,
    #                          checkpoint_path="/work/home/dsu/simpleHigherHRNet/pose_higher_hrnet_w32_512.pth",
    #                          return_heatmaps=False, return_bounding_boxes=True, max_batch_size=1,
    #                          device="cuda")
    print("done...")
    #extract_frames_with_poses_from_one_video(r'/media/external_hdd_1/IWSDS2023/Sessions/013_2016-03-30_Paris/Novice_video.mp4',
    #                                         model,
    #                                         r'/media/external_hdd_1/IWSDS2023/Pose_frames/013_2016-03-30_Paris/Novice_video/')
    #print("start extracting poses...")
    #extract_frames_with_poses_from_all_videos('/media/external_hdd_1/IWSDS2023/Sessions/',
    #                                          model,
    #                                          '/media/external_hdd_1/IWSDS2023/Pose_frames/')
    #print("start renaming files....")
    #rename_files_in_all_dirs('/media/external_hdd_1/IWSDS2023/Pose_frames/')
    print("start loading and resizing images....")
    load_resize_and_save_images_in_all_dirs(path_to_data='/media/external_hdd_1/IWSDS2023/Pose_frames/',
                                            output_dir='/media/external_hdd_1/Pose_frames_256/',
                                            new_size=(256,256))