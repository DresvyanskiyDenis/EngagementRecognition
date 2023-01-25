import sys
sys.path.extend(["/work/home/dsu/datatools/"])
sys.path.extend(["/work/home/dsu/engagement_recognition_project_server/"])
sys.path.extend(["/work/home/dsu/simpleHigherHRNet/"])

import gc
import glob
import os
from typing import List, Optional

import torch
import cv2
import pandas as pd
import numpy as np
from PIL import Image

from SimpleHigherHRNet import SimpleHigherHRNet
from decorators.common_decorators import timer
from feature_extraction.face_recognition_utils import recognize_one_face_bbox, extract_face_according_bbox, \
    load_and_prepare_detector_retinaFace_mobileNet
from feature_extraction.pose_recognition_utils import get_pose_bbox, crop_frame_to_pose


@timer
def crop_faces_in_video(path_to_video:str, output_path:str, detector:object, final_FPS:int)->pd.DataFrame:
    """
    Crops faces from video  using provided detector and saves them to the provided path.
    :param path_to_video: str
            path to the video, which should be cropped
    :param output_path: str
            path to the folder, where the cropped images should be saved
    :param detector: object
            the model, which has method detect. It should return bounding boxes.
    :param final_FPS: int
            FPS of the cropped video (some grames can be skipped).
    :return: pd.DataFrame
            DataFrame with information about the cropped images
    """
    # create output folder if needed
    if not os.path.exists(output_path):
        os.makedirs(output_path, exist_ok=True)
    # metadata
    metadata = pd.DataFrame(columns=["path_to_frame", "timestamp", "detected"])
    # load video file
    video = cv2.VideoCapture(path_to_video)
    # get FPS
    FPS = video.get(cv2.CAP_PROP_FPS)
    FPS_in_seconds = 1. / FPS
    # go through all frames
    counter = 0
    while video.isOpened():
        ret, frame = video.read()
        if ret:
            if counter % final_FPS == 0:
                # convert BGR to RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # calculate timestamp
                timestamp = counter * FPS_in_seconds
                # round it to 2 digits to make it readable
                timestamp = round(timestamp, 2)
                # recognize the face
                bbox = recognize_one_face_bbox(frame, detector)
                # if not recognized, note it as NaN
                if bbox is None:
                    output_filename = np.NaN
                    detected = False
                else:
                    # otherwise, extract the face and save it
                    detected = True
                    face = extract_face_according_bbox(frame, bbox)
                    output_filename = os.path.join(output_path, path_to_video.split(os.path.sep)[-1].split(".")[0]
                                                   + f"_%s.png" % (str(timestamp).replace(".", "_")))
                    # save extracted face
                    Image.fromarray(face).save(output_filename)
                metadata = pd.concat([metadata,
                                      pd.DataFrame.from_records([{
                                          "path_to_frame": output_filename,
                                          "timestamp": timestamp,
                                          "detected": detected}
                                      ])
                                      ], ignore_index=True)
            # increment counter
            counter += 1
        else:
            break
    return metadata

def crop_faces_in_all_videos(paths_to_videos:List[str], output_path:str, detector:object, final_FPS:int,
                             positions_for_output_path:Optional[int]=1)->pd.DataFrame:
    """
    Crops faces from all videos in the provided folder using provided detector and saves them to the output_path.
    :param paths_to_videos: str
            List of paths to the videos, which should be cropped
    :param output_path: str
            path to the folder, where the cropped images should be saved. Every video will have its own subfolder.
    :param detector: object
            the model, which has method detect. It should return bounding boxes.
    :param final_FPS: int
            FPS of the cropped video (some grames can be skipped).
    :param positions_for_output_path: int
            number of subfolder positions from the path to the video, which should be used to create the output path.
    :return: pd.DataFrame
            DataFrame with information about the cropped images
    """
    # create output folder if needed
    if not os.path.exists(output_path):
        os.makedirs(output_path, exist_ok=True)
    # metadata
    metadata = pd.DataFrame(columns=["path_to_frame", "timestamp", "detected"])
    # go through all videos
    counter = 0
    for path_to_video in paths_to_videos:
        # create subfolder for the video
        subfolder = path_to_video.split(os.path.sep)[-positions_for_output_path:]
        subfolder[-1] = subfolder[-1].split(".")[0]
        subfolder = os.path.join(output_path, os.path.sep.join(subfolder))
        if not os.path.exists(subfolder):
            os.makedirs(subfolder, exist_ok=True)
        # crop faces
        video_metadata = crop_faces_in_video(path_to_video, subfolder, detector, final_FPS)
        metadata = pd.concat([metadata, video_metadata], ignore_index=True)
        # increment counter
        counter += 1
        print("Video %s has been processed. The number of remained videos is: %i" % (path_to_video, len(paths_to_videos) - counter))
    return metadata


@timer
def crop_pose_in_video(path_to_video:str, output_path:str, detector:object, final_FPS:int)->pd.DataFrame:
    """
    Crops poses from video using provided detector and saves them to the output_path.
    :param path_to_video: str
            path to the video, poses from which should be cropped
    :param output_path: str
            path to the folder, where the cropped images should be saved
    :param detector: object
            the model, which has method detect. It should return bounding boxes.
    :param final_FPS: int
            FPS of the cropped video (some frames can be skipped).
    :return: pd.DataFrame
            DataFrame with information about the cropped images
    """
    # create output folder if needed
    if not os.path.exists(output_path):
        os.makedirs(output_path, exist_ok=True)
    # metadata
    metadata = pd.DataFrame(columns=["path_to_frame", "timestamp", "detected"])
    # load video file
    video = cv2.VideoCapture(path_to_video)
    # get FPS
    FPS = video.get(cv2.CAP_PROP_FPS)
    FPS_in_seconds = 1. / FPS
    # go through all frames
    counter = 0
    while video.isOpened():
        ret, frame = video.read()
        if ret:
            if counter % final_FPS == 0:
                # convert BGR to RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # calculate timestamp
                timestamp = counter * FPS_in_seconds
                # round it to 2 digits to make it readable
                timestamp = round(timestamp, 2)
                # recognize the face
                bbox = get_pose_bbox(frame, detector)
                # if not recognized, note it as NaN
                if bbox is None:
                    output_filename = np.NaN
                    detected = False
                else:
                    # otherwise, extract the face and save it
                    detected = True
                    pose = crop_frame_to_pose(frame, bbox, return_bbox = False)
                    output_filename = os.path.join(output_path, path_to_video.split(os.path.sep)[-1].split(".")[0]
                                                   + f"_%s.png" % (str(timestamp).replace(".", "_")))
                    # save extracted face
                    Image.fromarray(pose).save(output_filename)
                metadata = pd.concat([metadata,
                                      pd.DataFrame.from_records([{
                                          "path_to_frame": output_filename,
                                          "timestamp": timestamp,
                                          "detected": detected}
                                      ])
                                      ], ignore_index=True)
            # increment counter
            counter += 1
        else:
            break
    return metadata



def crop_pose_in_all_videos(paths_to_videos:List[str], output_path:str, detector:object, final_FPS:int,
                            positions_for_output_path:Optional[int]=1)->pd.DataFrame:
    """
        Crops poses in all videos in the provided folder using provided detector and saves them to the output_path.
        :param paths_to_videos: str
                List of paths to the videos, which should be cropped
        :param output_path: str
                path to the folder, where the cropped images should be saved. Every video will have its own subfolder.
        :param detector: object
                the model, which has method detect. It should return bounding boxes.
        :param final_FPS: int
                FPS of the cropped video (some frames can be skipped).
        :param positions_for_output_path: int
                number of subfolder positions from the path to the video, which should be used for the output path.
        :return: pd.DataFrame
                DataFrame with information about the cropped images
        """
    # create output folder if needed
    if not os.path.exists(output_path):
        os.makedirs(output_path, exist_ok=True)
    # metadata
    metadata = pd.DataFrame(columns=["path_to_frame", "timestamp", "detected"])
    # go through all videos
    counter = 0
    for path_to_video in paths_to_videos:
        # create subfolder for the video
        subfolder = path_to_video.split(os.path.sep)[-positions_for_output_path:]
        subfolder[-1] = subfolder[-1].split(".")[0]
        subfolder = os.path.join(output_path, os.path.sep.join(subfolder))
        if not os.path.exists(subfolder):
            os.makedirs(subfolder, exist_ok=True)
        # crop faces
        video_metadata = crop_pose_in_video(path_to_video, subfolder, detector, final_FPS)
        metadata = pd.concat([metadata, video_metadata], ignore_index=True)
        # increment counter
        counter += 1
        print("Video %s has been processed. The number of remained videos is: %i" % (
        path_to_video, len(paths_to_videos) - counter))
    return metadata


def preprocess_NoXi():
    # params for NoXi
    path_to_data = "/media/external_hdd_2/NoXi/Sessions"
    path_to_video_files = glob.glob(os.path.join(path_to_data, "*", "*.mp4"))
    output_path_faces = "/media/external_hdd_2/NoXi/prepared_data/faces"
    output_path_poses = "/media/external_hdd_2/NoXi/prepared_data/poses"
    final_FPS = 5

    # face extraction
    path_to_video_files = path_to_video_files[49:]
    face_detector = load_and_prepare_detector_retinaFace_mobileNet()
    metadata_faces = crop_faces_in_all_videos(path_to_video_files, output_path_faces, face_detector, final_FPS,
                                              positions_for_output_path=2)
    # clear metadata from NaNs
    print("Face detection: dropped %s frames" % (metadata_faces.shape[0] - metadata_faces.dropna().shape[0]))
    metadata_faces = metadata_faces[~metadata_faces["path_to_frame"].isna()]
    metadata_faces.to_csv(os.path.join(output_path_faces, "metadata.csv"), index=False)
    # clear RAM
    del face_detector
    gc.collect()
    torch.cuda.empty_cache()

    # pose extraction
    path_to_video_files = glob.glob(os.path.join(path_to_data, "*", "*.mp4"))
    pose_detector = SimpleHigherHRNet(c=32, nof_joints=17,
                                      checkpoint_path="/work/home/dsu/simpleHigherHRNet/pose_higher_hrnet_w32_512.pth",
                                      return_heatmaps=False, return_bounding_boxes=True, max_batch_size=1,
                                      device="cuda")
    metadata_poses = crop_pose_in_all_videos(path_to_video_files, output_path_poses, pose_detector, final_FPS,
                                             positions_for_output_path=2)
    # clear metadata from NaNs
    print("Pose detection: dropped %s frames" % (metadata_poses.shape[0] - metadata_poses.dropna().shape[0]))
    metadata_poses = metadata_poses[~metadata_poses["path_to_frame"].isna()]
    metadata_poses.to_csv(os.path.join(output_path_poses, "metadata.csv"), index=False)
    # clear RAM
    del pose_detector
    gc.collect()
    torch.cuda.empty_cache()



def preprocess_DAiSEE():
    # params for DAiSEE
    path_to_data = "/media/external_hdd_2/DAiSEE/DAiSEE/DataSet"
    path_to_video_files = glob.glob(os.path.join(path_to_data, "*", "*", "*", "*.avi"))
    output_path_faces = "/media/external_hdd_2/DAiSEE/prepared_data/faces"
    output_path_poses = "/media/external_hdd_2/DAiSEE/prepared_data/poses"
    final_FPS = 5

    # face extraction
    face_detector = load_and_prepare_detector_retinaFace_mobileNet()
    metadata_faces = crop_faces_in_all_videos(path_to_video_files, output_path_faces, face_detector, final_FPS,
                                              positions_for_output_path=4)
    # clear metadata from NaNs
    print("Face detection: dropped %s frames" % (metadata_faces.shape[0] - metadata_faces.dropna().shape[0]))
    metadata_faces = metadata_faces[~metadata_faces["path_to_frame"].isna()]
    metadata_faces.to_csv(os.path.join(output_path_faces, "metadata.csv"), index=False)
    # clear RAM
    del face_detector
    gc.collect()
    torch.cuda.empty_cache()

    # pose extraction
    pose_detector = SimpleHigherHRNet(c=32, nof_joints=17,
                                      checkpoint_path="/work/home/dsu/simpleHigherHRNet/pose_higher_hrnet_w32_512.pth",
                                      return_heatmaps=False, return_bounding_boxes=True, max_batch_size=1,
                                      device="cuda")
    metadata_poses = crop_pose_in_all_videos(path_to_video_files, output_path_poses, pose_detector, final_FPS,
                                             positions_for_output_path=4)
    # clear metadata from NaNs
    print("Pose detection: dropped %s frames" % (metadata_poses.shape[0] - metadata_poses.dropna().shape[0]))
    metadata_poses = metadata_poses[~metadata_poses["path_to_frame"].isna()]
    metadata_poses.to_csv(os.path.join(output_path_poses, "metadata.csv"), index=False)
    # clear RAM
    del pose_detector
    gc.collect()
    torch.cuda.empty_cache()




if __name__ == "__main__":
    preprocess_DAiSEE()
    preprocess_NoXi()






