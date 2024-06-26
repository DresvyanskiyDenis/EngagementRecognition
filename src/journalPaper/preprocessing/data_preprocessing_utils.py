import sys

from tqdm import tqdm

sys.path.extend(["/nfs/home/ddresvya/scripts/datatools/"])
sys.path.extend(["/nfs/home/ddresvya/scripts/engagement_recognition_project_server/"])
sys.path.extend(["/nfs/home/ddresvya/scripts/simple-HRNet-master/"])

import gc
import glob
import os
from typing import List, Optional

import torch
import cv2
import pandas as pd
import numpy as np
from PIL import Image

from decorators.common_decorators import timer
from feature_extraction.pytorch_based.face_recognition_utils import recognize_one_face_bbox, extract_face_according_bbox, \
    load_and_prepare_detector_retinaFace_mobileNet
from feature_extraction.pytorch_based.pose_recognition_utils import get_pose_bbox, crop_frame_to_pose
from utils.warnings_processing import IgnoreWarnings
from SimpleHRNet import SimpleHRNet


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
    # calculate every which frame should be taken
    every_n_frame = int(round(FPS / final_FPS))
    # go through all frames
    counter = 0
    while video.isOpened():
        ret, frame = video.read()
        if ret:
            if counter % every_n_frame == 0:
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
    for path_to_video in tqdm(paths_to_videos):
        # create subfolder for the video
        subfolder = path_to_video.split(os.path.sep)[-positions_for_output_path:]
        subfolder[-1] = subfolder[-1].split(".")[0]
        subfolder = os.path.join(output_path, os.path.sep.join(subfolder))
        if not os.path.exists(subfolder):
            os.makedirs(subfolder, exist_ok=True)
        # crop faces
        try:
            video_metadata = crop_faces_in_video(path_to_video, subfolder, detector, final_FPS)
            metadata = pd.concat([metadata, video_metadata], ignore_index=True)
            print("Video %s has been processed. The number of remained videos is: %i" % (
            path_to_video, len(paths_to_videos) - counter))
            counter += 1
        except Exception as e:
            print("Video %s has been skipped due to the following error: %s" % (path_to_video, e))
            counter += 1

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
    # calculate every which frame should be taken
    every_n_frame = int(round(FPS / final_FPS))
    # go through all frames
    counter = 0
    while video.isOpened():
        ret, frame = video.read()
        if ret:
            if counter % every_n_frame == 0:
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
    for path_to_video in tqdm(paths_to_videos):
        # create subfolder for the video
        subfolder = path_to_video.split(os.path.sep)[-positions_for_output_path:]
        subfolder[-1] = subfolder[-1].split(".")[0]
        subfolder = os.path.join(output_path, os.path.sep.join(subfolder))
        if not os.path.exists(subfolder):
            os.makedirs(subfolder, exist_ok=True)
        # crop faces
        try:
            video_metadata = crop_pose_in_video(path_to_video, subfolder, detector, final_FPS)
            metadata = pd.concat([metadata, video_metadata], ignore_index=True)
            counter += 1
            print("Video %s has been processed. The number of remained videos is: %i" % (
                path_to_video, len(paths_to_videos) - counter))
        except Exception as e:
            print("Video %s has been skipped due to the following error: %s" % (path_to_video, e))
            counter+=1

    return metadata


def preprocess_NoXi(device, extract_faces=True, extract_poses=True):

    # params for NoXi
    path_to_data = "/nfs/scratch/ddresvya/NoXi/NoXi/Sessions"
    path_to_video_files = glob.glob(os.path.join(path_to_data, "*", "*.mp4"))
    output_path_faces = "/nfs/scratch/ddresvya/NoXi/NoXi/prepared_data/faces"
    output_path_poses = "/nfs/scratch/ddresvya/NoXi/NoXi/prepared_data/poses"
    final_FPS = 5

    # face extraction
    if extract_faces:
        face_detector = load_and_prepare_detector_retinaFace_mobileNet(device=device)
        metadata_faces = crop_faces_in_all_videos(path_to_video_files, output_path_faces, face_detector, final_FPS,
                                                  positions_for_output_path=2)
        # save metadata
        print("Face detection: dropped %s frames" % (metadata_faces.shape[0] - metadata_faces.dropna().shape[0]))
        metadata_faces.to_csv(os.path.join(output_path_faces, "metadata.csv"), index=False)
        # clear RAM
        del face_detector
        gc.collect()
        torch.cuda.empty_cache()

    # pose extraction
    if extract_poses:
        pose_detector = SimpleHRNet(c=48, nof_joints=17, multiperson=True,
                               yolo_version = 'v3',
                               yolo_model_def=os.path.join("/nfs/home/ddresvya/scripts/simple-HRNet-master/","models_/detectors/yolo/config/yolov3.cfg"),
                               yolo_class_path=os.path.join("/nfs/home/ddresvya/scripts/simple-HRNet-master/","models_/detectors/yolo/data/coco.names"),
                               yolo_weights_path=os.path.join("/nfs/home/ddresvya/scripts/simple-HRNet-master/","models_/detectors/yolo/weights/yolov3.weights"),
                               checkpoint_path="/nfs/home/ddresvya/scripts/simple-HRNet-master/pose_hrnet_w48_384x288.pth",
                               return_heatmaps=False, return_bounding_boxes=True, max_batch_size=1, device=torch.device(device))

        metadata_poses = crop_pose_in_all_videos(path_to_video_files, output_path_poses, pose_detector, final_FPS,
                                                 positions_for_output_path=2)
        # save metadata
        print("Pose detection: dropped %s frames" % (metadata_poses.shape[0] - metadata_poses.dropna().shape[0]))
        metadata_poses.to_csv(os.path.join(output_path_poses, "metadata.csv"), index=False)
        # clear RAM
        del pose_detector
        gc.collect()
        torch.cuda.empty_cache()



def preprocess_DAiSEE(device, extract_faces=True, extract_poses=True):
    # params for DAiSEE
    path_to_data = "/media/external_hdd_2/DAiSEE/DAiSEE/DataSet"
    path_to_video_files = glob.glob(os.path.join(path_to_data, "*", "*", "*", "*.avi"))
    output_path_faces = "/work/home/dsu/Datasets/DAiSEE/prepared_data/faces"
    output_path_poses = "/work/home/dsu/Datasets/DAiSEE/prepared_data/poses"
    final_FPS = 5

    # pose extraction
    if extract_poses:
        pose_detector = SimpleHRNet(c=48, nof_joints=17, multiperson=True,
                                   yolo_version = 'v3',
                                   yolo_model_def=os.path.join("/nfs/home/ddresvya/scripts/simple-HRNet-master/","models_/detectors/yolo/config/yolov3.cfg"),
                                   yolo_class_path=os.path.join("/nfs/home/ddresvya/scripts/simple-HRNet-master/","models_/detectors/yolo/data/coco.names"),
                                   yolo_weights_path=os.path.join("/nfs/home/ddresvya/scripts/simple-HRNet-master/","models_/detectors/yolo/weights/yolov3.weights"),
                                   checkpoint_path="/nfs/home/ddresvya/scripts/simple-HRNet-master/pose_hrnet_w48_384x288.pth",
                                   return_heatmaps=False, return_bounding_boxes=True, max_batch_size=1, device=torch.device(device))
        metadata_poses = crop_pose_in_all_videos(path_to_video_files, output_path_poses, pose_detector, final_FPS,
                                                 positions_for_output_path=4)
        # save metadata
        print("Pose detection: dropped %s frames" % (metadata_poses.shape[0] - metadata_poses.dropna().shape[0]))
        metadata_poses.to_csv(os.path.join(output_path_poses, "metadata.csv"), index=False)
        # clear RAM
        del pose_detector
        gc.collect()
        torch.cuda.empty_cache()

    # face extraction
    if extract_faces:
        face_detector = load_and_prepare_detector_retinaFace_mobileNet(device=device)
        metadata_faces = crop_faces_in_all_videos(path_to_video_files, output_path_faces, face_detector, final_FPS,
                                                  positions_for_output_path=4)
        # save metadata
        print("Face detection: dropped %s frames" % (metadata_faces.shape[0] - metadata_faces.dropna().shape[0]))
        metadata_faces.to_csv(os.path.join(output_path_faces, "metadata.csv"), index=False)
        # clear RAM
        del face_detector
        gc.collect()
        torch.cuda.empty_cache()

def preprocess_MHHRI(part:str, device, extract_faces=True, extract_poses=True):
    if part not in ["HHI_Ego_Recordings", "HRI_Ego_Recordings"]:
        raise ValueError(f"Part should be either HHI_Ego_Recordings or HRI_Ego_Recordings. Got {part} instead.")
    # params for MHHRI
    path_to_data = "/media/external_hdd_2/MHHRI/mhhri/dataset/HHI_Ego_Recordings/HHI_Ego_Recordings".replace("HHI_Ego_Recordings", part)
    paths_to_video_files = glob.glob(os.path.join(path_to_data, "*.MOV"))
    output_path_faces = "/media/external_hdd_2/MHHRI/mhhri/prepared_data/HHI_Ego_Recordings/faces".replace("HHI_Ego_Recordings", part)
    output_path_poses = "/media/external_hdd_2/MHHRI/mhhri/prepared_data/HHI_Ego_Recordings/poses".replace("HHI_Ego_Recordings", part)
    final_FPS = 5

    # pose extraction
    if extract_poses:
        pose_detector = SimpleHRNet(c=48, nof_joints=17, multiperson=True,
                                   yolo_version = 'v3',
                                   yolo_model_def=os.path.join("/nfs/home/ddresvya/scripts/simple-HRNet-master/","models_/detectors/yolo/config/yolov3.cfg"),
                                   yolo_class_path=os.path.join("/nfs/home/ddresvya/scripts/simple-HRNet-master/","models_/detectors/yolo/data/coco.names"),
                                   yolo_weights_path=os.path.join("/nfs/home/ddresvya/scripts/simple-HRNet-master/","models_/detectors/yolo/weights/yolov3.weights"),
                                   checkpoint_path="/nfs/home/ddresvya/scripts/simple-HRNet-master/pose_hrnet_w48_384x288.pth",
                                   return_heatmaps=False, return_bounding_boxes=True, max_batch_size=1, device=torch.device(device))

        metadata_poses = crop_pose_in_all_videos(paths_to_video_files, output_path_poses, pose_detector, final_FPS,
                                                 positions_for_output_path=1)
        # save metadata
        print("Pose detection: dropped %s frames" % (metadata_poses.shape[0] - metadata_poses.dropna().shape[0]))
        metadata_poses.to_csv(os.path.join(output_path_poses, "metadata.csv"), index=False)
        # clear RAM
        del pose_detector
        gc.collect()
        torch.cuda.empty_cache()

    # face extraction
    if extract_faces:
        face_detector = load_and_prepare_detector_retinaFace_mobileNet(device=device)
        metadata_faces = crop_faces_in_all_videos(paths_to_video_files, output_path_faces, face_detector, final_FPS,
                                                  positions_for_output_path=1)
        # save metadata
        print("Face detection: dropped %s frames" % (metadata_faces.shape[0] - metadata_faces.dropna().shape[0]))
        metadata_faces.to_csv(os.path.join(output_path_faces, "metadata.csv"), index=False)
        # clear RAM
        del face_detector
        gc.collect()
        torch.cuda.empty_cache()







if __name__ == "__main__":
    with IgnoreWarnings("deprecated"):
        # identify device
        device = "cuda:0"
        extract_faces = False
        extract_poses = True
        # extract poses
        #preprocess_DAiSEE(device=device,extract_faces=extract_faces, extract_poses=extract_poses)
        preprocess_NoXi(device=device,extract_faces=extract_faces, extract_poses=extract_poses)
        #preprocess_MHHRI("HHI_Ego_Recordings", device=device,extract_faces=extract_faces, extract_poses=extract_poses)
        #preprocess_MHHRI("HRI_Ego_Recordings", device=device,extract_faces=extract_faces, extract_poses=extract_poses)







