import numpy as np
import pandas as pd
import cv2
import torch
import torchvision
import os

from SimpleHigherHRNet import SimpleHigherHRNet


def cut_frame_to_pose(extractor, frame)->np.ndarray:

    pose = extractor.predict(frame)
    return pose

def extract_frames_with_poses_from_one_video(path_to_videofile:str, extractor, output_path:str, every_n_frame:int=1)->None:

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    video_filename = os.path.basename(path_to_videofile)
    video_filename = os.path.splitext(video_filename)[0]

    if not os.path.exists(os.path.join(output_path, video_filename)):
        os.makedirs(os.path.join(output_path, video_filename))


    frames, _, _ =torchvision.io.read_video(filename=path_to_videofile)

    for num_frame, frame in enumerate(frames):
        if num_frame % every_n_frame == (every_n_frame-1):
            cut_frame = cut_frame_to_pose(extractor, frame)
            cut_frame = cut_frame[0]
            cut_frame = torchvision.transforms.Resize(size=(256, 256))(cut_frame)
            output_filename = os.path.join(output_path, video_filename, str(num_frame) + '.png')
            torchvision.utils.save_image(cut_frame, output_filename)
        if num_frame % 100 == 0:
            print('Number of preprocessed frames: %i, remaining: %i' % (num_frame, len(frames) - num_frame))




if __name__== '__main__':
    model = SimpleHigherHRNet(c=32, nof_joints=17, checkpoint_path=r"C:\Users\Dresvyanskiy\Desktop\Projects\simple-HigherHRNet-master\pose_higher_hrnet_w32_512.pth",
                              return_heatmaps=False, return_bounding_boxes=True, max_batch_size=1)
    extract_frames_with_poses_from_one_video(r'C:\Users\Dresvyanskiy\Desktop\011_2016-03-25_Paris\Novice_video.mp4', model, r'C:\Users\Dresvyanskiy\Desktop\011_2016-03-25_Paris\Novice_video_poses')