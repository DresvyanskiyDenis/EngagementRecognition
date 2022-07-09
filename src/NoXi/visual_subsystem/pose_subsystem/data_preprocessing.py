import glob

import numpy as np
import pandas as pd
import cv2
import torch
import torchvision
import os

from SimpleHigherHRNet import SimpleHigherHRNet


def cut_frame_to_pose(extractor, frame)->np.ndarray:

    height, width, _ = frame.shape
    prediction = extractor.predict(frame)
    if prediction is None:
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
    cut_frame = frame[bbox[1]:bbox[3], bbox[0]:bbox[2]]
    return cut_frame

def extract_frames_with_poses_from_one_video(path_to_videofile:str, extractor, output_path:str, every_n_frame:int=5)->None:

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    video_filename = os.path.basename(path_to_videofile)
    video_filename = os.path.splitext(video_filename)[0]

    if not os.path.exists(os.path.join(output_path, video_filename)):
        os.makedirs(os.path.join(output_path, video_filename))


    #frames, _, _ =torchvision.io.read_video(filename=path_to_videofile)
    reader = cv2.VideoCapture(path_to_videofile)
    num_frame=0

    #for num_frame, frame in enumerate(reader):
    while (reader.isOpened()):
        ret, frame = reader.read()
        # transform to RGB format and make a tensor from it

        if num_frame % every_n_frame == (every_n_frame-1):
            cut_frame = cut_frame_to_pose(extractor, frame)
            if cut_frame is None: continue
            #cut_frame = torchvision.transforms.Resize(size=(256, 256))(cut_frame)
            output_filename = os.path.join(output_path, video_filename, str(num_frame) + '.png')
            #torchvision.utils.save_image(cut_frame, output_filename)
            cv2.imwrite(output_filename, cut_frame)
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
        full_output_path = os.path.join(output_path,
                                        video_file.split(os.path.sep)[-2],
                                        os.path.basename(video_file).split(".")[0])
        extract_frames_with_poses_from_one_video(video_file, extractor, full_output_path, every_n_frame)




if __name__== '__main__':
    model = SimpleHigherHRNet(c=32, nof_joints=17,
                              checkpoint_path=r"C:\Users\Professional\PycharmProjects\simple-HigherHRNet-master\pose_higher_hrnet_w32_512.pth",
                              return_heatmaps=False, return_bounding_boxes=True, max_batch_size=1,
                              device="cuda")
    extract_frames_with_poses_from_one_video(r'E:\db\080_2016-05-24_Augsburg\Novice_video.mp4',
                                             model,
                                             r'E:\db\080_2016-05-24_Augsburg\Novice_video_poses')