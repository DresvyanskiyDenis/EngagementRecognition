import PIL
import cv2
import dsntnn
import numpy as np
import torch
import torch.nn as nn
from fastai.layers import TimeDistributed
from matplotlib import pyplot as plt
from skimage import io
from torchinfo import summary
from torchvision.transforms.functional import pil_to_tensor

from pytorch_utils.layers.attention_layers import Transformer_layer
from pytorch_utils.models.CNN_models import Modified_MobileNetV2_pose_estimation, Modified_InceptionResnetV1


class Facial_engagement_recognition_model(nn.Module):

    def __init__(self, facial_model:nn.Module, embeddings_layer_neurons:int,
                 num_classes:int, transformer_num_heads:int, num_timesteps:int):
        super(Facial_engagement_recognition_model, self).__init__()
        self.embeddings_layer_neurons = embeddings_layer_neurons
        self.num_classes = num_classes
        self.transformer_num_heads = transformer_num_heads
        self.num_timesteps = num_timesteps

        # create facial model
        self.facial_model = TimeDistributed(facial_model)

        # create transformer layer for multimodal cross-fusion
        self.transformer_layer_1 = Transformer_layer(input_dim = embeddings_layer_neurons,
                                              num_heads=transformer_num_heads,
                                              dropout=0.2,
                                              positional_encoding=True)

        self.transformer_layer_2 = Transformer_layer(input_dim=embeddings_layer_neurons,
                                                num_heads=transformer_num_heads,
                                                dropout=0.2,
                                                positional_encoding=True)

        # get rid of timesteps
        self.squeeze_layer_1 = nn.Conv1d(num_timesteps, 1, 1)
        self.squeeze_layer_2 = nn.Linear(embeddings_layer_neurons, embeddings_layer_neurons//2)
        self.batch_norm = nn.BatchNorm1d(embeddings_layer_neurons//2)
        self.activation_squeeze_layer = nn.Tanh()
        self.end_dropout = nn.Dropout(0.2)

        # create classifier
        self.classifier = nn.Linear(embeddings_layer_neurons//2, num_classes)


        # create regression
        self.regression = nn.Linear(embeddings_layer_neurons//2, 1)

    def forward(self, x):
        # facial model
        x = self.facial_model(x)
        # fusion
        x = self.transformer_layer_1(key=x, value=x, query=x)
        x = self.transformer_layer_2(key=x, value=x, query=x)
        # squeeze timesteps so that we have [batch_size, num_features]
        x = self.squeeze_layer_1(x)
        x = x.squeeze()
        # one more linear layer
        x = self.squeeze_layer_2(x)
        x = self.batch_norm(x)
        x = self.activation_squeeze_layer(x)
        x = self.end_dropout(x)
        # classifier
        x_classifier = self.classifier(x)
        # regression
        x_regression = self.regression(x)

        return x_classifier, x_regression




class Pose_engagement_recognition_model(nn.Module):

    def __init__(self, pose_model:nn.Module, embeddings_layer_neurons:int,
                 num_classes:int, transformer_num_heads:int, num_timesteps:int):
        super(Pose_engagement_recognition_model, self).__init__()
        self.embeddings_layer_neurons = embeddings_layer_neurons
        self.num_classes = num_classes
        self.transformer_num_heads = transformer_num_heads
        self.num_timesteps = num_timesteps

        # create facial model
        self.pose_model = TimeDistributed(pose_model)

        # create transformer layer for multimodal cross-fusion
        self.transformer_layer_1 = Transformer_layer(input_dim = embeddings_layer_neurons,
                                              num_heads=transformer_num_heads,
                                              dropout=0.2,
                                              positional_encoding=True)

        self.transformer_layer_2 = Transformer_layer(input_dim=embeddings_layer_neurons,
                                                num_heads=transformer_num_heads,
                                                dropout=0.2,
                                                positional_encoding=True)

        # get rid of timesteps
        self.squeeze_layer_1 = nn.Conv1d(num_timesteps, 1, 1)
        self.squeeze_layer_2 = nn.Linear(embeddings_layer_neurons, embeddings_layer_neurons//2)
        self.batch_norm = nn.BatchNorm1d(embeddings_layer_neurons//2)
        self.activation_squeeze_layer = nn.Tanh()
        self.end_dropout = nn.Dropout(0.2)

        # create classifier
        self.classifier = nn.Linear(embeddings_layer_neurons//2, num_classes)


        # create regression
        self.regression = nn.Linear(embeddings_layer_neurons//2, 1)

    def forward(self, x):
        # facial model
        x = self.pose_model(x)
        # fusion
        x = self.transformer_layer_1(key=x, value=x, query=x)
        x = self.transformer_layer_2(key=x, value=x, query=x)
        # squeeze timesteps so that we have [batch_size, num_features]
        x = self.squeeze_layer_1(x)
        x = x.squeeze()
        # one more linear layer
        x = self.squeeze_layer_2(x)
        x = self.batch_norm(x)
        x = self.activation_squeeze_layer(x)
        x = self.end_dropout(x)
        # classifier
        x_classifier = self.classifier(x)
        # regression
        x_regression = self.regression(x)

        return x_classifier, x_regression


def display_pose(img, pose, ids):
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    pose = pose.data.cpu().numpy()
    img = img.cpu().numpy().transpose(1, 2, 0)
    colors = ['g', 'g', 'g', 'g', 'g', 'g', 'm', 'm', 'r', 'r', 'y', 'y', 'y', 'y', 'y', 'y']
    pairs = [[8, 9], [11, 12], [11, 10], [2, 1], [1, 0], [13, 14], [14, 15], [3, 4], [4, 5], [8, 7], [7, 6], [6, 2],
             [6, 3], [8, 12], [8, 13]]
    colors_skeleton = ['r', 'y', 'y', 'g', 'g', 'y', 'y', 'g', 'g', 'm', 'm', 'g', 'g', 'y', 'y']
    img = np.clip(img * std + mean, 0.0, 1.0)
    img_width, img_height, _ = img.shape
    pose = ((pose + 1) * np.array([img_width, img_height]) - 1) / 2  # pose ~ [-1,1]

    plt.subplot(1, 1, ids + 1)
    ax = plt.gca()
    plt.imshow(img)
    for idx in range(len(colors)):
        plt.plot(pose[idx, 0], pose[idx, 1], marker='o', color=colors[idx])
    for idx in range(len(colors_skeleton)):
        plt.plot(pose[pairs[idx], 0], pose[pairs[idx], 1], color=colors_skeleton[idx])

    xmin = np.min(pose[:, 0])
    ymin = np.min(pose[:, 1])
    xmax = np.max(pose[:, 0])
    ymax = np.max(pose[:, 1])

    bndbox = np.array(expand_bbox(xmin, xmax, ymin, ymax, img_width, img_height))
    coords = (bndbox[0], bndbox[1]), bndbox[2] - bndbox[0] + 1, bndbox[3] - bndbox[1] + 1
    ax.add_patch(plt.Rectangle(*coords, fill=False, edgecolor='yellow', linewidth=1))

def expand_bbox(left, right, top, bottom, img_width, img_height):
    width = right-left
    height = bottom-top
    ratio = 0.15
    new_left = np.clip(left-ratio*width,0,img_width)
    new_right = np.clip(right+ratio*width,0,img_width)
    new_top = np.clip(top-ratio*height,0,img_height)
    new_bottom = np.clip(bottom+ratio*height,0,img_height)

    return [int(new_left), int(new_top), int(new_right), int(new_bottom)]


class Rescale(object):


    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image_= sample/256.0
        h, w = image_.shape[:2]
        im_scale = min(float(self.output_size[0]) / float(h), float(self.output_size[1]) / float(w))
        new_h = int(image_.shape[0] * im_scale)
        new_w = int(image_.shape[1] * im_scale)
        image = cv2.resize(image_, (new_w, new_h),
                    interpolation=cv2.INTER_LINEAR)
        left_pad = (self.output_size[1] - new_w) // 2
        right_pad = (self.output_size[1] - new_w) - left_pad
        top_pad = (self.output_size[0] - new_h) // 2
        bottom_pad = (self.output_size[0] - new_h) - top_pad
        mean=np.array([0.485, 0.456, 0.406])
        pad = ((top_pad, bottom_pad), (left_pad, right_pad))
        image = np.stack([np.pad(image[:,:,c], pad, mode='constant', constant_values=mean[c])
                        for c in range(3)], axis=2)

        return image

class ToTensor(object):

    def __call__(self, sample):
        image = sample
        h, w = image.shape[:2]

        mean=np.array([0.485, 0.456, 0.406])
        std=np.array([0.229, 0.224, 0.225])

        image[:,:,:3] = (image[:,:,:3]-mean)/(std)
        image = torch.from_numpy(image.transpose((2, 0, 1))).float()

        return image



if __name__ == "__main__":
    num_timesteps = 10
    image_resolution = (3,160,160)
    x = np.zeros((2,num_timesteps)+image_resolution)
    pose_model = Modified_MobileNetV2_pose_estimation(n_locations=16, pretrained=True)
    """facial_model = Modified_InceptionResnetV1(dense_layer_neurons=256, num_classes=None, pretrained='vggface2')
    unimodal_model_pose = Pose_engagement_recognition_model(pose_model=pose_model,
                                                            embeddings_layer_neurons=256,
                                                            num_classes=5,
                                                            transformer_num_heads=8,
                                                            num_timesteps=num_timesteps)
    unimodal_model_facial = Facial_engagement_recognition_model(facial_model=facial_model,
                                                                embeddings_layer_neurons=256,
                                                                num_classes=5,
                                                                transformer_num_heads=8,
                                                                num_timesteps=num_timesteps)
    summary(unimodal_model_pose, input_size=(2,num_timesteps)+image_resolution)
    print("-------------------------------------------------")
    image_resolution = (3, 160, 160)
    summary(unimodal_model_facial, input_size=(2,num_timesteps)+image_resolution)"""
    device = torch.device( "cpu")
    pose_recognition_model = pose_model.model
    pose_recognition_model = pose_recognition_model.to(device)
    transformation_1 = Rescale((224,224))
    transformation_2 = ToTensor()
    path_to_image = "/work/home/dsu/engagement_recognition_project_server/33.jpg"
    image = io.imread(path_to_image) # load
    image = transformation_1(image) # transform
    image = transformation_2(image)

    plt.figure(figsize=(20, 20))
    with torch.no_grad():
        pose_recognition_model.eval()
        image = image.unsqueeze(0)
        image = image.to(device)
        heatmaps = pose_recognition_model(image)
        # Normalize the heatmaps
        heatmaps = dsntnn.flat_softmax(heatmaps)
        # Calculate the coordinates
        coords = dsntnn.dsnt(heatmaps)
        # Display the image with the pose
        display_pose(image.squeeze(), coords.squeeze(), 0)
        plt.show()






