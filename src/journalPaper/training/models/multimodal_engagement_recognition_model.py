from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from fastai.layers import TimeDistributed
from torchinfo import summary

from pytorch_utils.layers.attention_layers import Transformer_layer
from pytorch_utils.models.CNN_models import Modified_MobileNetV2_pose_estimation, Modified_InceptionResnetV1


class Multimodal_engagement_recognition_model(nn.Module):

    def __init__(self, facial_model:nn.Module, pose_model:nn.Module, embeddings_layer_neurons:int,
                 num_classes:int, transformer_num_heads:int, num_timesteps:int):
        super(Multimodal_engagement_recognition_model, self).__init__()
        self.embeddings_layer_neurons = embeddings_layer_neurons
        self.num_classes = num_classes
        self.transformer_num_heads = transformer_num_heads
        self.num_timesteps = num_timesteps

        # create facial model
        self.facial_model = TimeDistributed(facial_model)

        # create pose model
        self.pose_model = TimeDistributed(pose_model)
        # create transformer layer for multimodal cross-fusion
        self.fusion_layer_1 = Transformer_layer(input_dim = embeddings_layer_neurons,
                                              num_heads=transformer_num_heads,
                                              dropout=0.2,
                                            positional_encoding=True)

        self.fusion_layer_2 = Transformer_layer(input_dim=embeddings_layer_neurons,
                                                num_heads=transformer_num_heads,
                                                dropout=0.2,
                                                positional_encoding=True)

        # get rid of timesteps
        self.squeeze_layer_1 = nn.Conv1d(num_timesteps, 1, 1)
        self.squeeze_layer_2 = nn.Linear(embeddings_layer_neurons, embeddings_layer_neurons//2)
        self.batch_norm = nn.BatchNorm1d(embeddings_layer_neurons//2)
        self.activation_squeeze_layer = nn.Tanh()
        self.dropout_sqeeze_layer = nn.Dropout(0.2)

        # create classifier
        self.classifier = nn.Linear(embeddings_layer_neurons//2, num_classes)

        # create regression
        self.regression = nn.Linear(embeddings_layer_neurons//2, 1)

    def forward(self, x_facial, x_pose):
        # facial model
        x_facial = self.facial_model(x_facial)
        # pose model
        x_pose = self.pose_model(x_pose)
        # fusion
        x = self.fusion_layer_1(key=x_pose, value=x_pose, query=x_facial)
        x = self.fusion_layer_2(key=x, value=x, query=x)
        # squeeze timesteps so that we have [batch_size, num_features]
        x = self.squeeze_layer_1(x)
        x = x.squeeze()
        # one more linear layer
        x = self.squeeze_layer_2(x)
        x = self.batch_norm(x)
        x = self.activation_squeeze_layer(x)
        x = self.dropout_sqeeze_layer(x)
        # classifier
        x_classifier = self.classifier(x)
        # regression
        x_regression = self.regression(x)

        return x_classifier, x_regression




if __name__ == "__main__":
    # device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # facial model
    facial_model = Modified_InceptionResnetV1(dense_layer_neurons=256,
                                                num_classes=None, pretrained='vggface2',
                                                device=None)
    facial_model = facial_model.to(device)
    print('-----------------------------facial model architecture---------------------------')
    summary(facial_model, (1, 3, 160, 160))
    # pose model
    pose_model = Modified_MobileNetV2_pose_estimation(n_locations=16, pretrained=True,
                                                        embeddings_layer_neurons=256)
    pose_model = pose_model.to(device)
    print('------------------------------pose model architecture-----------------------------')
    summary(pose_model, (1, 3, 224, 224))

    # test
    model = Multimodal_engagement_recognition_model(facial_model=facial_model,
                                                    pose_model=pose_model,
                                                    embeddings_layer_neurons=256,
                                                    num_classes=5,
                                                    transformer_num_heads=8,
                                                    num_timesteps=5)
    model = model.to(device)
    print('-----------------------------multimodal model architecture---------------------------')
    x_facial = torch.randn(2, 5, 3, 160, 160)
    x_pose = torch.randn(2, 5, 3, 224, 224)
    x_facial, x_pose = x_facial.to(device), x_pose.to(device)
    x_classifier, x_regression = model(x_facial, x_pose)
    print(x_classifier.shape)
    print(x_regression.shape)
    summary(model, [(2, 5, 3, 160, 160), (2, 5, 3, 224, 224)])


