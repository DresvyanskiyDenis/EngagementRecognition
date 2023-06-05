import torch.nn as nn
from fastai.layers import TimeDistributed

from pytorch_utils.layers.attention_layers import Transformer_layer


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

        return x_classifier




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

    def forward(self, x):
        # HRNet model
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

        return x_classifier







