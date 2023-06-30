from typing import List

import torch


from pytorch_utils.layers.attention_layers import Transformer_layer


class AttentionFusionModel_2dim(torch.nn.Module):

    def __init__(self, e1_num_features: int, e2_num_features: int, num_classes: int):
        # build cross-attention layers
        super(AttentionFusionModel_2dim, self).__init__()
        self.e1_num_features = e1_num_features
        self.e2_num_features = e2_num_features
        self.num_classes = num_classes
        self._build_cross_attention_modules(e1_num_features, e2_num_features)

        # build last fully connected layers
        self.classifier = torch.nn.Linear(128, num_classes)

    def _build_cross_attention_modules(self, e1_num_features: int, e2_num_features: int):
        self.e1_cross_att_layer = Transformer_layer(input_dim=e1_num_features, num_heads=e1_num_features // 4,
                                                    dropout=0.1, positional_encoding=True)
        self.e2_cross_att_layer = Transformer_layer(input_dim=e2_num_features, num_heads=e2_num_features // 4,
                                                    dropout=0.1, positional_encoding=True)
        self.avg_pool = torch.nn.AdaptiveAvgPool1d(1)
        self.max_pool = torch.nn.AdaptiveMaxPool1d(1)
        self.cross_ett_dense_layer = torch.nn.Linear((e1_num_features + e2_num_features)*2, 128)
        self.cross_att_batch_norm = torch.nn.BatchNorm1d(128)
        self.cross_att_activation = torch.nn.ReLU()

    def forward_cross_attention(self, e1, e2):
        # cross attention
        e1 = self.e1_cross_att_layer(key=e1, value=e1, query=e2) # Output shape (batch_size, sequence_length, e1_num_features)
        e2 = self.e2_cross_att_layer(key=e2, value=e2, query=e1) # Output shape (batch_size, sequence_length, e2_num_features)
        # concat e1 and e2
        x = torch.cat((e1, e2), dim=-1) # Output shape (batch_size, sequence_length, num_features = e1_num_features + e2_num_features)
        # permute it to (batch_size, num_features, sequence_length) for calculating 1D avg pooling and 1D max pooling
        x = x.permute(0, 2, 1) # Output size (batch_size, num_features, sequence_length)
        # calculate 1D avg pooling and 1D max pooling
        avg_pool = self.avg_pool(x)
        max_pool = self.max_pool(x)
        # get rid of the dimension with size 1 (last dimension)
        avg_pool = avg_pool.squeeze(dim=-1) # Output shape (batch_size, num_features)
        max_pool = max_pool.squeeze(dim=-1) # Output shape (batch_size, num_features)
        # concat avg_pool and max_pool
        x = torch.cat((avg_pool, max_pool), dim=-1) # Output shape (batch_size, num_features*2)
        # dense layer
        x = self.cross_ett_dense_layer(x)
        x = self.cross_att_batch_norm(x)
        x = self.cross_att_activation(x)
        return x

    def forward(self, feature_set_1, feature_set_2):
        # features is a list of tensors
        # every element of the list has a shape of (batch_size, sequence_length, num_features)

        e1, e2 = feature_set_1, feature_set_2
        # cross attention
        x = self.forward_cross_attention(e1, e2)
        # classifier
        x = self.classifier(x)
        return x

