from typing import List

import torch
from torchinfo import summary

from pytorch_utils.layers.attention_layers import Transformer_layer, PositionalEncoding




class AttentionFusionModel_2dim_continuous(torch.nn.Module):

    def __init__(self, e1_num_features: int, e2_num_features: int):
        # build cross-attention layers
        super(AttentionFusionModel_2dim_continuous, self).__init__()
        self.e1_num_features = e1_num_features
        self.e2_num_features = e2_num_features
        self._build_cross_attention_modules(e1_num_features, e2_num_features)

        # build last fully connected layers
        self.regressor = torch.nn.Sequential(
            torch.nn.Linear(256, 1),
            torch.nn.Tanh()
        )


    def _build_cross_attention_modules(self, e1_num_features: int, e2_num_features: int):
        self.e1_cross_att_layer_1 = Transformer_layer(input_dim=e1_num_features, num_heads=e1_num_features // 4,
                                                      dropout=0.1, positional_encoding=True)
        self.e2_cross_att_layer_1 = Transformer_layer(input_dim=e2_num_features, num_heads=e2_num_features // 4,
                                                      dropout=0.1, positional_encoding=True)
        self.e1_cross_att_layer_2 = Transformer_layer(input_dim=e1_num_features, num_heads=e1_num_features // 4,
                                                      dropout=0.1, positional_encoding=True)
        self.e2_cross_att_layer_2 = Transformer_layer(input_dim=e2_num_features, num_heads=e2_num_features // 4,
                                                      dropout=0.1, positional_encoding=True)

        self.cross_ett_dense_layer = torch.nn.Linear(e1_num_features + e2_num_features, 256)
        self.cross_att_batch_norm = torch.nn.BatchNorm1d(256)
        self.cross_att_activation = torch.nn.GELU()

    def forward_cross_attention(self, e1, e2):
        # cross attention 1
        e1 = self.e1_cross_att_layer_1(key=e1, value=e1,
                                       query=e2)  # Output shape (batch_size, sequence_length, e1_num_features)
        e2 = self.e2_cross_att_layer_1(key=e2, value=e2,
                                       query=e1)  # Output shape (batch_size, sequence_length, e2_num_features)
        # cross attention 2
        e1 = self.e1_cross_att_layer_2(key=e1, value=e1,
                                       query=e2)  # Output shape (batch_size, sequence_length, e1_num_features)
        e2 = self.e2_cross_att_layer_2(key=e2, value=e2,
                                       query=e1)  # Output shape (batch_size, sequence_length, e2_num_features)
        # concat e1 and e2
        x = torch.cat((e1, e2),
                      dim=-1)  # Output shape (batch_size, sequence_length, num_features = e1_num_features + e2_num_features)
        # dense layer
        x = self.cross_ett_dense_layer(x) # shape: (batch_size, sequence_length, 256)
        # permute to (batch_size, 256, sequence_length) for batch norm
        x = x.permute(0, 2, 1)
        x = self.cross_att_batch_norm(x)
        x = x.permute(0, 2, 1)
        # activation
        x = self.cross_att_activation(x)
        return x

    def forward(self, feature_set_1, feature_set_2):
        # features is a list of tensors
        # every element of the list has a shape of (batch_size, sequence_length, num_features)

        e1, e2 = feature_set_1, feature_set_2
        # cross attention
        x = self.forward_cross_attention(e1, e2)
        # classifier
        x = self.regressor(x)
        return x






class AttentionFusionModel_3dim_v1_continuous(torch.nn.Module):

    def __init__(self, e1_num_features: int, e2_num_features: int, e3_num_features: int):
        # check if all the embeddings have the same sequence length
        super().__init__()
        assert e1_num_features == e2_num_features == e3_num_features

        self.e1_num_features = e1_num_features
        self.e2_num_features = e2_num_features
        self.e3_num_features = e3_num_features
        # create cross-attention blocks
        self.__build_cross_attention_blocks()
        # build last fully connected layers
        self.regressor = torch.nn.Sequential(
            torch.nn.Linear(256, 1),
            torch.nn.Tanh()
        )


    def __build_cross_attention_blocks(self):
        # assuming that the embeddings are ordered as 'facial', 'pose', 'affective', we need to build several cross-attention blocks
        # first block will be the block, where the facial modality is the main one, while pose or affective are the secondary ones
        # f - facial, p - pose, a - affective.  THen f_p - facial is the main modality, pose is the secondary one,
        # f_a - facial is the main modality, affective is the secondary one, and so on
        # main - facial
        self.block_f_p = Transformer_layer(input_dim=self.e1_num_features, num_heads=8, dropout=0.1,
                                           positional_encoding=True)
        self.block_f_a = Transformer_layer(input_dim=self.e1_num_features, num_heads=8, dropout=0.1,
                                           positional_encoding=True)
        # main - pose
        self.block_p_f = Transformer_layer(input_dim=self.e2_num_features, num_heads=8, dropout=0.1,
                                           positional_encoding=True)
        self.block_p_a = Transformer_layer(input_dim=self.e2_num_features, num_heads=8, dropout=0.1,
                                           positional_encoding=True)
        # main - affective
        self.block_a_f = Transformer_layer(input_dim=self.e3_num_features, num_heads=8, dropout=0.1,
                                           positional_encoding=True)
        self.block_a_p = Transformer_layer(input_dim=self.e3_num_features, num_heads=8, dropout=0.1,
                                           positional_encoding=True)

        # now we need to do a cross attention with the remaining modality. For example, for the f_p block it will be
        # cross attention with the affective modelity. As a result, we get f_p_a block
        # for f_a block it will be cross attention with the pose modality. As a result, we get f_a_p block, and so on.
        # main - facial
        self.block_f_p_a = Transformer_layer(input_dim=self.e1_num_features, num_heads=8, dropout=0.1,
                                             positional_encoding=True)
        self.block_f_a_p = Transformer_layer(input_dim=self.e1_num_features, num_heads=8, dropout=0.1,
                                             positional_encoding=True)
        # main - pose
        self.block_p_f_a = Transformer_layer(input_dim=self.e2_num_features, num_heads=8, dropout=0.1,
                                             positional_encoding=True)
        self.block_p_a_f = Transformer_layer(input_dim=self.e2_num_features, num_heads=8, dropout=0.1,
                                             positional_encoding=True)
        # main - affective
        self.block_a_f_p = Transformer_layer(input_dim=self.e3_num_features, num_heads=8, dropout=0.1,
                                             positional_encoding=True)
        self.block_a_p_f = Transformer_layer(input_dim=self.e3_num_features, num_heads=8, dropout=0.1,
                                             positional_encoding=True)

        # at the end of the cross attention we want to calculate 1D avg pooling and 1D max pooling to 'aggregate'
        # the temporal information
        # at the time of the pooling, embeddings from all different cross-attention blocks will be concatenated
        self.cross_att_dense_layer = torch.nn.Linear(
            (self.e1_num_features + self.e2_num_features + self.e3_num_features)*2, 256)
        self.cross_att_batch_norm = torch.nn.BatchNorm1d(256)
        self.cross_att_activation = torch.nn.GELU()

    def __forward_cross_attention_first_stage(self, f, p, a):
        # f - facial, p - pose, a - affective
        # main - facial
        f_p = self.block_f_p(key = p, value = p, query = f)
        f_a = self.block_f_a(key = a, value = a, query = f)
        # main - pose
        p_f = self.block_p_f(key = f, value = f, query = p)
        p_a = self.block_p_a(key = a, value = a, query = p)
        # main - affective
        a_f = self.block_a_f(key = f, value = f, query = a)
        a_p = self.block_a_p(key = p, value = p, query = a)
        return f_p, f_a, p_f, p_a, a_f, a_p

    def __forward_cross_attention_second_stage(self, f_p, f_a, p_f, p_a, a_f, a_p, f, p, a):
        # f - facial, p - pose, a - affective
        # main - facial
        f_p_a = self.block_f_p_a(key = a, value = a, query = f_p)
        f_a_p = self.block_f_a_p(key = p, value = p, query = f_a)
        # main - pose
        p_f_a = self.block_p_f_a(key = a, value = a, query = p_f)
        p_a_f = self.block_p_a_f(key = f, value = f, query = p_a)
        # main - affective
        a_f_p = self.block_a_f_p(key = p, value = p, query = a_f)
        a_p_f = self.block_a_p_f(key = f, value = f, query = a_p)
        return f_p_a, f_a_p, p_f_a, p_a_f, a_f_p, a_p_f

    def forward(self, f, p, a):
        # cross attention first stage
        f_p, f_a, p_f, p_a, a_f, a_p = self.__forward_cross_attention_first_stage(f, p, a)
        # cross attention second stage
        f_p_a, f_a_p, p_f_a, p_a_f, a_f_p, a_p_f = self.__forward_cross_attention_second_stage(f_p, f_a,
                                                                                               p_f, p_a,
                                                                                               a_f, a_p,
                                                                                               f, p, a)
        # now we need to concatenate all the embeddings from the second stage
        # concat
        output = torch.cat((f_p_a, f_a_p, p_f_a, p_a_f, a_f_p, a_p_f),
                           dim=-1)  # Output shape (batch_size, sequence_length, num_features = e1_num_features + e2_num_features + e3_num_features)
        # dense layer + batch norm + activation
        output = self.cross_att_dense_layer(output)  # Output size (batch_size, sequence_length, 256)
        # permute to (batch_size, 256, sequence_length) for batch norm
        output = output.permute(0, 2, 1)
        output = self.cross_att_batch_norm(output)  # Output size (batch_size, 256)
        output = output.permute(0, 2, 1)
        # activation
        output = self.cross_att_activation(output)  # Output size (batch_size, 256)
        # last classification layer
        output = self.regressor(output)  # Output size (batch_size, num_classes)
        return output


def construct_model(model_type:str)->torch.nn.Module:

    if model_type == "2dim":
        model = AttentionFusionModel_2dim_continuous(e1_num_features=256, e2_num_features=256)
    elif model_type == "3dim_v1":
        model = AttentionFusionModel_3dim_v1_continuous(e1_num_features=256, e2_num_features=256, e3_num_features=256)
    else:
        raise ValueError("Unknown model type")

    return model




if __name__ == "__main__":
    model = AttentionFusionModel_2dim_continuous(e1_num_features=256, e2_num_features=256)
    print(summary(model, input_size=[(2,20, 256), (2,20, 256)]))

    model = AttentionFusionModel_3dim_v1_continuous(e1_num_features=256, e2_num_features=256, e3_num_features=256)
    print(summary(model, input_size=[(2,20, 256), (2,20, 256), (2,20, 256)]))