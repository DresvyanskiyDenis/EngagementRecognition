from typing import List

import torch
from torchinfo import summary

from pytorch_utils.layers.attention_layers import Transformer_layer, PositionalEncoding


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
        self.e1_cross_att_layer_1 = Transformer_layer(input_dim=e1_num_features, num_heads=e1_num_features // 4,
                                                      dropout=0.1, positional_encoding=True)
        self.e2_cross_att_layer_1 = Transformer_layer(input_dim=e2_num_features, num_heads=e2_num_features // 4,
                                                      dropout=0.1, positional_encoding=True)
        self.e1_cross_att_layer_2 = Transformer_layer(input_dim=e1_num_features, num_heads=e1_num_features // 4,
                                                      dropout=0.1, positional_encoding=True)
        self.e2_cross_att_layer_2 = Transformer_layer(input_dim=e2_num_features, num_heads=e2_num_features // 4,
                                                      dropout=0.1, positional_encoding=True)
        self.avg_pool = torch.nn.AdaptiveAvgPool1d(1)
        self.max_pool = torch.nn.AdaptiveMaxPool1d(1)
        self.cross_ett_dense_layer = torch.nn.Linear((e1_num_features + e2_num_features) * 2, 128)
        self.cross_att_batch_norm = torch.nn.BatchNorm1d(128)
        self.cross_att_activation = torch.nn.ReLU()

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
        # permute it to (batch_size, num_features, sequence_length) for calculating 1D avg pooling and 1D max pooling
        x = x.permute(0, 2, 1)  # Output size (batch_size, num_features, sequence_length)
        # calculate 1D avg pooling and 1D max pooling
        avg_pool = self.avg_pool(x)
        max_pool = self.max_pool(x)
        # get rid of the dimension with size 1 (last dimension)
        avg_pool = avg_pool.squeeze(dim=-1)  # Output shape (batch_size, num_features)
        max_pool = max_pool.squeeze(dim=-1)  # Output shape (batch_size, num_features)
        # concat avg_pool and max_pool
        x = torch.cat((avg_pool, max_pool), dim=-1)  # Output shape (batch_size, num_features*2)
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


class AttentionFusionModel_3dim_v1(torch.nn.Module):

    def __init__(self, e1_num_features: int, e2_num_features: int, e3_num_features: int, num_classes: int):
        # check if all the embeddings have the same sequence length
        super().__init__()
        assert e1_num_features == e2_num_features == e3_num_features

        self.e1_num_features = e1_num_features
        self.e2_num_features = e2_num_features
        self.e3_num_features = e3_num_features
        self.num_classes = num_classes
        # create cross-attention blocks
        self.__build_cross_attention_blocks()
        # build last fully connected layers
        self.classifier = torch.nn.Linear(256, num_classes)

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
        self.avg_pool = torch.nn.AdaptiveAvgPool1d(1)
        self.max_pool = torch.nn.AdaptiveMaxPool1d(1)
        self.cross_att_dense_layer = torch.nn.Linear(
            (self.e1_num_features + self.e2_num_features + self.e3_num_features) * 2*2, 256)
        self.cross_att_batch_norm = torch.nn.BatchNorm1d(256)
        self.cross_att_activation = torch.nn.ReLU()

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
        # permute it to (batch_size, num_features, sequence_length) for calculating 1D avg pooling and 1D max pooling
        output = output.permute(0, 2, 1)  # Output size (batch_size, num_features, sequence_length)
        # 1D avg pooling and 1D max pooling
        avg_pool = self.avg_pool(output)  # Output size (batch_size, num_features, 1)
        max_pool = self.max_pool(output)  # Output size (batch_size, num_features, 1)
        # get rid of the dimension with size 1 (last dimension)
        avg_pool = avg_pool.squeeze(-1)  # Output size (batch_size, num_features)
        max_pool = max_pool.squeeze(-1)  # Output size (batch_size, num_features)
        # concat avg_pool and max_pool
        output = torch.cat((avg_pool, max_pool), dim=1)  # Output size (batch_size, num_features * 2)
        # dense layer + batch norm + activation
        output = self.cross_att_dense_layer(output)  # Output size (batch_size, 256)
        output = self.cross_att_batch_norm(output)  # Output size (batch_size, 256)
        output = self.cross_att_activation(output)  # Output size (batch_size, 256)
        # last classification layer
        output = self.classifier(output)  # Output size (batch_size, num_classes)
        return output

class AttentionFusionModel_3dim_v2(torch.nn.Module):

    def __init__(self, e1_num_features: int, e2_num_features: int, e3_num_features: int, num_classes: int):
        # check if all the embeddings have the same sequence length
        super().__init__()
        assert e1_num_features == e2_num_features == e3_num_features

        self.e1_num_features = e1_num_features
        self.e2_num_features = e2_num_features
        self.e3_num_features = e3_num_features
        self.num_classes = num_classes

        # create cross-attention blocks
        self.__build_cross_attention_blocks()
        # build last fully connected layers
        self.classifier = torch.nn.Linear(256, num_classes)


    def __build_cross_attention_blocks(self):
        # build cross-attention blocks. In the first stage we have facial and pose features and affective features
        # as the supporting features.
        self.block_f_a = Transformer_layer(input_dim=self.e1_num_features, num_heads=8, dropout=0.1,
                                           positional_encoding=True)

        self.block_p_a = Transformer_layer(input_dim=self.e2_num_features, num_heads=8, dropout=0.1,
                                           positional_encoding=True)
        # in te second stage, we unite facial-affective and pose-affective features. The supportive features in this case
        # are facial-affective or pose-affective features depending on the main features.
        # For example, the main is facial-affective, then supportive is pose-affective. We call this f_a_p_a

        self.block_f_a_p_a = Transformer_layer(input_dim=self.e1_num_features, num_heads=8, dropout=0.1,
                                               positional_encoding=True)
        self.block_p_a_f_a = Transformer_layer(input_dim=self.e2_num_features, num_heads=8, dropout=0.1,
                                                  positional_encoding=True)

        # at the end of the cross attention we want to calculate 1D avg pooling and 1D max pooling to 'aggregate'
        # the temporal information
        # at the time of the pooling, embeddings from all different cross-attention blocks will be concatenated
        self.avg_pool = torch.nn.AdaptiveAvgPool1d(1)
        self.max_pool = torch.nn.AdaptiveMaxPool1d(1)
        self.cross_att_dense_layer = torch.nn.Linear((self.e1_num_features + self.e2_num_features)*2, 256)
        self.cross_att_batch_norm = torch.nn.BatchNorm1d(256)
        self.cross_att_activation = torch.nn.Tanh()


    def __forward_cross_attention(self, f, p, a):
        # forward pass through the first stage of cross-attention
        # query - main feature, key - supportive feature, value - supportive feature
        f_a = self.block_f_a(query=f, key=a, value=a)
        p_a = self.block_p_a(query=p, key=a, value=a)
        # forward pass through the second stage of cross-attention
        f_a_p_a = self.block_f_a_p_a(query=f_a, key=p_a, value=p_a)
        p_a_f_a = self.block_p_a_f_a(query=p_a, key=f_a, value=f_a)
        return f_a_p_a, p_a_f_a



    def forward(self, f, p, a):
        # cross-attention
        f_a_p_a, p_a_f_a = self.__forward_cross_attention(f, p, a)
        # concat
        output = torch.cat((f_a_p_a, p_a_f_a), dim=-1)  # Output shape (batch_size, sequence_length, num_features = e1_num_features + e2_num_features)
        # permute it to (batch_size, num_features, sequence_length) for calculating 1D avg pooling and 1D max pooling
        output = output.permute(0, 2, 1)  # Output size (batch_size, num_features, sequence_length)
        # 1D avg pooling and 1D max pooling
        avg_pool = self.avg_pool(output)  # Output size (batch_size, num_features, 1)
        max_pool = self.max_pool(output)  # Output size (batch_size, num_features, 1)
        # get rid of the dimension with size 1 (last dimension)
        avg_pool = avg_pool.squeeze(-1)  # Output size (batch_size, num_features)
        max_pool = max_pool.squeeze(-1)  # Output size (batch_size, num_features)
        # concat avg_pool and max_pool
        output = torch.cat((avg_pool, max_pool), dim=-1)  # Output size (batch_size, num_features * 2)
        # dense layer + batch norm + activation
        output = self.cross_att_dense_layer(output)  # Output size (batch_size, 256)
        output = self.cross_att_batch_norm(output)  # Output size (batch_size, 256)
        output = self.cross_att_activation(output)  # Output size (batch_size, 256)
        # last classification layer
        output = self.classifier(output)  # Output size (batch_size, num_classes)
        return output



class AttentionFusionModel_3dim_v3(torch.nn.Module):


    def __init__(self, e1_num_features:int, e2_num_features:int, e3_num_features:int, num_classes:int):
        super().__init__()
        # check if all the embeddings have the same sequence length
        assert e1_num_features == e2_num_features == e3_num_features
        self.e1_num_features = e1_num_features
        self.e2_num_features = e2_num_features
        self.e3_num_features = e3_num_features
        self.num_classes = num_classes

        # create self-attention blocks
        self.__build_self_attention_blocks()

        # create last classification layer
        self.classifier = torch.nn.Linear(256, num_classes)



    def __build_self_attention_blocks(self):
        self.block_1 = Transformer_layer(input_dim=self.e1_num_features+self.e2_num_features+self.e3_num_features,
                                         num_heads=32, dropout=0.1, positional_encoding=True)
        self.block_2 = Transformer_layer(input_dim=self.e1_num_features+self.e2_num_features+self.e3_num_features,
                                            num_heads=32, dropout=0.1, positional_encoding=True)
        # avg and max pooling to get rid of the sequence length dimension
        self.avg_pool = torch.nn.AdaptiveAvgPool1d(1)
        self.max_pool = torch.nn.AdaptiveMaxPool1d(1)

        # at the end - dropout, linear, batchnorm, activation
        self.self_att_dropout = torch.nn.Dropout(0.1)
        self.self_att_dense_layer = torch.nn.Linear((self.e1_num_features+self.e2_num_features+self.e3_num_features)*2, 256)
        self.self_att_batch_norm = torch.nn.BatchNorm1d(256)
        self.self_att_activation = torch.nn.Tanh()

    def __forward_self_attention(self, f_p_a):
        # f_p_a is already concatenated vector of f, p, a
        output = self.block_1(key=f_p_a, query=f_p_a, value=f_p_a)
        output = self.block_2(key=output, query=output, value=output) # Output shape (batch_size, sequence_length, num_features = e1_num_features + e2_num_features + e3_num_features)
        # permute it to (batch_size, num_features, sequence_length) for calculating 1D avg pooling and 1D max pooling
        output = output.permute(0, 2, 1)  # Output size (batch_size, num_features, sequence_length)
        # 1D avg pooling and 1D max pooling
        avg_pool = self.avg_pool(output)  # Output size (batch_size, num_features, 1)
        max_pool = self.max_pool(output)  # Output size (batch_size, num_features, 1)
        # get rid of the dimension with size 1 (last dimension)
        avg_pool = avg_pool.squeeze(-1)  # Output size (batch_size, num_features)
        max_pool = max_pool.squeeze(-1)  # Output size (batch_size, num_features)
        # concat avg_pool and max_pool
        output = torch.cat((avg_pool, max_pool), dim=-1)  # Output size (batch_size, num_features * 2)
        # at the end - dropout, linear, batchnorm, activation
        output = self.self_att_dropout(output)
        output = self.self_att_dense_layer(output)
        output = self.self_att_batch_norm(output)
        output = self.self_att_activation(output)
        return output



    def forward(self, f, p, a):
        # first, concatenate f, p, a
        f_p_a = torch.cat((f, p, a), dim=-1)  # Output shape (batch_size, sequence_length, num_features = e1_num_features + e2_num_features + e3_num_features)
        # self-attention
        output = self.__forward_self_attention(f_p_a) # Output shape (batch_size, 256)
        # last classification layer
        output = self.classifier(output)  # Output size (batch_size, num_classes)
        return output


class AttentionFusionModel_3dim_v4(torch.nn.Module):

    def __init__(self, e1_num_features: int, e2_num_features: int, e3_num_features: int, num_classes: int):
        # check if all the embeddings have the same sequence length
        super().__init__()
        assert e1_num_features == e2_num_features == e3_num_features

        self.e1_num_features = e1_num_features
        self.e2_num_features = e2_num_features
        self.e3_num_features = e3_num_features
        self.num_classes = num_classes

        # create cross-attention blocks
        self.__build_cross_attention_blocks()
        # build last fully connected layers
        self.classifier = torch.nn.Linear(256, num_classes)


    def __build_cross_attention_blocks(self):
        # build cross-attention blocks. In the first stage we have facial, pose, and affective features, which do self-attention
        # on its own (to model temporal dependencies within each modality)
        self.block_f = Transformer_layer(input_dim=self.e1_num_features, num_heads=8, dropout=0.1,
                                           positional_encoding=True)

        self.block_p = Transformer_layer(input_dim=self.e2_num_features, num_heads=8, dropout=0.1,
                                           positional_encoding=True)
        self.block_a = Transformer_layer(input_dim=self.e3_num_features, num_heads=8, dropout=0.1,
                                             positional_encoding=True)
        # in the second stage, we unite separately facial and affective and pose and affective features.
        # THe supportive features in both cases are affective features.

        self.block_f_a= Transformer_layer(input_dim=self.e1_num_features, num_heads=8, dropout=0.1,
                                               positional_encoding=True)
        self.block_p_a = Transformer_layer(input_dim=self.e2_num_features, num_heads=8, dropout=0.1,
                                                  positional_encoding=True)

        # finally, we concatenate f_a and p_a and do self-attention. However, we need to turn off the positional encoding
        # as we need to do the positional encoding for them separately (for both f_a and p_a)

        self.block_f_p_a = Transformer_layer(input_dim=self.e1_num_features+self.e2_num_features, num_heads=8, dropout=0.1,
                                                    positional_encoding=False)

        self.f_a_positional_encoding = PositionalEncoding(self.e1_num_features)
        self.p_a_positional_encoding = PositionalEncoding(self.e2_num_features)


        # at the end of the cross attention we want to calculate 1D avg pooling and 1D max pooling to 'aggregate'
        # the temporal information
        # at the time of the pooling, embeddings from all different cross-attention blocks will be concatenated
        self.avg_pool = torch.nn.AdaptiveAvgPool1d(1)
        self.max_pool = torch.nn.AdaptiveMaxPool1d(1)
        self.cross_att_dense_layer = torch.nn.Linear((self.e1_num_features + self.e2_num_features)*2, 256)
        self.cross_att_batch_norm = torch.nn.BatchNorm1d(256)
        self.cross_att_activation = torch.nn.Tanh()


    def __forward_cross_attention(self, f, p, a):
        # forward pass through the first stage of self-attention
        # query, keys, values - main feature
        f = self.block_f(query=f, key=f, value=f)
        p = self.block_p(query=p, key=p, value=p)
        a = self.block_a(query=a, key=a, value=a)
        # forward pass through the second stage of cross-attention
        # f or p - main feature, a - supportive feature
        f_a = self.block_f_a(query=f, key=a, value=a)
        p_a = self.block_p_a(query=p, key=a, value=a)
        # forward pass through the third stage of cross-attention
        # first, apply positional encoding to f_a and p_a
        f_a = self.f_a_positional_encoding(f_a)
        p_a = self.p_a_positional_encoding(p_a)
        # concatenate f_a and p_a
        f_p_a = torch.cat((f_a, p_a), dim=-1)
        # apply self-attention
        f_p_a = self.block_f_p_a(query=f_p_a, key=f_p_a, value=f_p_a)

        return f_p_a



    def forward(self, f, p, a):
        # cross-attention
        f_p_a = self.__forward_cross_attention(f, p, a)
        # permute it to (batch_size, num_features, sequence_length) for calculating 1D avg pooling and 1D max pooling
        f_p_a = f_p_a.permute(0, 2, 1)  # Output size (batch_size, num_features, sequence_length)
        # 1D avg pooling and 1D max pooling
        avg_pool = self.avg_pool(f_p_a)  # Output size (batch_size, num_features, 1)
        max_pool = self.max_pool(f_p_a)  # Output size (batch_size, num_features, 1)
        # get rid of the dimension with size 1 (last dimension)
        avg_pool = avg_pool.squeeze(-1)  # Output size (batch_size, num_features)
        max_pool = max_pool.squeeze(-1)  # Output size (batch_size, num_features)
        # concat avg_pool and max_pool
        output = torch.cat((avg_pool, max_pool), dim=-1)  # Output size (batch_size, num_features * 2)
        # dense layer + batch norm + activation
        output = self.cross_att_dense_layer(output)  # Output size (batch_size, 256)
        output = self.cross_att_batch_norm(output)  # Output size (batch_size, 256)
        output = self.cross_att_activation(output)  # Output size (batch_size, 256)
        # last classification layer
        output = self.classifier(output)  # Output size (batch_size, num_classes)
        return output



class AttentionFusionModel_3dim_v5(torch.nn.Module):

    def __init__(self, e1_num_features: int, e2_num_features: int, e3_num_features: int, num_classes: int):
        # check if all the embeddings have the same sequence length
        super().__init__()
        assert e1_num_features == e2_num_features == e3_num_features

        self.e1_num_features = e1_num_features
        self.e2_num_features = e2_num_features
        self.e3_num_features = e3_num_features
        self.num_classes = num_classes

        # create cross-attention blocks
        self.__build_cross_attention_blocks()
        # build last fully connected layers
        self.classifier = torch.nn.Linear(256, num_classes)


    def __build_cross_attention_blocks(self):
        # build cross-attention blocks. In the first stage we have facial, pose, and affective features, which do self-attention
        # on its own (to model temporal dependencies within each modality)
        self.block_f = Transformer_layer(input_dim=self.e1_num_features, num_heads=8, dropout=0.1,
                                           positional_encoding=True)

        self.block_p = Transformer_layer(input_dim=self.e2_num_features, num_heads=8, dropout=0.1,
                                           positional_encoding=True)
        self.block_a = Transformer_layer(input_dim=self.e3_num_features, num_heads=8, dropout=0.1,
                                             positional_encoding=True)
        # in the second stage, we unite separately facial and affective and pose and affective features.
        # THe supportive features in both cases are affective features.

        self.block_f_a= Transformer_layer(input_dim=self.e1_num_features, num_heads=8, dropout=0.1,
                                               positional_encoding=True)
        self.block_p_a = Transformer_layer(input_dim=self.e2_num_features, num_heads=8, dropout=0.1,
                                                  positional_encoding=True)

        # finally, we do cross attention between f_a and p_a. We call the result f_a_p_a and p_a_f_a
        self.block_f_a_p_a = Transformer_layer(input_dim=self.e1_num_features, num_heads=8, dropout=0.1,
                                                    positional_encoding=True)
        self.block_p_a_f_a = Transformer_layer(input_dim=self.e2_num_features, num_heads=8, dropout=0.1,
                                                    positional_encoding=True)


        # at the end of the cross attention we want to calculate 1D avg pooling and 1D max pooling to 'aggregate'
        # the temporal information
        # at the time of the pooling, embeddings from all different cross-attention blocks will be concatenated
        self.avg_pool = torch.nn.AdaptiveAvgPool1d(1)
        self.max_pool = torch.nn.AdaptiveMaxPool1d(1)
        self.cross_att_dense_layer = torch.nn.Linear((self.e1_num_features + self.e2_num_features)*2, 256)
        self.cross_att_batch_norm = torch.nn.BatchNorm1d(256)
        self.cross_att_activation = torch.nn.Tanh()


    def __forward_cross_attention(self, f, p, a):
        # forward pass through the first stage of self-attention
        # query, keys, values - main feature
        f = self.block_f(query=f, key=f, value=f)
        p = self.block_p(query=p, key=p, value=p)
        a = self.block_a(query=a, key=a, value=a)
        # forward pass through the second stage of cross-attention
        # f or p - main feature, a - supportive feature
        f_a = self.block_f_a(query=f, key=a, value=a)
        p_a = self.block_p_a(query=p, key=a, value=a)
        # forward pass through the third stage of cross-attention
        f_a_p_a = self.block_f_a_p_a(query=f_a, key=p_a, value=p_a)
        p_a_f_a = self.block_p_a_f_a(query=p_a, key=f_a, value=f_a)
        # concatenate f_a and p_a
        result = torch.cat((f_a_p_a, p_a_f_a), dim=-1)

        return result



    def forward(self, f, p, a):
        # cross-attention
        f_p_a = self.__forward_cross_attention(f, p, a)
        # permute it to (batch_size, num_features, sequence_length) for calculating 1D avg pooling and 1D max pooling
        f_p_a = f_p_a.permute(0, 2, 1)  # Output size (batch_size, num_features, sequence_length)
        # 1D avg pooling and 1D max pooling
        avg_pool = self.avg_pool(f_p_a)  # Output size (batch_size, num_features, 1)
        max_pool = self.max_pool(f_p_a)  # Output size (batch_size, num_features, 1)
        # get rid of the dimension with size 1 (last dimension)
        avg_pool = avg_pool.squeeze(-1)  # Output size (batch_size, num_features)
        max_pool = max_pool.squeeze(-1)  # Output size (batch_size, num_features)
        # concat avg_pool and max_pool
        output = torch.cat((avg_pool, max_pool), dim=-1)  # Output size (batch_size, num_features * 2)
        # dense layer + batch norm + activation
        output = self.cross_att_dense_layer(output)  # Output size (batch_size, 256)
        output = self.cross_att_batch_norm(output)  # Output size (batch_size, 256)
        output = self.cross_att_activation(output)  # Output size (batch_size, 256)
        # last classification layer
        output = self.classifier(output)  # Output size (batch_size, num_classes)
        return output







if __name__ == "__main__":
    model = AttentionFusionModel_3dim_v5(256, 256, 256, 3)
    print(model)
    # torch summary
    summary(model, [(10, 20, 256), (10, 20, 256), (10, 20, 256)], device='cuda:0')
