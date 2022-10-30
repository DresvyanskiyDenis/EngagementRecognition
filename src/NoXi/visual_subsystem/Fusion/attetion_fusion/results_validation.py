import sys

sys.path.extend(["/work/home/dsu/datatools/"])
sys.path.extend(["/work/home/dsu/engagement_recognition_project_server/"])
sys.path.extend(["/work/home/dsu/simpleHRNet/"])


import gc
from functools import partial
from typing import Tuple, Union, Optional, List

import pandas as pd
import numpy as np
import torch
import os

import wandb
from sklearn.metrics import recall_score, precision_score, f1_score, accuracy_score
from sklearn.utils import compute_class_weight

from pytorch_utils.callbacks import TorchEarlyStopping, TorchMetricEvaluator
from pytorch_utils.general_functions import train_step
from pytorch_utils.losses import SoftFocalLoss
from src.NoXi.visual_subsystem.Fusion.utils import cut_filenames_to_original_names, FusionDataLoader, \
    Nflow_FusionDataLoader
from torchinfo import summary

from src.NoXi.visual_subsystem.Fusion.utils_seq2one import Nflow_FusionSequenceDataLoader

from pytorch_utils.layers.attention_layers import MultiHeadAttention


class TwoFlowCrossAttentionModel(torch.nn.Module):
    activation_functions_mapping = {'relu': torch.nn.ReLU,
                                    'sigmoid': torch.nn.Sigmoid,
                                    'tanh': torch.nn.Tanh,
                                    'softmax': torch.nn.Softmax,
                                    'elu': torch.nn.ELU,
                                    'leaky_relu': torch.nn.LeakyReLU,
                                    'linear': None
                                    }

    def __init__(self, input_shapes:Tuple[Tuple[int,...],...], gru_neurons:int, num_heads:int, dropout:Optional[float]=None,
                 output_neurons:Union[Tuple[int],int]=5):
        super(TwoFlowCrossAttentionModel, self).__init__()
        # instance of the input shape is (n_flows, sequence_length, num_features)
        self.input_shapes = input_shapes
        self.sequence_length = input_shapes[0][1]
        self.num_flows = len(input_shapes)
        self.gru_neurons = gru_neurons
        self.num_heads = num_heads
        self.dropout = dropout
        self.output_neurons = output_neurons

        self._build_model()

    def _build_model(self):
        # create GRU layers
        self.gru_layers = torch.nn.ModuleList()
        for i in range(self.num_flows):
            layer = torch.nn.GRU(input_size=self.input_shapes[i][-1], hidden_size=self.gru_neurons,
                                 num_layers=1, batch_first=True)
            self.gru_layers.append(layer)
        # create cross-attention layers
        self.cross_attention_pose = MultiHeadAttention(input_dim=self.gru_neurons,
                                                       # num of hidden neurons from the GRU layer
                                                       num_heads=self.num_heads, dropout=self.dropout)
        self.cross_attention_facial = MultiHeadAttention(input_dim=self.gru_neurons,
                                                         # num of hidden neurons from the GRU layer
                                                         num_heads=self.num_heads, dropout=self.dropout)
        # create self.attention layer after concat
        self.self_attention_layer = MultiHeadAttention(input_dim=self.gru_neurons * 2,
                                                       # here we have *2, since the features after
                                                       # concatenation (after 2 cross-attention layers) will be doubled
                                                       num_heads=self.num_heads * 2, dropout=self.dropout)
        # create averaging 2D 1x1 Conv
        self.averaging_conv = torch.nn.Conv1d(in_channels=self.gru_neurons * 2, out_channels=1,
                                              kernel_size=1, stride=1, padding="same")

        # create output layer
        self.output_layer = torch.nn.Linear(in_features=self.sequence_length, out_features=self.output_neurons)

    def forward(self, x):
        # x has the shape (batch_size, n_flows, sequence_length, num_features)
        # first flow is pose, second is facial
        # go through the GRU layers
        gru_outputs = []
        for i in range(self.num_flows):
            gru_output, hidden_state = self.gru_layers[i](x[:, i])
            gru_outputs.append(gru_output)
        # take data from gru outputs
        pose_flow = gru_outputs[0]
        facial_flow = gru_outputs[1]
        # go through the pose cross-attention
        pose_cross_attention_output = self.cross_attention_pose(queries=pose_flow,
                                                                keys=facial_flow, values=facial_flow)
        # go through the facial cross-attention
        facial_cross_attention_output = self.cross_attention_facial(queries=facial_flow,
                                                                    keys=pose_flow, values=pose_flow)
        # concatenate the outputs from the cross-attention layers
        concat_output = torch.cat([pose_cross_attention_output, facial_cross_attention_output], dim=-1)
        # go through the self-attention layer
        self_attention_output = self.self_attention_layer(queries=concat_output, keys=concat_output,
                                                          values=concat_output)
        # average the output from the self-attention layer for every timestep using Conv2D
        self_attention_output = self_attention_output.permute(0, 2, 1)
        averaged_output = self.averaging_conv(self_attention_output)
        # flatten the output
        averaged_output = averaged_output.squeeze(dim=1)
        #averaged_output = averaged_output.view(averaged_output.shape[0], -1)
        # go through the output layer
        output = self.output_layer(averaged_output)

        # output shape is (batch_size, output_neurons)
        return output


def load_data(language:str, window_length:int, window_shift:int):
    # load train data
    train_1 = pd.read_csv("/work/home/dsu/NoXi/NoXi_embeddings/Cross-corpus/%s/"
                          "pose_model/embeddings_train.csv"%language.capitalize())
    train_2 = pd.read_csv("/work/home/dsu/NoXi/NoXi_embeddings/Cross-corpus/%s/"
                          "facial_model/train_embeddings.csv"%language.capitalize())
    train_1, train_2 = cut_filenames_to_original_names(train_1), cut_filenames_to_original_names(train_2)
    # change the filenames to the same format for all dataframes
    train_1['filename'] = train_1['filename'].apply(lambda x: os.path.join(*x.split(os.path.sep)[-3:]))
    train_2['filename'] = train_2['filename'].apply(lambda x: os.path.join(*x.split(os.path.sep)[-3:]))

    # load validation data
    dev_1 = pd.read_csv("/work/home/dsu/NoXi/NoXi_embeddings/Cross-corpus/%s/"
                          "pose_model/embeddings_dev.csv"%language.capitalize())
    dev_2 = pd.read_csv("/work/home/dsu/NoXi/NoXi_embeddings/Cross-corpus/%s/"
                          "facial_model/dev_embeddings.csv"%language.capitalize())
    dev_1, dev_2 = cut_filenames_to_original_names(dev_1), cut_filenames_to_original_names(dev_2)
    # change the filenames to the same format for all dataframes
    dev_1['filename'] = dev_1['filename'].apply(lambda x: os.path.join(*x.split(os.path.sep)[-3:]))
    dev_2['filename'] = dev_2['filename'].apply(lambda x: os.path.join(*x.split(os.path.sep)[-3:]))

    # load test data
    test_1 = pd.read_csv("/work/home/dsu/NoXi/NoXi_embeddings/Cross-corpus/%s/"
                            "pose_model/embeddings_test.csv"%language.capitalize())
    test_2 = pd.read_csv("/work/home/dsu/NoXi/NoXi_embeddings/Cross-corpus/%s/"
                            "facial_model/test_embeddings.csv"%language.capitalize())
    test_1, test_2 = cut_filenames_to_original_names(test_1), cut_filenames_to_original_names(test_2)
    # change the filenames to the same format for all dataframes
    test_1['filename'] = test_1['filename'].apply(lambda x: os.path.join(*x.split(os.path.sep)[-3:]))
    test_2['filename'] = test_2['filename'].apply(lambda x: os.path.join(*x.split(os.path.sep)[-3:]))

    # create generators
    train_generator = Nflow_FusionSequenceDataLoader(embeddings=[train_1, train_2], window_length=window_length,
                                                     window_shift=window_shift,
                                                     scalers=["standard", "standard"], labels_included=True,
                                                     sequence_to_one=True)
    scalers = train_generator.scalers
    dev_generator = Nflow_FusionSequenceDataLoader(embeddings=[dev_1, dev_2], window_length=window_length,
                                                   window_shift=window_shift,
                                                   scalers=[scalers[0], scalers[1]], labels_included=True,
                                                   sequence_to_one=True)

    test_generator = Nflow_FusionSequenceDataLoader(embeddings=[test_1, test_2], window_length=window_length,
                                                    window_shift=window_shift,
                                                    scalers=[scalers[0], scalers[1]], labels_included=True,
                                                    sequence_to_one=True)

    return train_generator, dev_generator, test_generator



def validate_model(model, data_loader, device):
    metrics = {
        'recall': partial(recall_score, average='macro'),
        'precision': partial(precision_score, average='macro'),
        'f1_score': partial(f1_score, average='macro'),
        'accuracy': accuracy_score
    }


    model.eval()

    metric_evaluator = TorchMetricEvaluator(generator=data_loader,
                                            model=model,
                                            metrics=metrics,
                                            device=device,
                                            output_argmax=True,
                                            output_softmax=True,
                                            labels_argmax=True,
                                            loss_func=None)

    metric_results = metric_evaluator()
    for metric_name, metric_value in metric_results.items():
        print(f'{metric_name}: {metric_value}')
    s = str(metric_results['recall'])+','+str(metric_results['precision'])+','+str(metric_results['f1_score'])+','+str(metric_results['accuracy'])
    print(s)
    model.train()
    return s




def run(language:str, path_to_model_weights):
    # load models parameters
    model_params = pd.read_csv(os.path.join(path_to_model_weights, 'info.csv'))

    config = {
        "optimizer": "Adam",  # SGD, RMSprop, AdamW
        "learning_rate_max": 0.001,  # up to 0.0001
        "learning_rate_min": 0.00001,  # up to 0.000001
        "lr_scheduller": "Cyclic",  # "reduceLRonPlateau"
        "annealing_period": 10,
        "epochs": 100,
        "batch_size": 128,
        "architecture": "2Flow_fusion_network_Seq2One",
        "dataset": "NoXi",
        "num_classes": 5,
        "gru_neurons": 128,  # 64, 128, 256
        "num_heads": 4,
        "dropout": 0.1
    }


    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    final_results = pd.DataFrame(columns=['ID', 'val_recall', 'val_precision', 'val_f1_score', 'val_accuracy',
                                          'test_recall', 'test_precision', 'test_f1_score', 'test_accuracy'])


    for index, row in model_params.iterrows():

        # load data
        train_generator, dev_generator, test_generator = load_data(language=language,
                                                                   window_length=row["window_length"],
                                                                   window_shift=row["window_length"] // 2)
        # create PyTorch generators out of the created generators
        train_generator = torch.utils.data.DataLoader(train_generator, batch_size=128, shuffle=True)
        dev_generator = torch.utils.data.DataLoader(dev_generator, batch_size=128, shuffle=False)
        test_generator = torch.utils.data.DataLoader(test_generator, batch_size=128, shuffle=False)

        data_shape = np.array(iter(train_generator).next()[0].shape)  # (n_flows, sequence_length, num_features)
        input_shape = data_shape[-3], data_shape[-2], data_shape[-1]
        input_shape = (input_shape,input_shape)


        print("-------------------------------------")
        print("-------------------val data-------------------")
        print("model: ", row['ID'])

        model = TwoFlowCrossAttentionModel(input_shapes=input_shape, gru_neurons=row["gru_neurons"],
                                           num_heads=config["num_heads"], dropout=0.1, output_neurons=5)
        model_weights_path= os.path.join(path_to_model_weights, row['ID'])
        model.load_state_dict(torch.load(model_weights_path))
        model.to(device)

        val_results=validate_model(model, dev_generator, device)
        print("-------------------test data-------------------")
        test_results=validate_model(model, test_generator, device)
        val_test = val_results+','+test_results
        val_test = val_test.split(',')
        val_test = [float(i) for i in val_test]
        val_test = [row['ID']]+val_test
        final_results = final_results.append(pd.DataFrame(data = np.array(val_test)[np.newaxis,...], columns=final_results.columns), ignore_index=True)
    final_results.to_csv(os.path.join(path_to_model_weights, 'final_results.csv'), index=False)





if __name__ == '__main__':
    run(language="german",
        path_to_model_weights="/work/home/dsu/Model_weights/weights_of_best_models/Fusion/Cross_corpus/German/cross_attention_2Flow/")