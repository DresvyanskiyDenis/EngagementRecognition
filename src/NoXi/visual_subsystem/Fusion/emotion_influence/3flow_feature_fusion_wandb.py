import gc
from functools import partial
from typing import Tuple, Union, Optional, List

import pandas as pd
import numpy as np
import torch
import os
import sys

import wandb
from sklearn.metrics import recall_score, precision_score, f1_score
from sklearn.utils import compute_class_weight

from pytorch_utils.callbacks import TorchEarlyStopping, TorchMetricEvaluator
from pytorch_utils.general_functions import train_step
from pytorch_utils.losses import SoftFocalLoss
from src.NoXi.visual_subsystem.Fusion.utils import cut_filenames_to_original_names, FusionDataLoader, \
    Nflow_FusionDataLoader
from torchinfo import summary


class n_flow_model(torch.nn.Module):
    activation_functions_mapping = {'relu': torch.nn.ReLU,
                                    'sigmoid': torch.nn.Sigmoid,
                                    'tanh': torch.nn.Tanh,
                                    'softmax': torch.nn.Softmax,
                                    'elu': torch.nn.ELU,
                                    'leaky_relu': torch.nn.LeakyReLU,
                                    'linear': None
                                    }

    def __init__(self, input_shapes:Tuple[int,...], n_flows:int, n_flow_layers:int, n_flow_neurons:int,
                 activation_function:str, n_fusion_layers:int, n_fusion_neurons:Tuple[int,...],
                 fusion_activation_function:str, dropout:Optional[float]=None,
                 output_neurons:Union[Tuple[int],int]=7, output_activation_function:str='softmax'):
        super(n_flow_model, self).__init__()
        self.input_shapes = input_shapes
        self.n_flows = n_flows
        self.n_flow_layers = n_flow_layers
        self.n_flow_neurons = n_flow_neurons
        self.activation_function = activation_function
        self.n_fusion_layers = n_fusion_layers
        self.n_fusion_neurons = n_fusion_neurons
        self.fusion_activation_function = fusion_activation_function
        self.dropout = dropout
        self.output_neurons = output_neurons
        self.output_activation_function = output_activation_function
        # build the model
        self._build_model()


    def _build_flow(self, input_shape:int, n_flow_layers:int, n_flow_neurons:int,
                 activation_function:str, dropout:Optional[float]=None):
        # create one "flow" (branch) of the model
        flow = torch.nn.ModuleList()
        # input layer
        flow.append(torch.nn.Linear(input_shape, n_flow_neurons))
        flow.append(self.activation_functions_mapping[activation_function]())
        if dropout:
            flow.append(torch.nn.Dropout(dropout))
        # hidden layers
        for i in range(1, n_flow_layers):
            flow.append(torch.nn.Linear(n_flow_neurons, n_flow_neurons))
            flow.append(self.activation_functions_mapping[activation_function]())
            if dropout:
                flow.append(torch.nn.Dropout(dropout))
        return flow

    def _build_model(self):
        # create flow (branch) layers
        self.flow_layers = torch.nn.ModuleList()
        for i in range(self.n_flows):
            self.flow_layers.append(self._build_flow(self.input_shapes[i], self.n_flow_layers,
                                                     self.n_flow_neurons, self.activation_function,
                                                     self.dropout))
        # create fusion layers and fuse (concatenate first) all outputs of flow layers
        self.fusion_layers = torch.nn.ModuleList()
        self.fusion_layers.append(torch.nn.Linear(self.n_flows*self.n_flow_neurons, self.n_fusion_neurons[0]))
        self.fusion_layers.append(self.activation_functions_mapping[self.fusion_activation_function]())
        for i in range(1, self.n_fusion_layers):
            self.fusion_layers.append(torch.nn.Linear(self.n_fusion_neurons[i-1], self.n_fusion_neurons[i]))
            self.fusion_layers.append(self.activation_functions_mapping[self.fusion_activation_function]())
        # create output layer
        self.output_layer = torch.nn.ModuleList()
        self.output_layer.append(torch.nn.Linear(self.n_fusion_neurons[-1], self.output_neurons))
        if self.output_activation_function == 'softmax':
            self.output_layer.append(torch.nn.Softmax(dim=-1))
        elif self.output_activation_function == 'linear':
            pass
        else:
            self.output_layer.append(self.activation_functions_mapping[self.output_activation_function]())


    def forward(self, x):
        # !important!
        # x will have the shape (batch_size, n_flows, num_features) due to the way the data is loaded (in generator)
        # flow layers
        flow_outputs = []
        for i in range(self.n_flows):
            # take the i-th flow data and squeeze it to (batch_size, num_features)
            flow_output = x[:,i]
            flow_output = torch.squeeze(flow_output)
            for layer in self.flow_layers[i]:
                flow_output = layer(flow_output)
            flow_outputs.append(flow_output)
        # fusion layers
        fusion_output = torch.cat(flow_outputs, dim=-1)
        for layer in self.fusion_layers:
            fusion_output = layer(fusion_output)
        # output layer
        for layer in self.output_layer:
            fusion_output = layer(fusion_output)
        return fusion_output

def preprocess_and_align_data(df1, df2, df3):
    # take only frames that are in both dataframes
    df1 = df1[df1.set_index(['filename']).index.isin(df2.set_index(['filename']).index)]
    df2 = df2[df2.set_index(['filename']).index.isin(df1.set_index(['filename']).index)]
    df3 = df3[df3.set_index(['filename']).index.isin(df1.set_index(['filename']).index)]
    # sort by filename
    df1 = df1.sort_values(by=['filename'])
    df2 = df2.sort_values(by=['filename'])
    df3 = df3.sort_values(by=['filename'])
    # check congruity
    if df1.shape[0] != df2.shape[0] or df1.shape[0] != df3.shape[0]:
        raise ValueError('Dataframes have different number of rows')
    df2.set_index("filename", inplace=True)
    df2.reindex(index=df1['filename'])
    df2.reset_index(inplace=True)

    df3.set_index("filename", inplace=True)
    df3.reindex(index=df1['filename'])
    df3.reset_index(inplace=True)
    return df1, df2, df3

def load_data(scaler:str):
    # load train data
    train_1 = pd.read_csv("/work/home/dsu/NoXi/NoXi_embeddings/All_languages/Pose_model/embeddings_train.csv")
    train_2 = pd.read_csv("/work/home/dsu/NoXi/NoXi_embeddings/All_languages/Xception_model/train_extracted_deep_embeddings.csv")
    train_3 = pd.read_csv("/work/home/dsu/NoXi/NoXi_embeddings/All_languages/EmoVGGFace2/train_extracted_deep_embeddings.csv")
    train_1, train_2, train_3 = cut_filenames_to_original_names(train_1), cut_filenames_to_original_names(train_2), cut_filenames_to_original_names(train_3)
    # change the filenames to the same format for all dataframes
    #train_1['filename'] = train_1['filename'].apply(lambda x: os.path.join(*x.split("/")[2:]))
    #train_2['filename'] = train_2['filename'].apply(lambda x: os.path.join(*x.split("/")[4:]))
    # preprocess and align dataframes. Check the congruity as well
    train_1, train_2, train_3 = preprocess_and_align_data(train_1, train_2, train_3)

    # load validation data
    dev_1 = pd.read_csv("/work/home/dsu/NoXi/NoXi_embeddings/All_languages/Pose_model/embeddings_dev.csv")
    dev_2 = pd.read_csv("/work/home/dsu/NoXi/NoXi_embeddings/All_languages/Xception_model/dev_extracted_deep_embeddings.csv")
    dev_3 = pd.read_csv("/work/home/dsu/NoXi/NoXi_embeddings/All_languages/EmoVGGFace2/dev_extracted_deep_embeddings.csv")
    dev_1, dev_2, dev_3 = cut_filenames_to_original_names(dev_1), cut_filenames_to_original_names(dev_2), cut_filenames_to_original_names(dev_3)
    # change the filenames to the same format for all dataframes
    #dev_1['filename'] = dev_1['filename'].apply(lambda x: os.path.join(*x.split("/")[2:]))
    #dev_2['filename'] = dev_2['filename'].apply(lambda x: os.path.join(*x.split("/")[4:]))
    # preprocess and align dataframes. Check the congruity as well
    dev_1, dev_2, dev_3 = preprocess_and_align_data(dev_1, dev_2, dev_3)

    # create generators out of the loaded data
    if scaler == "PCA":
        PCA_components = 100
    else:
        PCA_components = None
    # create generators
    train_generator_flow_1 = FusionDataLoader([train_1], scaling=scaler, PCA_components=PCA_components,
                                       labels_included=True)
    train_generator_flow_2 = FusionDataLoader([train_2], scaling=scaler, PCA_components=PCA_components,
                                       labels_included=True)
    train_generator_flow_3 = FusionDataLoader([train_3], scaling=scaler, PCA_components=PCA_components,
                                        labels_included=True)
    dev_generator_flow_1 = FusionDataLoader([dev_1], scaling=train_generator_flow_1.scaler, PCA_components=PCA_components,
                                            labels_included=True)
    dev_generator_flow_2 = FusionDataLoader([dev_2], scaling=train_generator_flow_2.scaler,
                                            PCA_components=PCA_components,
                                            labels_included=True)
    dev_generator_flow_3 = FusionDataLoader([dev_3], scaling=train_generator_flow_3.scaler,
                                            PCA_components=PCA_components,
                                            labels_included=True)

    # create train and dev generators, which will concatenate outputs from generators made above
    train_generator = Nflow_FusionDataLoader((train_generator_flow_1, train_generator_flow_2, train_generator_flow_3))
    dev_generator = Nflow_FusionDataLoader((dev_generator_flow_1, dev_generator_flow_2, dev_generator_flow_3))

    return train_generator, dev_generator

def create_model(input_shapes:Tuple[int,...], n_flows:int, n_flow_layers:int, n_flow_neurons:int,
                 activation_function:str, n_fusion_layers:int, n_fusion_neurons:Tuple[int,...],
                 fusion_activation_function:str, dropout:Optional[float]=None,
                 output_neurons:Union[Tuple[int],int]=7, output_activation_function:str='linear'):

    model = n_flow_model(input_shapes, n_flows, n_flow_layers, n_flow_neurons,
                              activation_function, n_fusion_layers, n_fusion_neurons,
                              fusion_activation_function, dropout, output_neurons,
                              output_activation_function)
    model.train()
    return model



def train_model(train:torch.utils.data.DataLoader, dev:torch.utils.data.DataLoader,
                epochs:int, class_weights:Optional=None, loss_function:str="Crossentropy"):
    # metaparams
    data_shape = np.array(iter(train).next()[0].shape)
    metaparams = {
        "optimizer": "Adam",  # SGD, RMSprop, AdamW
        "learning_rate_max": 0.001,  # up to 0.0001
        "learning_rate_min": 0.00001,  # up to 0.000001
        "lr_scheduller": "Cyclic",  # "reduceLRonPlateau"
        "annealing_period": 10,
        "epochs": 100,
        "batch_size": 256,
        "architecture": "3Flow_fusion_network",
        "dataset": "NoXi",
        "num_classes": 5,
        "input_shapes": [data_shape[-1], data_shape[-1], data_shape[-1]],
        "n_flows": data_shape[1],
        "n_flow_layers": 2, # 1,2,3
        "n_flow_neurons": 128, # 64, 128, 256
        "activation_function": "relu", # relu, elu, tanh, sigmoid
        "n_fusion_layers": 2, #  1,2,3
        "n_fusion_neurons": 128, # 32, 64, 128
        "fusion_activation_function": "relu", # relu, elu, tanh, sigmoid
    }

    # initialization of Weights and Biases

    wandb.init(project="Engagement_recognition_fusion", config=metaparams)
    config = wandb.config

    # create model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = create_model(input_shapes=config.input_shapes, n_flows=config.n_flows, n_flow_layers=config.n_flow_layers,
                         n_flow_neurons=config.n_flow_neurons, activation_function=config.activation_function,
                         n_fusion_layers=config.n_fusion_layers,
                         n_fusion_neurons=tuple(int(config.n_fusion_neurons/i) for i in range(1, config.n_fusion_layers+1)),
                         fusion_activation_function=config.fusion_activation_function,
                         dropout=0.3, output_neurons=config.num_classes,
                         output_activation_function='linear')
    model.to(device)
    # check the model graph
    input = [torch.rand(1, metaparams['n_flows'], data_shape[-1]), torch.rand(1, metaparams['n_flows'], data_shape[-1])]
    input = torch.cat(input, dim=0)
    input = input.to(device)
    summary(model, input_data=input)

    # Select optimizer
    optimizers = {'Adam': torch.optim.Adam,
                  'SGD': torch.optim.SGD,
                  'RMSprop': torch.optim.RMSprop,
                  'AdamW': torch.optim.AdamW}

    optimizer = optimizers[config.optimizer](model.parameters(), lr=config.learning_rate_max)
    # select loss function
    class_weights = torch.from_numpy(class_weights).float()
    class_weights = class_weights.to(device)
    criterions = {'Crossentropy': torch.nn.CrossEntropyLoss(weight=class_weights),
                   'Focal_loss': SoftFocalLoss(softmax=True, alpha=class_weights, gamma=2)}
    criterion = criterions[loss_function]
    wandb.config.update({'loss': criterion})
    # Select lr scheduller
    lr_schedullers = {
        'Cyclic':torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.annealing_period, eta_min=config.learning_rate_min),
        'ReduceLRonPlateau':torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode = 'max', patience = 8),
    }
    lr_scheduller = lr_schedullers[config.lr_scheduller]
    # callbacks
    val_metrics = {
        'val_recall': partial(recall_score, average='macro'),
        'val_precision': partial(precision_score, average='macro'),
        'val_f1_score': partial(f1_score, average='macro')
    }
    best_val_recall = 0
    early_stopping_callback = TorchEarlyStopping(verbose= True, patience = 30,
                                                 save_path = wandb.run.dir,
                                                 mode = "max")

    metric_evaluator = TorchMetricEvaluator(generator = dev,
                 model=model,
                 metrics=val_metrics,
                 device=device,
                 output_argmax=True,
                 output_softmax=True,
                 labels_argmax=True,
                 loss_func=criterion)
    # print information about run
    print('Metaparams: optimizer: %s, learning_rate:%f, lr_scheduller: %s, annealing_period: %d, epochs: %d, '
          'batch_size: %d, architecture: %s, '
          'dataset: %s, num_classes: %d' % (config.optimizer, config.learning_rate_max, config.lr_scheduller,
                                            config.annealing_period, config.epochs, config.batch_size,
                                            config.architecture, config.dataset,
                                            config.num_classes))
    # go through epochs
    for epoch in range(epochs):
        # train model one epoch
        loss = train_step(model=model, train_generator=train, optimizer=optimizer, criterion=criterion,
                   device=device, print_step = 100)
        loss = loss/config.batch_size

        # evaluate model on dev set
        print('model evaluation...')
        with torch.no_grad():
            dev_results = metric_evaluator()
            print("Epoch: %i, dev results:"% epoch)
            for metric_name, metric_value in dev_results.items():
                print("%s: %.4f" % (metric_name, metric_value))
            # check early stopping
            early_stopping_result = early_stopping_callback(dev_results['val_recall'], model)
            # check if we have new best recall result on the validation set
            if dev_results['val_recall'] > best_val_recall:
                best_val_recall = dev_results['val_recall']
                wandb.config.update({'best_val_recall' : best_val_recall}, allow_val_change=True)
        # log everything using wandb
        wandb.log({'epoch': epoch}, commit=False)
        wandb.log({'learning_rate':optimizer.param_groups[0]["lr"]}, commit=False)
        wandb.log(dev_results, commit=False)
        wandb.log({'train_loss': loss})
        # update lr
        if config.lr_scheduller == "ReduceLRonPlateau":
            lr_scheduller.step(dev_results['loss'])
        elif config.lr_scheduller == "Cyclic":
            lr_scheduller.step()
        # break the training loop if the model is not improving for a while
        if early_stopping_result:
            break
    # clear RAM
    gc.collect()
    torch.cuda.empty_cache()


def run(scaler:str):
    print("New start...!With Emotions...")
    sweep_config = {
        'name': "3Flow_fusion_f2f_%sScaler_focal_loss(facial, pose, emotional)" % scaler,
        'method': 'random',
        'metric': {
            'name': 'val_loss',
            'goal': 'minimize'
        },
        'parameters': {
            'optimizer': {
                'values': ['Adam', 'SGD', 'RMSprop', 'AdamW']
            },
            'learning_rate_max': {
                'distribution': 'uniform',
                'max': 0.001,
                'min': 0.0001
            },
            'learning_rate_min': {
                'distribution': 'uniform',
                'max': 0.00001,
                'min': 0.000001
            },
            'lr_scheduller': {
                'values': ['Cyclic', 'ReduceLRonPlateau']
            },
            'n_flow_layers': {
                'values': [1, 2, 3]
            },
            'n_flow_neurons': {
                'values': [64, 128, 256]
            },
            'activation_function': {
                'values': ['relu', 'elu', 'tanh', 'sigmoid']
            },
            'n_fusion_layers': {
                'values': [1, 2, 3]
            },
            'n_fusion_neurons': {
                'values': [32, 64, 128]
            },
            'fusion_activation_function': {
                'values': ['relu', 'elu', 'tanh', 'sigmoid']
            }
        }
    }
    BATCH_SIZE = 256

    # load data
    train_generator, dev_generator = load_data(scaler)

    # compute class weights
    train_data = np.concatenate([y[np.newaxis, ...] for x, y in train_generator], axis=0)
    class_weights = compute_class_weight(class_weight='balanced',
                                         classes=np.unique(np.argmax(train_data, axis=1, keepdims=True)),
                                         y=np.argmax(train_data, axis=1, keepdims=True).flatten())
    del train_data
    gc.collect()

    # create PyTorch generators out of the created generators
    train_generator = torch.utils.data.DataLoader(train_generator, batch_size=BATCH_SIZE, shuffle=True, num_workers=16)
    dev_generator = torch.utils.data.DataLoader(dev_generator, batch_size=BATCH_SIZE, shuffle=False, num_workers=16)

    print("Wandb with Focal_loss")
    sweep_id = wandb.sweep(sweep_config, project='Engagement_recognition_fusion')
    wandb.agent(sweep_id, function=lambda: train_model(train_generator, dev_generator, epochs=100,
                                                       loss_function="Focal_loss", class_weights=class_weights),
                count=500,
                project='Engagement_recognition_fusion')
    gc.collect()
    torch.cuda.empty_cache()




if __name__ == '__main__':
    run(scaler="standard")
    run(scaler="PCA")
