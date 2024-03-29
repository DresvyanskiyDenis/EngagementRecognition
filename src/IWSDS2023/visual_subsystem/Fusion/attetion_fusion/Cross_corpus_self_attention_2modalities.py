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
from sklearn.metrics import recall_score, precision_score, f1_score
from sklearn.utils import compute_class_weight

from pytorch_utils.callbacks import TorchEarlyStopping, TorchMetricEvaluator
from pytorch_utils.losses import SoftFocalLoss
from src.IWSDS2023.visual_subsystem.Fusion.utils import cut_filenames_to_original_names, FusionDataLoader, \
    Nflow_FusionDataLoader
from torchinfo import summary

from src.IWSDS2023.visual_subsystem.Fusion.utils_seq2one import Nflow_FusionSequenceDataLoader
from pytorch_utils.layers.attention_layers import MultiHeadAttention


class SelfAttentionModel_2modalities(torch.nn.Module):
    activation_functions_mapping = {'relu': torch.nn.ReLU,
                                    'sigmoid': torch.nn.Sigmoid,
                                    'tanh': torch.nn.Tanh,
                                    'softmax': torch.nn.Softmax,
                                    'elu': torch.nn.ELU,
                                    'leaky_relu': torch.nn.LeakyReLU,
                                    'linear': None
                                    }

    def __init__(self, input_shapes:Tuple[Tuple[int,...],...], num_heads:int, dropout:Optional[float]=None,
                 output_neurons:Union[Tuple[int],int]=5):
        super(SelfAttentionModel_2modalities, self).__init__()
        # instance of the input shape is (n_flows, sequence_length, num_features)
        self.input_shapes = input_shapes
        self.sequence_length = input_shapes[0][1]
        self.num_flows = len(input_shapes)
        self.num_heads = num_heads
        self.dropout = dropout
        self.output_neurons = output_neurons

        self._build_model()


    def _build_model(self):
        # create self.attention layer after concat
        # input_sim = sum(element[-1] for element in self.input_shapes), since we have several "flows" of data (different modalities)
        # each of them has a different number of features
        num_features = sum(element[-1] for element in self.input_shapes)
        self.self_attention_layer1 = MultiHeadAttention(input_dim=num_features,
                                                         num_heads=self.num_heads, dropout=self.dropout)
        self.self_attention_layer2 = MultiHeadAttention(input_dim=num_features,
                                                       num_heads=self.num_heads, dropout=self.dropout)
        # create averaging 2D 1x1 Conv
        self.averaging_conv = torch.nn.Conv1d(in_channels=num_features, out_channels=1,
                                              kernel_size=1, stride=1, padding="same")

        # create output layer
        self.output_layer = torch.nn.Linear(in_features=self.sequence_length, out_features=self.output_neurons)


    def forward(self, flow_0, flow_1):
        # flow_n has a shape of (batch_size, sequence_length, num_features)
        # concatenate them first
        concatenated_flows = torch.cat([flow_0, flow_1], dim=-1)
        # apply first self-attention layer
        output = self.self_attention_layer1(concatenated_flows, concatenated_flows, concatenated_flows)
        # apply second self-attention layer
        output = self.self_attention_layer2(output, output, output)
        # permute channels and time dimensions
        output = output.permute(0, 2, 1)
        output = self.averaging_conv(output)
        # squeeze the channel dimension
        output = torch.squeeze(output, dim=1)

        # apply output layer (Conv1D)
        output = self.output_layer(output)
        return output


def train_step(model:torch.nn.Module, train_generator:torch.utils.data.DataLoader,
               optimizer:torch.optim.Optimizer, criterion:torch.nn.Module,
               device:torch.device, print_step:int=100):

    running_loss = 0.0
    total_loss = 0.0
    counter = 0.0
    for i, data in enumerate(train_generator):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        inputs = inputs[0].float().to(device), inputs[1].float().to(device)
        labels = labels.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(*inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        total_loss += loss.item()
        counter += 1.
        if i % print_step == (print_step - 1):  # print every print_step mini-batches
            print("Mini-batch: %i, loss: %.10f" % (i, running_loss / print_step))
            running_loss = 0.0
    return total_loss / counter


def load_data(language:str, window_length:int, window_shift:int):
    # load train data
    train_1 = pd.read_csv("/work/home/dsu/IWSDS2023/NoXi_embeddings/Cross-corpus/%s/"
                          "pose_model/embeddings_train.csv"%language.capitalize())
    train_2 = pd.read_csv("/work/home/dsu/IWSDS2023/NoXi_embeddings/Cross-corpus/%s/"
                          "facial_model/train_embeddings.csv"%language.capitalize())
    train_1, train_2 = cut_filenames_to_original_names(train_1), cut_filenames_to_original_names(train_2)
    # change the filenames to the same format for all dataframes
    train_1['filename'] = train_1['filename'].apply(lambda x: os.path.join(*x.split(os.path.sep)[-3:]))
    train_2['filename'] = train_2['filename'].apply(lambda x: os.path.join(*x.split(os.path.sep)[-3:]))

    # load validation data
    dev_1 = pd.read_csv("/work/home/dsu/IWSDS2023/NoXi_embeddings/Cross-corpus/%s/"
                          "pose_model/embeddings_dev.csv"%language.capitalize())
    dev_2 = pd.read_csv("/work/home/dsu/IWSDS2023/NoXi_embeddings/Cross-corpus/%s/"
                          "facial_model/dev_embeddings.csv"%language.capitalize())
    dev_1, dev_2= cut_filenames_to_original_names(dev_1), cut_filenames_to_original_names(dev_2)
    # change the filenames to the same format for all dataframes
    dev_1['filename'] = dev_1['filename'].apply(lambda x: os.path.join(*x.split(os.path.sep)[-3:]))
    dev_2['filename'] = dev_2['filename'].apply(lambda x: os.path.join(*x.split(os.path.sep)[-3:]))

    # create generators
    train_generator = Nflow_FusionSequenceDataLoader(embeddings=[train_1, train_2], window_length=window_length,
                                                     window_shift=window_shift,
                                                     scalers=["standard", "standard"], labels_included=True,
                                                     sequence_to_one=True, data_as_list=True)
    scalers = train_generator.scalers
    dev_generator = Nflow_FusionSequenceDataLoader(embeddings=[dev_1, dev_2], window_length=window_length,
                                                   window_shift=window_shift,
                                                   scalers=[scalers[0], scalers[1]], labels_included=True,
                                                   sequence_to_one=True, data_as_list=True)
    return train_generator, dev_generator



def train_model(train:torch.utils.data.DataLoader, dev:torch.utils.data.DataLoader,
                epochs:int, class_weights:Optional=None, loss_function:str="Crossentropy"):
    # metaparams
    data_shape = [element.shape for element in iter(train).next()[0]]
    # data_shape is a list of 3 elements, each of them has a shape of (batch_size, sequence_length, num_features)
    # we need to get the number of features for each element



    metaparams = {
        "optimizer": "Adam",  # SGD, RMSprop, AdamW
        "learning_rate_max": 0.001,  # up to 0.0001
        "learning_rate_min": 0.00001,  # up to 0.000001
        "lr_scheduller": "Cyclic",  # "reduceLRonPlateau"
        "annealing_period": 10,
        "epochs": 100,
        "batch_size": 128,
        "architecture": "2Flow_fusion_network_Seq2One",
        "dataset": "IWSDS2023",
        "num_classes": 5,
        "input_shapes": data_shape,
        "n_flows": len(data_shape),
        "num_heads": 8,
        "dropout": 0.1
    }

    # initialization of Weights and Biases

    wandb.init(project="Engagement_recognition_fusion", config=metaparams)
    config = wandb.config

    # create model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = SelfAttentionModel_2modalities(input_shapes=config.input_shapes,
                                       num_heads=config.num_heads, dropout=0.1, output_neurons=5)
    model.to(device)
    # check the model graph
    input = [torch.rand(*data_shape[0]), torch.rand(*data_shape[1])]
    input = [input[0].to(device), input[1].to(device)]
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
        'ReduceLRonPlateau':torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode = 'max', patience = 15),
    }
    lr_scheduller = lr_schedullers[config.lr_scheduller]
    # callbacks
    val_metrics = {
        'val_recall': partial(recall_score, average='macro'),
        'val_precision': partial(precision_score, average='macro'),
        'val_f1_score': partial(f1_score, average='macro')
    }
    best_val_recall = 0
    early_stopping_callback = TorchEarlyStopping(verbose= True, patience = 25,
                                                 save_path = wandb.run.dir,
                                                 mode = "max")

    metric_evaluator = TorchMetricEvaluator(generator = dev,
                 model=model,
                 metrics=val_metrics,
                 device=device,
                 output_argmax=True,
                 output_softmax=True,
                 labels_argmax=True,
                 loss_func=criterion,
                 separate_inputs=True)
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


def run(window_length:int, language:str):
    print("New start...!!!")
    sweep_config = {
        'name': "self_attention_2Flow_all_vs_%s_window_%i" % (language, window_length),
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
        }
    }
    BATCH_SIZE = 128
    # load data
    train_generator, dev_generator = load_data(language = language, window_length=window_length, window_shift=window_length//2)

    # compute class weights
    train_data = np.concatenate([y[np.newaxis, ...] for x, y in train_generator], axis=0)
    class_weights = compute_class_weight(class_weight='balanced',
                                         classes=np.unique(np.argmax(train_data, axis=1, keepdims=True)),
                                         y=np.argmax(train_data, axis=1, keepdims=True).flatten())
    del train_data
    gc.collect()

    # create PyTorch generators out of the created generators
    train_generator = torch.utils.data.DataLoader(train_generator, batch_size=BATCH_SIZE, shuffle=True)
    dev_generator = torch.utils.data.DataLoader(dev_generator, batch_size=BATCH_SIZE, shuffle=False)

    print("Wandb with Focal_loss")
    sweep_id = wandb.sweep(sweep_config, project='Engagement_recognition_fusion')
    wandb.agent(sweep_id, function=lambda: train_model(train_generator, dev_generator, epochs=100,
                                                       loss_function="Focal_loss", class_weights=class_weights),
                count=100,
                project='Engagement_recognition_fusion')
    gc.collect()
    torch.cuda.empty_cache()




if __name__ == '__main__':
    run(window_length=80, language='english')
    run(window_length=60, language='english')
    run(window_length=40, language='english')
    run(window_length=20, language='english')

    run(window_length=80, language='german')
    run(window_length=60, language='german')
    run(window_length=40, language='german')
    run(window_length=20, language='german')

    run(window_length=80, language='french')
    run(window_length=60, language='french')
    run(window_length=40, language='french')
    run(window_length=20, language='french')
