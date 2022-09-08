import gc
import sys

import wandb

from pytorch_utils.callbacks import TorchEarlyStopping, TorchMetricEvaluator
from pytorch_utils.losses import SoftFocalLoss
from pytorch_utils.models.Dense_models import DenseModel
from src.NoXi.visual_subsystem.Fusion.utils import cut_filenames_to_original_names, FusionDataLoader

sys.path.extend(["/work/home/dsu/datatools/"])
sys.path.extend(["/work/home/dsu/engagement_recognition_project_server/"])
sys.path.extend(["/work/home/dsu/simpleHRNet/"])

from collections import Callable
from functools import partial
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torchvision.transforms as T
from sklearn.metrics import recall_score, precision_score, f1_score
from sklearn.utils import compute_class_weight
from torchinfo import summary

def create_model(input_shape:int, num_layers:int, num_neurons:int, activation_function:str, num_classes:int = 5):
    activation_functions = {'relu', 'sigmoid', 'tanh', 'elu'}
    if activation_function not in activation_functions:
        raise ValueError("Activation function must be one of the following: {}".format(activation_functions))




    model = DenseModel(input_shape= input_shape,
                       dense_neurons=tuple(num_neurons for _ in range(num_layers)),
                       activations=activation_function,
                       dropout = 0.3,
                       output_neurons = num_classes,
                       activation_function_for_output='softmax')

    model.train()
    return model



def train_step(model:torch.nn.Module, train_generator:torch.utils.data.DataLoader,
               optimizer:torch.optim.Optimizer, criterion:torch.nn.Module,
               device:torch.device, print_step:int=100):

    running_loss=0.0
    for i, data in enumerate(train_generator):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        inputs = inputs.float()
        inputs, labels = inputs.to(device), labels.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % print_step == (print_step-1):  # print every 100 mini-batches
            print("Mini-batch: %i, loss: %.10f" % (i, running_loss / print_step))
            running_loss = 0.0


def train_model(train, dev, epochs:int, class_weights:Optional=None, loss_function:str="Crossentropy"):
    # metaparams
    metaparams = {
        "optimizer": "Adam",  # SGD, RMSprop, AdamW
        "learning_rate_max": 0.001,  # up to 0.0001
        "learning_rate_min": 0.00001,  # up to 0.000001
        "lr_scheduller": "Cyclic",  # "reduceLRonPlateau"
        "annealing_period": 10,
        "epochs": 100,
        "batch_size": 256,
        "architecture": "Dense_network",
        "dataset": "NoXi",
        "num_classes": 5,
        "num_embeddings": train.get_data_width(),
        "num_layers": 3, # 1,2,3,4
        "num_neurons": 128, # 64, 128, 256, 512
        "activation_function": "relu", # relu, sigmoid, tanh, elu
    }
    # initialization of Weights and Biases
    wandb.init(project="Engagement_recognition_fusion", config=metaparams)
    config = wandb.config


    # create model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_model(input_shape=config.num_embeddings, num_layers=config.num_layers, num_neurons=config.num_neurons,
                         activation_function=config.activation_function, num_classes= config.num_classes)
    model.to(device)
    summary(model, input_size=(config.batch_size, 3, 256, 256))
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
    early_stopping_callback = TorchEarlyStopping(verbose= True, patience = 15,
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

def main():
    print("New start...")
    sweep_config = {
        'name': "Direct_fusion_f2f_PCA_focal_loss",
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
            'num_neurons': {
                'values': [64, 128, 256, 512]
            },
            'num_layers': {
                'values': [1, 2, 3, 4]
            },
            'activation_function': {
                'values': ['ReLU', 'sigmoid', 'tanh', 'elu']
            }
        }
    }
    BATCH_SIZE=256
    # load data
    train_1 = pd.read_csv("/work/home/dsu/NoXi/NoXi_embeddings/All_languages/Pose_model/embeddings_train.csv")
    train_2 = pd.read_csv("/work/home/dsu/NoXi/NoXi_embeddings/All_languages/Xception_model/train_extracted_deep_embeddings.csv")
    train_1, train_2 = cut_filenames_to_original_names(train_1), cut_filenames_to_original_names(train_2)

    dev_1 = pd.read_csv("/work/home/dsu/NoXi/NoXi_embeddings/All_languages/Pose_model/embeddings_dev.csv")
    dev_2 = pd.read_csv("/work/home/dsu/NoXi/NoXi_embeddings/All_languages/Xception_model/dev_extracted_deep_embeddings.csv")
    dev_1, dev_2 = cut_filenames_to_original_names(dev_1), cut_filenames_to_original_names(dev_2)


    # create generators
    train_generator = FusionDataLoader([train_1, train_2], scaling="PCA", PCA_components=100, labels_included=True)
    dev_generator = FusionDataLoader([dev_1, dev_2], scaling="PCA", PCA_components=100, labels_included=True)


    # compute class weights
    train_data = np.concatenate([y for x,y in train_generator], axis=0)
    class_weights = compute_class_weight(class_weight='balanced',
                                         classes=np.unique(np.argmax(train_data, axis=1, keepdims=True)),
                                         y=np.argmax(train_data, axis=1, keepdims=True).flatten())

    # get data PyTorch data loaders

    train_generator_pytorch = torch.utils.data.DataLoader(train_generator, batch_size=BATCH_SIZE, shuffle=True,
                                num_workers=16, pin_memory=False)

    dev_generator_pytorch = torch.utils.data.DataLoader(dev_generator, batch_size=BATCH_SIZE, shuffle=False,
                                                          num_workers=16, pin_memory=False)


    print("Wandb with Focal_loss")
    sweep_id = wandb.sweep(sweep_config, project='VGGFace2_FtF_training')
    wandb.agent(sweep_id, function=lambda: train_model(train_generator_pytorch, dev_generator_pytorch, epochs=100,
                loss_function="Focal_loss", class_weights=class_weights),
                count=300,
                project='VGGFace2_FtF_training')
    gc.collect()


if __name__ == "__main__":
    main()
