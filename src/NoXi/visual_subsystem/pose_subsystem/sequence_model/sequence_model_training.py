import gc
import sys

sys.path.extend(["/work/home/dsu/datatools/"])
sys.path.extend(["/work/home/dsu/engagement_recognition_project_server/"])

from functools import partial
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import wandb
import torch
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
from sklearn.metrics import recall_score, precision_score, f1_score
from sklearn.utils import compute_class_weight
from torchinfo import summary

from decorators.common_decorators import timer
from pytorch_utils.callbacks import TorchEarlyStopping, TorchMetricEvaluator
from pytorch_utils.losses import SoftFocalLoss

from src.NoXi.visual_subsystem.pose_subsystem.sequence_model.sequence_data_loader import SequenceDataLoader
from src.NoXi.visual_subsystem.pose_subsystem.sequence_model.sequence_model import Seq2One_model


@timer
def train_step(model: torch.nn.Module, train_generator: torch.utils.data.DataLoader,
               optimizer: torch.optim.Optimizer, criterion: torch.nn.Module,
               device: torch.device, print_step: int = 100):
    running_loss = 0.0
    total_loss = 0.0
    counter = 0.0
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
        total_loss += loss.item()
        counter += 1.
        if i % print_step == (print_step - 1):  # print every 100 mini-batches
            print("Mini-batch: %i, loss: %.10f" % (i, running_loss / print_step))
            running_loss = 0.0
    return total_loss / counter


def train_model(train, dev, test, epochs: int,  input_shape:Tuple[int,...], class_weights: Optional = None, loss_function: str = "Crossentropy"):
    # metaparams
    metaparams = {
        "optimizer": "Adam",  # SGD, RMSprop, AdamW
        "learning_rate_max": 0.001,  # up to 0.0001
        "learning_rate_min": 0.00001,  # up to 0.000001
        "lr_scheduller": "Cyclic",  # "reduceLRonPlateau"
        "annealing_period": 6,
        "epochs": 100,
        "batch_size": 128,
        "architecture": "Seq2One_LSTM_PyTorch",
        "dataset": "NoXi",
        "num_classes": 5,
        "num_lstm_layers":3, # 1, 2, 3
        "num_lstm_neurons":256, # 512, 256, 128, 64
        'num_embeddings': 256,
        'window_length': 80,
        'window_shift': 40,
    }
    # initialization of Weights and Biases
    wandb.init(project="NoXi_Seq_to_One", config=metaparams)
    config = wandb.config

    # create model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Seq2One_model(input_shape=input_shape, LSTM_neurons=tuple(config.num_lstm_neurons for i in range(config.num_lstm_layers)),
                          dropout=0.3,
                          dense_neurons=(256,),
                          dense_neurons_activation_functions=('tanh',), dense_dropout=None,
                          output_layer_neurons=config.num_classes, output_activation_function='linear')

    model.to(device)
    summary(model, input_size=(config.batch_size, config.window_length, config.num_embeddings))
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
        'Cyclic': torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.annealing_period,
                                                             eta_min=config.learning_rate_min),
        'ReduceLRonPlateau': torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=8),
    }
    lr_scheduller = lr_schedullers[config.lr_scheduller]
    # callbacks
    val_metrics = {
        'val_recall': partial(recall_score, average='macro'),
        'val_precision': partial(precision_score, average='macro'),
        'val_f1_score': partial(f1_score, average='macro')
    }
    best_val_recall = 0
    early_stopping_callback = TorchEarlyStopping(verbose=True, patience=20,
                                                 save_path=wandb.run.dir,
                                                 mode="max")

    metric_evaluator = TorchMetricEvaluator(generator=dev,
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
                          device=device, print_step=100)
        loss = loss / config.batch_size
        print("overall training loss: %f" % loss)
        # evaluate model on dev set
        print('model evaluation...')
        with torch.no_grad():
            dev_results = metric_evaluator()
            print("Epoch: %i, dev results:" % epoch)
            for metric_name, metric_value in dev_results.items():
                print("%s: %.4f" % (metric_name, metric_value))
            # check early stopping
            early_stopping_result = early_stopping_callback(dev_results['val_recall'], model)
            # check if we have new best recall result on the validation set
            if dev_results['val_recall'] > best_val_recall:
                best_val_recall = dev_results['val_recall']
                wandb.config.update({'best_val_recall': best_val_recall}, allow_val_change=True)
        # log everything using wandb
        wandb.log({'epoch': epoch}, commit=False)
        wandb.log({'learning_rate': optimizer.param_groups[0]["lr"]}, commit=False)
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


def run(window_length:int, window_shift:int, sweep_name:str, loss:str = "Focal_loss"):
    print("Start.....")
    # parameters
    BATCH_SIZE = 128
    num_classes=5
    num_embeddings = 256
    sweep_config = {
        'name': sweep_name,
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
            'num_lstm_layers': {
                'values': [1, 2, 3]
            },
            'num_lstm_neurons': {
                'values': [512, 256, 128, 64]
            },
            'window_length': {
                'values': [window_length]
            }
        }
    }

    # load data
    train = pd.read_csv("/work/home/dsu/NoXi/NoXi_embeddings/All_languages/Pose_model/embeddings_train.csv")
    dev = pd.read_csv("/work/home/dsu/NoXi/NoXi_embeddings/All_languages/Pose_model/embeddings_dev.csv")

    # compute class weights
    class_weights = compute_class_weight(class_weight='balanced',
                                         classes=np.unique(np.argmax(train.iloc[:, -num_classes:].values, axis=1, keepdims=True)),
                                         y=np.argmax(train.iloc[:, -num_classes:].values, axis=1, keepdims=True).flatten())

    # create data loaders
    train = SequenceDataLoader(dataframe=train, window_length=window_length, window_shift=window_shift,
                               labels_included=True, scaler="standard")
    scaler = train.scaler
    dev = SequenceDataLoader(dataframe=dev, window_length=window_length, window_shift=window_shift,
                               labels_included=True, scaler=scaler)


    # create generators for torch
    train_generator = torch.utils.data.DataLoader(train, batch_size=BATCH_SIZE, shuffle=True,
                                                  num_workers=16, pin_memory=False)
    dev_generator = torch.utils.data.DataLoader(dev, batch_size=BATCH_SIZE, num_workers=16, pin_memory=False)

    # clear RAM
    train.clear_RAM()
    dev.clear_RAM()
    del train, dev
    gc.collect()

    print("Wandb with Crossentropy, window_length: %i, window_shift: %i" % (window_length, window_shift))
    sweep_id = wandb.sweep(sweep_config, project='NoXi_Seq_to_One')
    wandb.agent(sweep_id, function=lambda: train_model(train_generator, dev_generator, None, epochs=100,
                                                       loss_function=loss, class_weights=class_weights,
                                                       input_shape=(BATCH_SIZE, window_length, num_embeddings)),
                count=195,
                project="NoXi_Seq_to_One")
    del train_generator, dev_generator
    gc.collect()


if __name__ == "__main__":
    print("Hello, my dear friend!!!! Crossentropy loss.")
    loss = "Crossentropy"
    run(window_length=80, window_shift=40, sweep_name='PyTorch_Seq2One_All_languages_Crossentropy_80_40', loss=loss)
    run(window_length=70, window_shift=35, sweep_name='PyTorch_Seq2One_All_languages_Crossentropy_70_35', loss=loss)
    run(window_length=60, window_shift=30, sweep_name='PyTorch_Seq2One_All_languages_Crossentropy_60_30', loss=loss)
    run(window_length=50, window_shift=25, sweep_name='PyTorch_Seq2One_All_languages_Crossentropy_50_25', loss=loss)
    run(window_length=40, window_shift=20, sweep_name='PyTorch_Seq2One_All_languages_Crossentropy_40_20', loss=loss)
    run(window_length=30, window_shift=15, sweep_name='PyTorch_Seq2One_All_languages_Crossentropy_30_15', loss=loss)
    run(window_length=20, window_shift=10, sweep_name='PyTorch_Seq2One_All_languages_Crossentropy_20_10', loss=loss)

    print("Hello, my dear friend!!!! Focal loss.")
    loss = "Focal_loss"
    run(window_length=80, window_shift=40, sweep_name='PyTorch_Seq2One_All_languages_Focal_loss_80_40', loss=loss)
    run(window_length=70, window_shift=35, sweep_name='PyTorch_Seq2One_All_languages_Focal_loss_70_35', loss=loss)
    run(window_length=60, window_shift=30, sweep_name='PyTorch_Seq2One_All_languages_Focal_loss_60_30', loss=loss)
    run(window_length=50, window_shift=25, sweep_name='PyTorch_Seq2One_All_languages_Focal_loss_50_25', loss=loss)
    run(window_length=40, window_shift=20, sweep_name='PyTorch_Seq2One_All_languages_Focal_loss_40_20', loss=loss)
    run(window_length=30, window_shift=15, sweep_name='PyTorch_Seq2One_All_languages_Focal_loss_30_15', loss=loss)
    run(window_length=20, window_shift=10, sweep_name='PyTorch_Seq2One_All_languages_Focal_loss_20_10', loss=loss)

