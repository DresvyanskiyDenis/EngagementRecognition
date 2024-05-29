import gc
import sys
import glob
import itertools
import os
from functools import partial
from typing import List, Optional

# dynamically append the path to the project to the system path
path_to_project = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, os.pardir, os.pardir))+os.sep
sys.path.append(path_to_project)
sys.path.append(path_to_project.replace('EngagementRecognition', 'datatools'))
sys.path.append(path_to_project.replace('EngagementRecognition', 'simple-HRNet-master'))

import torch
import wandb
from torchinfo import summary

from pytorch_utils.lr_schedullers import WarmUpScheduler
from pytorch_utils.training_utils.callbacks import TorchEarlyStopping
from src.journalPaper.training.fusion_continuous_NoXi.data_preparation import get_train_dev_dataloaders
from src.journalPaper.training.fusion_continuous_NoXi.losses import Batch_CCCloss
from src.journalPaper.training.fusion_continuous_NoXi.model_evaluation import evaluate_model, evaluate_model_full
from src.journalPaper.training.fusion_continuous_NoXi.models import construct_model



from tqdm import tqdm
import click
import numpy as np
import pandas as pd

@click.group()
def cli():
    pass


def train_step(model: torch.nn.Module, criterion: torch.nn.Module,
               inputs: List[torch.Tensor], ground_truth: torch.Tensor) -> List[torch.Tensor]:
    # forward pass
    output = model(*inputs)
    # calculate loss
    loss = criterion(output, ground_truth)
    # clear RAM from unused variables
    del output, ground_truth
    return [loss]


def train_epoch(model: torch.nn.Module, train_generator: torch.utils.data.DataLoader,
                modalities: List[str],
                optimizer: torch.optim.Optimizer, criterion: torch.nn.Module,
                device: torch.device, print_step: int = 100,
                accumulate_gradients: Optional[int] = 1,
                warmup_lr_scheduller: Optional[object] = None,
                loss_multiplication_factor: Optional[float] = None) -> float:
    running_loss = 0.0
    total_loss = 0.0
    counter = 0
    for i, data in enumerate(train_generator):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        # order of modalities is always [face, pose, emo]. We need to choose the ones that are included in current experiment
        tmp = []
        if 'face' in modalities: tmp.append(inputs[0])
        if 'pose' in modalities: tmp.append(inputs[1])
        if 'emo' in modalities: tmp.append(inputs[2])
        inputs = tmp
        # now we have only the modalities that are included in the experiment
        # inputs are the list of tensors. Every tensor has shape (batch_size, sequence_length, num_features)
        inputs = [input.float().to(device) for input in inputs]
        # inputs shape: List of (batch_size, sequence_length, num_features)
        # labels shape: List of (batch_size, sequence_length, 1)
        # as all labels are the same, just take the first element of list
        labels = labels[0].float().to(device)
        # do train step
        with torch.set_grad_enabled(True):
            # form indices of labels which should be one-hot encoded
            step_losses = train_step(model, criterion, inputs, labels)
            # normalize losses by number of accumulate gradient steps
            step_losses = [step_loss / accumulate_gradients for step_loss in step_losses]
            # backward pass
            sum_losses = sum(step_losses)
            if loss_multiplication_factor is not None:
                sum_losses = sum_losses * loss_multiplication_factor
            sum_losses.backward()
            # update weights if we have accumulated enough gradients
            if (i + 1) % accumulate_gradients == 0 or (i + 1 == len(train_generator)):
                optimizer.step()
                optimizer.zero_grad()
                if warmup_lr_scheduller is not None:
                    warmup_lr_scheduller.step()

        # print statistics
        running_loss += sum_losses.item()
        total_loss += sum_losses.item()
        counter += 1
        if i % print_step == (print_step - 1):  # print every print_step mini-batches
            print("Mini-batch: %i, loss: %.10f" % (i, running_loss / print_step))
            running_loss = 0.0
        # clear RAM from all the intermediate variables
        del inputs, labels, step_losses, sum_losses
    # clear RAM at the end of the epoch
    torch.cuda.empty_cache()
    gc.collect()
    return total_loss / counter

def train_model(modalities:str, train_generator: torch.utils.data.DataLoader, dev_generator: torch.utils.data.DataLoader,
                window_size: float, stride: float, consider_timesteps: bool,
                model_type: str, batch_size: int, accumulate_gradients: int,
                loss_multiplication_factor: Optional[float] = None) -> None:
    print("Start of the model training.")
    # metaparams
    metaparams = {
        # general params
        "modalities": modalities.split("_"),
        "architecture": model_type,
        "MODEL_TYPE": model_type,
        "dataset": "NoXi, DAiSEE",
        "BEST_MODEL_SAVE_PATH": "best_models/",
        "NUM_WORKERS": 8,
        # temporal params
        "window_size": window_size,
        "stride": stride,
        "consider_timestamps": consider_timesteps,
        # training metaparams
        "NUM_EPOCHS": 100,
        "BATCH_SIZE": batch_size,
        "OPTIMIZER": "SGD",
        "EARLY_STOPPING_PATIENCE": 50,
        "WEIGHT_DECAY": 0.0001,
        # LR scheduller params
        "LR_SCHEDULLER": "Warmup_cyclic",
        "ANNEALING_PERIOD": 5,
        "LR_MAX_CYCLIC": 0.01,
        "LR_MIN_CYCLIC": 0.0001,
        "LR_MIN_WARMUP": 0.00001,
        "WARMUP_STEPS": 300,
        "WARMUP_MODE": "linear",
        # loss params
        "loss_multiplication_factor": loss_multiplication_factor,
    }
    print("____________________________________________________")
    print("Training params:")
    for key, value in metaparams.items():
        print(f"{key}: {value}")
    print("____________________________________________________")
    # initialization of Weights and Biases
    wandb.init(project="EngagementRecFusion_NoXi_continuous", config=metaparams, entity="denisdresvyanskiy")
    config = wandb.config
    wandb.config.update({'BEST_MODEL_SAVE_PATH': wandb.run.dir}, allow_val_change=True)
    # get one iteration of train generator to get sequence length
    inputs, labels = next(iter(train_generator))
    sequence_length = inputs[0].shape[1]

    # create model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # construct sequence-to-one model out of base model
    model = construct_model(config.MODEL_TYPE)
    model = model.to(device)
    # print model architecture
    #summary(model, [(10, sequence_length, 256), (10, sequence_length, 256), (10, sequence_length, 256)])

    # select optimizer
    model_parameters = model.parameters()
    optimizers = {'Adam': torch.optim.Adam,
                  'SGD': torch.optim.SGD,
                  'RMSprop': torch.optim.RMSprop,
                  'AdamW': partial(torch.optim.AdamW, weight_decay=config.WEIGHT_DECAY)}
    optimizer = optimizers[config.OPTIMIZER](model_parameters, lr=config.LR_MAX_CYCLIC)
    # Loss functions
    criterion = Batch_CCCloss()
    # create LR scheduler
    lr_schedullers = {
        'Warmup_cyclic': WarmUpScheduler(optimizer=optimizer,
                                         lr_scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                                                                 T_max=config.ANNEALING_PERIOD,
                                                                                                 eta_min=config.LR_MIN_CYCLIC),
                                         len_loader=len(train_generator) // accumulate_gradients,
                                         warmup_steps=config.WARMUP_STEPS,
                                         warmup_start_lr=config.LR_MIN_WARMUP,
                                         warmup_mode=config.WARMUP_MODE)
    }
    # if we use discriminative learning, we don't need LR scheduler
    lr_scheduller = lr_schedullers[config.LR_SCHEDULLER]
    # if lr_scheduller is warmup_cyclic, we need to change the learning rate of optimizer
    if config.LR_SCHEDULLER == 'Warmup_cyclic':
        optimizer.param_groups[0]['lr'] = config.LR_MIN_WARMUP

    # early stopping
    best_val_gen_avg_CCC = -100
    best_val_gen_avg_PCC = -100
    best_val_concat_CCC = -100
    best_val_concat_PCC = -100
    early_stopping_callback = TorchEarlyStopping(verbose=True, patience=config.EARLY_STOPPING_PATIENCE,
                                                 save_path=config.BEST_MODEL_SAVE_PATH,
                                                 mode="max")

    # train model
    for epoch in range(config.NUM_EPOCHS):
        print("Epoch: %i" % epoch)
        # train the model
        model.train()
        train_loss = train_epoch(model, train_generator, config.modalities,
                                 optimizer, criterion, device, print_step=100,
                                 accumulate_gradients=accumulate_gradients,
                                 warmup_lr_scheduller=lr_scheduller if config.LR_SCHEDULLER == 'Warmup_cyclic' else None,
                                 loss_multiplication_factor=config.loss_multiplication_factor)
        print("Train loss: %.10f" % train_loss)

        # validate the model
        model.eval()
        print("Evaluation of the model on dev set.")
        #val_metrics = evaluate_model(model=model, generator=dev_generator, device=device, modalities=config.modalities,
        #           metrics_name_prefix="val_", print_metrics=True)
        val_metrics = evaluate_model_full(modalities=config.modalities, model=model,
                                          device=device, metrics_name_prefix="val_", print_metrics=True)

        # update best val metrics got on validation set and log them using wandb
        # also, save model if we got better CCC
        if val_metrics['gen_avg_val_CCC'] > best_val_gen_avg_CCC:
            best_val_gen_avg_CCC = val_metrics['gen_avg_val_CCC']
            wandb.config.update({'gen_avg_val_CCC': best_val_gen_avg_CCC},
                                allow_val_change=True)
            # save best model
            if not os.path.exists(config.BEST_MODEL_SAVE_PATH):
                os.makedirs(config.BEST_MODEL_SAVE_PATH)
            torch.save(model.state_dict(), os.path.join(config.BEST_MODEL_SAVE_PATH, 'best_model_CCC_gen_avg_val.pth'))
       # the same, but for gen_avg_val_PCC
        if val_metrics['gen_avg_val_PCC'] > best_val_gen_avg_PCC:
            best_val_gen_avg_PCC = val_metrics['gen_avg_val_PCC']
            wandb.config.update({'gen_avg_val_PCC': best_val_gen_avg_PCC},
                                allow_val_change=True)
            # save best model
            if not os.path.exists(config.BEST_MODEL_SAVE_PATH):
                os.makedirs(config.BEST_MODEL_SAVE_PATH)
            torch.save(model.state_dict(), os.path.join(config.BEST_MODEL_SAVE_PATH, 'best_model_PCC_gen_avg_val.pth'))
        # the same, but for concat_val_CCC
        if val_metrics['concat_val_CCC'] > best_val_concat_CCC:
            best_val_concat_CCC = val_metrics['concat_val_CCC']
            wandb.config.update({'concat_val_CCC': best_val_concat_CCC},
                                allow_val_change=True)
            # save best model
            if not os.path.exists(config.BEST_MODEL_SAVE_PATH):
                os.makedirs(config.BEST_MODEL_SAVE_PATH)
            torch.save(model.state_dict(), os.path.join(config.BEST_MODEL_SAVE_PATH, 'best_model_CCC_concat_val.pth'))
        # the same, but for concat_val_PCC
        if val_metrics['concat_val_PCC'] > best_val_concat_PCC:
            best_val_concat_PCC = val_metrics['concat_val_PCC']
            wandb.config.update({'concat_val_PCC': best_val_concat_PCC},
                                allow_val_change=True)
            # save best model
            if not os.path.exists(config.BEST_MODEL_SAVE_PATH):
                os.makedirs(config.BEST_MODEL_SAVE_PATH)
            torch.save(model.state_dict(), os.path.join(config.BEST_MODEL_SAVE_PATH, 'best_model_PCC_concat_val.pth'))


        # log everything using wandb
        wandb.log({'epoch': epoch}, commit=False)
        wandb.log({'learning_rate': optimizer.param_groups[0]["lr"]}, commit=False)
        wandb.log(val_metrics, commit=False)
        wandb.log({'train_loss': train_loss})
        # check early stopping
        early_stopping_result = early_stopping_callback(val_metrics['gen_avg_val_CCC'], model)
        if early_stopping_result:
            print("Early stopping")
            break
    # clear RAM
    del model
    gc.collect()
    torch.cuda.empty_cache()





@cli.command("main")
@click.option('--model_type', default=None, help="", type=str)
@click.option('--modalities', default=None, help="", type=str)
@click.option('--window_size', default=None, help="", type=float)
@click.option('--stride', default=None, help="", type=float)
@click.option('--consider_timesteps', default=None, help="", type=bool)
@click.option('--batch_size', default=None, help="", type=int)
@click.option('--accumulate_gradients', default=None, help="", type=int)
@click.option('--loss_multiplication_factor', default=None, help="", type=int)
def main(model_type:str, modalities:str, window_size:float, stride:float, consider_timesteps:bool,
         batch_size:int, accumulate_gradients:int, loss_multiplication_factor:int):

    paths_to_embeddings = {
        "face_train": "/nfs/scratch/ddresvya/NoXi/NoXi/prepared_data/embeddings/static/NoXi_face_embeddings_train.csv",
        "face_dev": "/nfs/scratch/ddresvya/NoXi/NoXi/prepared_data/embeddings/static/NoXi_face_embeddings_dev.csv",
        "pose_train": "/nfs/scratch/ddresvya/NoXi/NoXi/prepared_data/embeddings/static/NoXi_pose_embeddings_train.csv",
        "pose_dev": "/nfs/scratch/ddresvya/NoXi/NoXi/prepared_data/embeddings/static/NoXi_pose_embeddings_dev.csv",
        "emo_train": "/nfs/scratch/ddresvya/NoXi/NoXi/prepared_data/embeddings/static/NoXi_affective_embeddings_train.csv",
        "emo_dev": "/nfs/scratch/ddresvya/NoXi/NoXi/prepared_data/embeddings/static/NoXi_affective_embeddings_dev.csv",
    }
    feature_columns = [
        ['embedding_%i' % i for i in range(256)],
        ['embedding_%i' % i for i in range(256)],
        ['embedding_%i' % i for i in range(256)]
    ]

    train_dataloader, dev_dataloader = get_train_dev_dataloaders(paths_to_embeddings=paths_to_embeddings,
                                                                 window_size=window_size, stride=stride,
                                                                 consider_timesteps=consider_timesteps,
                                                                 feature_columns=feature_columns,
                                                                 label_columns=['engagement'],
                                                                 preprocessing_functions=None,
                                                                 batch_size=batch_size, num_workers = 8)

    train_model(modalities = modalities, train_generator=train_dataloader, dev_generator=dev_dataloader,
    window_size=window_size, stride=stride, consider_timesteps=consider_timesteps,
    model_type=model_type, batch_size=batch_size, accumulate_gradients=accumulate_gradients,
    loss_multiplication_factor = loss_multiplication_factor)



if __name__ == "__main__":
    cli()