import sys



sys.path.append('/nfs/home/ddresvya/scripts/EngagementRecognition/')
sys.path.append('/nfs/home/ddresvya/scripts/datatools/')
sys.path.append('/nfs/home/ddresvya/scripts/simple-HRNet-master/')

import argparse
from torchinfo import summary
import gc
import os
from functools import partial
from typing import Tuple, List, Dict, Optional

import numpy as np
import torch
from sklearn.metrics import recall_score, precision_score, f1_score, accuracy_score

from pytorch_utils.lr_schedullers import WarmUpScheduler
from pytorch_utils.training_utils.callbacks import TorchEarlyStopping
from pytorch_utils.training_utils.losses import SoftFocalLoss

import wandb

from src.journalPaper.training.fusion import training_config
from src.journalPaper.training.fusion.data_preparation import get_train_dev_test, calculate_class_weights, \
    construct_data_loaders
from src.journalPaper.training.fusion.fusion_models import AttentionFusionModel_2dim
from src.journalPaper.training.fusion.model_evaluation import evaluate_model

def construct_model()->torch.nn.Module:

    model = AttentionFusionModel_2dim(e1_num_features=256, e2_num_features=256, num_classes=3)

    return model


def train_step(model: torch.nn.Module, criterion: torch.nn.Module,
               inputs: List[torch.Tensor], ground_truth: torch.Tensor,
               device: torch.device) -> List:
    # forward pass
    output = model(*inputs)
    # criterion
    classification_criterion = criterion
    # calculate loss based on mask
    ground_truth = ground_truth.to(device)
    loss = classification_criterion(output, ground_truth)

    # clear RAM from unused variables
    del output, ground_truth

    return [loss]


def train_epoch(model: torch.nn.Module, train_generator: torch.utils.data.DataLoader,
                optimizer: torch.optim.Optimizer, criterion: torch.nn.Module,
                device: torch.device, print_step: int = 100,
                accumulate_gradients: Optional[int] = 1,
                warmup_lr_scheduller: Optional[object] = None,
                loss_multiplication_factor: Optional[float] = None) -> float:
    """ Performs one epoch of training for a model.

    :param model: torch.nn.Module
            Model to train.
    :param train_generator: torch.utils.data.DataLoader
            Generator for training data. Note that it should output the ground truths as a tuple of torch.Tensor
            (thus, we have several outputs).
    :param optimizer: torch.optim.Optimizer
            Optimizer for training.
    :param criterion: torch.nn.Module
            Loss functions for each output of the model.
    :param device: torch.device
            Device to use for training.
    :param print_step: int
            Number of mini-batches between two prints of the running loss.
    :param accumulate_gradients: Optional[int]
            Number of mini-batches to accumulate gradients for. If 1, no accumulation is performed.
    :param warmup_lr_scheduller: Optional[torch.optim.lr_scheduler]
            Learning rate scheduller in case we have warmup lr scheduller. In that case, the learning rate is being changed
            after every mini-batch, therefore should be passed to this function.
    :return: float
            Average loss for the epoch.
    """

    running_loss = 0.0
    total_loss = 0.0
    counter = 0
    for i, data in enumerate(train_generator):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        # inputs are the list of tensors. Every tensor has shape (batch_size, sequence_length, num_features)
        inputs = [input.float().to(device) for input in inputs]
        # transform labels to the sequence-to-one problem
        # however, here, in the training, we have soft labels. Therefore, we do not take mode, instead, we
        # average them and normalize so that the sum of the values is 1
        # labels shape: (batch_size, sequence_length, num_classes)
        labels = labels.sum(axis=-2) / labels.sum(axis=-2).sum(axis=-1, keepdims=True)
        # labels shape after transformation: (batch_size, num_classes)

        # do train step
        with torch.set_grad_enabled(True):
            # form indices of labels which should be one-hot encoded
            step_losses = train_step(model, criterion, inputs, labels, device)
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


def train_model(train_generator: torch.utils.data.DataLoader, dev_generator: torch.utils.data.DataLoader,
                window_size: float, stride: float, consider_timestamps: bool,
                class_weights: torch.Tensor, model_type: str, batch_size: int, accumulate_gradients: int,
                loss_multiplication_factor: Optional[float] = None) -> None:
    print("Start of the model training.")
    # metaparams
    metaparams = {
        # general params
        "architecture": model_type,
        "MODEL_TYPE": model_type,
        "dataset": "NoXi, DAiSEE",
        "BEST_MODEL_SAVE_PATH": training_config.BEST_MODEL_SAVE_PATH,
        "NUM_WORKERS": training_config.NUM_WORKERS,
        # temporal params
        "window_size": window_size,
        "stride": stride,
        "consider_timestamps": consider_timestamps,
        # model architecture
        "NUM_CLASSES": training_config.NUM_CLASSES,
        # training metaparams
        "NUM_EPOCHS": training_config.NUM_EPOCHS,
        "BATCH_SIZE": batch_size,
        "OPTIMIZER": training_config.OPTIMIZER,
        "EARLY_STOPPING_PATIENCE": training_config.EARLY_STOPPING_PATIENCE,
        "WEIGHT_DECAY": training_config.WEIGHT_DECAY,
        # LR scheduller params
        "LR_SCHEDULLER": training_config.LR_SCHEDULLER,
        "ANNEALING_PERIOD": training_config.ANNEALING_PERIOD,
        "LR_MAX_CYCLIC": training_config.LR_MAX_CYCLIC,
        "LR_MIN_CYCLIC": training_config.LR_MIN_CYCLIC,
        "LR_MIN_WARMUP": training_config.LR_MIN_WARMUP,
        "WARMUP_STEPS": training_config.WARMUP_STEPS,
        "WARMUP_MODE": training_config.WARMUP_MODE,
        # loss params
        "loss_multiplication_factor": loss_multiplication_factor,
    }
    print("____________________________________________________")
    print("Training params:")
    for key, value in metaparams.items():
        print(f"{key}: {value}")
    print("____________________________________________________")
    # initialization of Weights and Biases
    wandb.init(project="EngagementRecFusion", config=metaparams)
    config = wandb.config
    wandb.config.update({'BEST_MODEL_SAVE_PATH': wandb.run.dir}, allow_val_change=True)
    # get one iteration of train generator to get sequence length
    inputs, labels = next(iter(train_generator))
    sequence_length = inputs[0].shape[1]

    # create model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # construct sequence-to-one model out of base model
    model = construct_model()
    model = model.to(device)
    # print model architecture
    summary(model, [(10, sequence_length, 256), (10, sequence_length, 256)])

    # select optimizer
    model_parameters = model.parameters()
    optimizers = {'Adam': torch.optim.Adam,
                  'SGD': torch.optim.SGD,
                  'RMSprop': torch.optim.RMSprop,
                  'AdamW': torch.optim.AdamW}
    optimizer = optimizers[config.OPTIMIZER](model_parameters, lr=config.LR_MAX_CYCLIC,
                                             weight_decay=config.WEIGHT_DECAY)
    # Loss functions
    class_weights = class_weights.to(device)
    criterion = SoftFocalLoss(softmax=True, alpha=class_weights, gamma=2)
    # create LR scheduler
    lr_schedullers = {
        'Cyclic': torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.ANNEALING_PERIOD,
                                                             eta_min=config.LR_MIN_CYCLIC),
        'ReduceLRonPlateau': torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=8),
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
    best_val_recall_classification = 0
    early_stopping_callback = TorchEarlyStopping(verbose=True, patience=config.EARLY_STOPPING_PATIENCE,
                                                 save_path=config.BEST_MODEL_SAVE_PATH,
                                                 mode="max")

    # train model
    for epoch in range(config.NUM_EPOCHS):
        print("Epoch: %i" % epoch)
        # train the model
        model.train()
        train_loss = train_epoch(model, train_generator, optimizer, criterion, device, print_step=100,
                                 accumulate_gradients=accumulate_gradients,
                                 warmup_lr_scheduller=lr_scheduller if config.LR_SCHEDULLER == 'Warmup_cyclic' else None,
                                 loss_multiplication_factor=config.loss_multiplication_factor)
        print("Train loss: %.10f" % train_loss)

        # validate the model
        model.eval()
        print("Evaluation of the model on dev set.")
        val_metrics_classification = evaluate_model(model=model, generator=dev_generator, device=device,
                   metrics_name_prefix="val_", print_metrics=True)

        # update best val metrics got on validation set and log them using wandb
        # also, save model if we got better recall
        if val_metrics_classification['val_recall_classification'] > best_val_recall_classification:
            best_val_recall_classification = val_metrics_classification['val_recall_classification']
            wandb.config.update({'best_val_recall_classification': best_val_recall_classification},
                                allow_val_change=True)
            # save best model
            if not os.path.exists(config.BEST_MODEL_SAVE_PATH):
                os.makedirs(config.BEST_MODEL_SAVE_PATH)
            torch.save(model.state_dict(), os.path.join(config.BEST_MODEL_SAVE_PATH, 'best_model_recall.pth'))

        # log everything using wandb
        wandb.log({'epoch': epoch}, commit=False)
        wandb.log({'learning_rate': optimizer.param_groups[0]["lr"]}, commit=False)
        wandb.log(val_metrics_classification, commit=False)
        wandb.log({'train_loss': train_loss})
        # update LR if needed
        if config.LR_SCHEDULLER == 'ReduceLRonPlateau':
            lr_scheduller.step(best_val_recall_classification)
        elif config.LR_SCHEDULLER == 'Cyclic':
            lr_scheduller.step()

        # check early stopping
        early_stopping_result = early_stopping_callback(best_val_recall_classification, model)
        if early_stopping_result:
            print("Early stopping")
            break
    # clear RAM
    del model
    gc.collect()
    torch.cuda.empty_cache()


def main(path_to_embeddings:str, embeddings_type:List[str],
         window_size, stride, consider_timestamps, model_type, batch_size, accumulate_gradients,
         loss_multiplication_factor):
    # embeddings type can be ['face', 'pose', 'emo']
    labels_columns = ['label_0', 'label_1', 'label_2']
    # load data
    train, dev, test = get_train_dev_test(path_to_embeddings=path_to_embeddings, embeddings_type=embeddings_type)
    # the train, for example, is the Dict[str, pd.DataFrame], where keys are video names and values are dataframes
    # get class weights
    class_weights = calculate_class_weights(train[0], labels_columns)
    # get train, dev, test generators
    train_loader, dev_loder, test_loader = construct_data_loaders(train=train, dev=dev,
                           test=test,
                           window_size=window_size, stride=stride, consider_timestamps=consider_timestamps,
                           label_columns=labels_columns,
                           preprocessing_functions=None,
                           batch_size=batch_size,
                           num_workers = 8)

    # train model
    train_model(train_generator=train_loader, dev_generator=dev_loder,
                window_size=window_size, stride=stride, consider_timestamps=consider_timestamps,
                class_weights=class_weights, model_type=model_type, batch_size=batch_size, accumulate_gradients=accumulate_gradients,
                loss_multiplication_factor=loss_multiplication_factor)



if __name__ == '__main__':
    # parse args
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_embeddings', type=str, required=True)
    parser.add_argument('--embeddings_type', type=str, required=True) # shoud be written as a string with spaces between types Example: 'face pose emo'
    parser.add_argument('--model_type', type=str, required=True)
    parser.add_argument('--window_size', type=float, required=True)
    parser.add_argument('--stride', type=float, required=True)
    parser.add_argument('--consider_timestamps', type=int, required=True)
    parser.add_argument('--batch_size', type=int, required=True)
    parser.add_argument('--accumulate_gradients', type=int, required=True)
    parser.add_argument('--loss_multiplication_factor', type=float, required=False, default=1.0)
    args = parser.parse_args()
    print("Passed args: ", args)
    # convert args to variables
    path_embeddings = args.path_embeddings
    embeddings_type = args.embeddings_type.split(' ')
    model_type = args.model_type
    batch_size = args.batch_size
    accumulate_gradients = args.accumulate_gradients
    loss_multiplication_factor = args.loss_multiplication_factor
    window_size = args.window_size
    stride = args.stride

    # check args
    for emb_type in embeddings_type:
        assert emb_type in ['face', 'pose', 'emo'], "embeddings_type should be one of ['face', 'pose', 'emo']"

    consider_timestamps = bool(args.consider_timestamps)
    # run main script with passed args
    main(path_to_embeddings=path_embeddings, embeddings_type=embeddings_type,
         window_size=window_size, stride=stride, consider_timestamps=consider_timestamps,
         model_type=model_type, batch_size=batch_size, accumulate_gradients=accumulate_gradients,
         loss_multiplication_factor=loss_multiplication_factor)
    # clear RAM
    gc.collect()
    torch.cuda.empty_cache()

