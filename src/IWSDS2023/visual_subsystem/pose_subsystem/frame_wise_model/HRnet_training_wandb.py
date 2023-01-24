import gc
import os
import sys
sys.path.extend(["/work/home/dsu/datatools/"])
sys.path.extend(["/work/home/dsu/engagement_recognition_project_server/"])
sys.path.extend(["/work/home/dsu/simpleHRNet/"])

from collections import Callable
from functools import partial
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import wandb
import torch
import torchvision.transforms as T
from sklearn.metrics import recall_score, precision_score, f1_score
from sklearn.utils import compute_class_weight
from torchinfo import summary

from decorators.common_decorators import timer
from pytorch_utils.callbacks import TorchEarlyStopping, TorchMetricEvaluator
from pytorch_utils.generators.ImageDataGenerator import ImageDataLoader
from pytorch_utils.generators.pytorch_augmentations import pad_image_random_factor, grayscale_image, \
    collor_jitter_image_random, gaussian_blur_image_random, random_perspective_image, random_rotation_image, \
    random_crop_image, random_posterize_image, random_adjust_sharpness_image, random_equalize_image, \
    random_horizontal_flip_image, random_vertical_flip_image
from pytorch_utils.losses import FocalLoss, SoftFocalLoss
from src.IWSDS2023.visual_subsystem.pose_subsystem.frame_wise_model.HRNet import load_HRNet_model, modified_HRNet
from src.IWSDS2023.visual_subsystem.pose_subsystem.frame_wise_model.utils import load_NoXi_data_all_languages, \
    convert_image_to_float_and_scale

@timer
def train_step(model:torch.nn.Module, train_generator:torch.utils.data.DataLoader,
               optimizer:torch.optim.Optimizer, criterion:torch.nn.Module,
               device:torch.device, print_step:int=100):

    running_loss=0.0
    total_loss=0.0
    counter=0.0
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
        counter+=1.
        if i % print_step == (print_step-1):  # print every 100 mini-batches
            print("Mini-batch: %i, loss: %.10f" % (i, running_loss / print_step))
            running_loss = 0.0
    return total_loss/counter

def get_data_loaders_from_data(train, dev, test, augment:bool, augment_prob:float, preprocessing_functions:List[Callable]=None, batch_size:int=32):
    if augment:
        augmentation_functions = {
            pad_image_random_factor: augment_prob,
            grayscale_image: augment_prob,
            partial(collor_jitter_image_random, brightness=0.5, hue=0.3, contrast=0.3, saturation=0.3): augment_prob,
            partial(gaussian_blur_image_random, kernel_size=(5, 9), sigma=(0.1, 5)): augment_prob,
            random_perspective_image: augment_prob,
            random_rotation_image: augment_prob,
            partial(random_crop_image, cropping_factor_limits=(0.7, 0.9)): augment_prob,
            random_posterize_image: augment_prob,
            partial(random_adjust_sharpness_image, sharpness_factor_limits=(0.1, 3)): augment_prob,
            random_equalize_image: augment_prob,
            random_horizontal_flip_image: augment_prob,
            random_vertical_flip_image: augment_prob
        }
    else:
        augmentation_functions = {}

    # train
    train_generator = ImageDataLoader(labels=pd.DataFrame(train.iloc[:,1:]), paths_to_images=pd.DataFrame(train.iloc[:,0]), paths_prefix=None,
                                      preprocessing_functions=preprocessing_functions,
                                      augment=augment,
                                      augmentation_functions=augmentation_functions)

    train_generator = torch.utils.data.DataLoader(train_generator, batch_size=batch_size, shuffle=True,
                                              num_workers=16, pin_memory=False)

    # dev
    dev_generator = ImageDataLoader(labels=pd.DataFrame(dev.iloc[:, 1:]),
                                      paths_to_images=pd.DataFrame(dev.iloc[:, 0]), paths_prefix=None,
                                      preprocessing_functions=preprocessing_functions,
                                      augment=False,
                                      augmentation_functions=None)

    dev_generator = torch.utils.data.DataLoader(dev_generator, batch_size=batch_size, shuffle=False,
                                                  num_workers=16, pin_memory=False)
    # test
    test_generator = ImageDataLoader(labels=pd.DataFrame(test.iloc[:, 1:]),
                                    paths_to_images=pd.DataFrame(test.iloc[:, 0]), paths_prefix=None,
                                    preprocessing_functions=preprocessing_functions,
                                    augment=False,
                                    augmentation_functions=None)

    test_generator = torch.utils.data.DataLoader(test_generator, batch_size=batch_size, shuffle=False,
                                                num_workers=8, pin_memory=False)

    return train_generator, dev_generator, test_generator

def train_model(train, dev, test, epochs:int, class_weights:Optional=None, loss_function:str="Crossentropy"):
    # metaparams
    metaparams = {
        "optimizer": "Adam",  # SGD, RMSprop, AdamW
        "learning_rate_max": 0.001,  # up to 0.0001
        "learning_rate_min": 0.00001,  # up to 0.000001
        "lr_scheduller": "Cyclic",  # "reduceLRonPlateau"
        "annealing_period": 6,
        "epochs": 30,
        "batch_size": 64,
        "augmentation_rate": 0.05,
        "architecture": "HRNet_modified_additional_convs",
        "dataset": "IWSDS2023",
        "num_classes": 5
    }
    # initialization of Weights and Biases
    wandb.init(project="VGGFace2_FtF_training", config=metaparams)
    config = wandb.config


    # create model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    HRNet = load_HRNet_model(device="cuda" if torch.cuda.is_available() else "cpu",
                             path_to_weights = "/work/home/dsu/simpleHRNet/pose_hrnet_w32_256x192.pth")
    model = modified_HRNet(HRNet, num_classes=config.num_classes)
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
          'batch_size: %d, augmentation_rate: %f, architecture: %s, '
          'dataset: %s, num_classes: %d' % (config.optimizer, config.learning_rate_max, config.lr_scheduller,
                                            config.annealing_period, config.epochs, config.batch_size,
                                            config.augmentation_rate, config.architecture, config.dataset,
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
    print("Start...")
    sweep_config = {
        'name': "HRNet_f2f_Focal_loss",
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
            'augmentation_rate':{
                'values': [0.05]
            }
        }
    }
    BATCH_SIZE=128


    # load data
    train, dev, test = load_NoXi_data_all_languages(train_labels_as_categories=False,
                                                    dev_labels_as_categories=False,
                                                    test_labels_as_categories=False)
    # compute class weights
    class_weights = compute_class_weight(class_weight='balanced',
                                         classes=np.unique(np.argmax(train.iloc[:, 1:].values, axis=1, keepdims=True)),
                                         y=np.argmax(train.iloc[:, 1:].values, axis=1, keepdims=True).flatten())

    train_gen, dev_gen, test_gen = get_data_loaders_from_data(train, dev, test, augment=True, augment_prob=sweep_config['parameters']['augmentation_rate']['values'][0],
                                                              batch_size=BATCH_SIZE,
                                                              preprocessing_functions=[T.Resize(size=(256, 256)),
                                                                                       convert_image_to_float_and_scale,
                                                                                       T.Normalize(
                                                                                           mean=[0.485, 0.456, 0.406],
                                                                                           std=[0.229, 0.224, 0.225])
                                                                                       ])  # From HRNet

    print("Wandb with Focal_loss")
    sweep_id = wandb.sweep(sweep_config, project='VGGFace2_FtF_training')
    wandb.agent(sweep_id, function=lambda: train_model(train_gen, dev_gen, test_gen, epochs=100,
                loss_function="Focal_loss", class_weights=class_weights),
                count=20,
                project='VGGFace2_FtF_training')
    gc.collect()


if __name__ == "__main__":
    main()
