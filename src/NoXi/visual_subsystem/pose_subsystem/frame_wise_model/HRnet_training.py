import sys
sys.path.extend(["/work/home/dsu/datatools/"])
sys.path.extend(["/work/home/dsu/engagement_recognition_project_server/"])
sys.path.extend(["/work/home/dsu/simpleHRNet/"])

from collections import Callable
from functools import partial
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import torchvision.transforms as T
from sklearn.metrics import recall_score, precision_score, f1_score
from sklearn.utils import compute_class_weight
from torchinfo import summary

from pytorch_utils.callbacks import TorchEarlyStopping, TorchMetricEvaluator
from pytorch_utils.generators.ImageDataGenerator import ImageDataLoader
from pytorch_utils.generators.pytorch_augmentations import pad_image_random_factor, grayscale_image, \
    collor_jitter_image_random, gaussian_blur_image_random, random_perspective_image, random_rotation_image, \
    random_crop_image, random_posterize_image, random_adjust_sharpness_image, random_equalize_image, \
    random_horizontal_flip_image, random_vertical_flip_image
from pytorch_utils.losses import FocalLoss
from src.NoXi.visual_subsystem.pose_subsystem.frame_wise_model.HRNet import load_HRNet_model, modified_HRNet
from src.NoXi.visual_subsystem.pose_subsystem.frame_wise_model.utils import load_NoXi_data_all_languages, \
    convert_image_to_float_and_scale


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
        #print(outputs.shape)
        #print(labels.shape)
        #print("------------")
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % print_step == (print_step-1):  # print every 100 mini-batches
            print("Mini-batch: %i, loss: %.10f" % (i, running_loss / print_step))
            running_loss = 0.0


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
    dev_generator = ImageDataLoader(labels=pd.DataFrame(dev.iloc[:, 1]),
                                      paths_to_images=pd.DataFrame(dev.iloc[:, 0]), paths_prefix=None,
                                      preprocessing_functions=preprocessing_functions,
                                      augment=False,
                                      augmentation_functions=None)

    dev_generator = torch.utils.data.DataLoader(dev_generator, batch_size=batch_size, shuffle=False,
                                                  num_workers=16, pin_memory=False)
    # test
    test_generator = ImageDataLoader(labels=pd.DataFrame(test.iloc[:, 1]),
                                    paths_to_images=pd.DataFrame(test.iloc[:, 0]), paths_prefix=None,
                                    preprocessing_functions=preprocessing_functions,
                                    augment=False,
                                    augmentation_functions=None)

    test_generator = torch.utils.data.DataLoader(test_generator, batch_size=batch_size, shuffle=False,
                                                num_workers=8, pin_memory=False)

    return train_generator, dev_generator, test_generator

def train_model(train, dev, test, epochs:int, class_weights:Optional=None):
    # create model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    HRNet = load_HRNet_model(device="cuda" if torch.cuda.is_available() else "cpu",
                             path_to_weights = "/work/home/dsu/simpleHRNet/pose_hrnet_w32_256x192.pth")
    model = modified_HRNet(HRNet, num_classes=5)
    model.to(device)
    summary(model, input_size=(32, 3, 256, 256))
    # Select optimizer
    lr = 0.001
    optimizers = {'Adam': torch.optim.Adam,
                  'SGD': torch.optim.SGD,
                  'RMSprop': torch.optim.RMSprop,
                  'AdamW': torch.optim.AdamW}

    optimizer = optimizers['Adam'](model.parameters(), lr=lr)
    # select loss function
    class_weights = torch.from_numpy(class_weights).float()
    class_weights = class_weights.to(device)
    criterions = {'Cross_entropy': torch.nn.CrossEntropyLoss(weight=class_weights),
                   'Focal_loss': FocalLoss(alpha=class_weights, gamma=2)}
    criterion = criterions['Cross_entropy']
    # Select lr scheduller
    min_lr=0.00001
    lr_schedullers = {
        'Cyclic':torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=6, eta_min=min_lr),
        'ReduceLRonPlateau':torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode = 'max', patience = 10),
    }
    lr_scheduller = lr_schedullers['Cyclic']
    # callbacks
    val_metrics = {
        'val_recall': partial(recall_score, average='macro'),
        'val_precision': partial(precision_score, average='macro'),
        'val_f1_score:': partial(f1_score, average='macro')
    }
    early_stopping_callback = TorchEarlyStopping(verbose= True, patience = 10, save_path = "")
    metric_evaluator = TorchMetricEvaluator(generator = dev,
                 model=model,
                 metrics=val_metrics,
                 device=device,
                 need_argmax=True,
                 need_softmax=True,
                 loss_func=torch.nn.CrossEntropyLoss(weight=class_weights))

    # go through epochs
    for epoch in range(epochs):
        # train model one epoch
        train_step(model=model, train_generator=train, optimizer=optimizer, criterion=criterion,
                   device=device, print_step = 10)
        # evaluate model on dev set
        with torch.no_grad():
            dev_results = metric_evaluator()
            print("Epoch: %i, dev results:"% epoch)
            for metric_name, metric_value in dev_results.items():
                print("%s: %.4f" % (metric_name, metric_value))
            # check early stopping
            early_stopping_result = early_stopping_callback(dev_results['val_recall'], model)
        # update lr
        lr_scheduller.step()
        if early_stopping_result:
            break

def main():
    # load data
    BATCH_SIZE = 64
    train, dev, test = load_NoXi_data_all_languages()
    train = train.iloc[:1000,:]
    # compute class weights
    class_weights = compute_class_weight(class_weight='balanced',
                                         classes=np.unique(np.argmax(train.iloc[:, 1:].values, axis=1, keepdims=True)),
                                         y=np.argmax(train.iloc[:, 1:].values, axis=1, keepdims=True).flatten())

    train_gen, dev_gen, test_gen = get_data_loaders_from_data(train, dev, test, augment=True, augment_prob=0.05, batch_size=BATCH_SIZE,
                                                              preprocessing_functions=[T.Resize(size=(256,256)),
                                                                                       convert_image_to_float_and_scale,
                                                                                       T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                                                                       ]) # From HRNet
    train_model(train_gen, dev_gen, test_gen, epochs=100, class_weights=class_weights)


if __name__ == "__main__":
    main()
