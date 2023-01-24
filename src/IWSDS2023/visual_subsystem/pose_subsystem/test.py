import os

from functools import partial


import pandas as pd
import numpy as np
import torch
import torchvision.transforms as T
from torchsummary import summary

from pytorch_utils.generators.ImageDataGenerator import ImageDataLoader
from pytorch_utils.generators.pytorch_augmentations import pad_image_random_factor, grayscale_image, \
    collor_jitter_image_random, gaussian_blur_image_random, random_perspective_image, random_rotation_image, \
    random_crop_image, random_posterize_image, random_adjust_sharpness_image, random_equalize_image, \
    random_horizontal_flip_image, random_vertical_flip_image

class Conv2dModel(torch.nn.Module):
    activation_functions_mapping = {'relu': torch.nn.ReLU,
                                    'sigmoid': torch.nn.Sigmoid,
                                    'tanh': torch.nn.Tanh,
                                    'softmax': torch.nn.Softmax,
                                    'linear': torch.nn.Linear
                                    }

    def __init__(self, input_shape:int):
        super(Conv2dModel, self).__init__()
        self.input_shape = input_shape
        # build the model
        self._build_model()


    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


    def _build_model(self):
        self.layers=torch.nn.ModuleList()
        # 1 block
        self.layers.append(torch.nn.Conv2d(in_channels=3, out_channels=128, kernel_size=(5,5), padding="same"))
        self.layers.append(self.activation_functions_mapping["relu"]())
        self.layers.append(torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(5,5), padding="same"))
        self.layers.append(self.activation_functions_mapping["relu"]())
        self.layers.append(torch.nn.MaxPool2d(kernel_size=(2,2), stride=(2,2)))

        # 2 block
        self.layers.append(torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(4, 4), padding="same"))
        self.layers.append(self.activation_functions_mapping["relu"]())
        self.layers.append(torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(4, 4), padding="same"))
        self.layers.append(self.activation_functions_mapping["relu"]())
        self.layers.append(torch.nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)))

        # 3 block
        self.layers.append(torch.nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), padding="same"))
        self.layers.append(self.activation_functions_mapping["relu"]())
        self.layers.append(torch.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), padding="same"))
        self.layers.append(self.activation_functions_mapping["relu"]())
        self.layers.append(torch.nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)))

        # 4 block
        self.layers.append(torch.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), padding="same"))
        self.layers.append(self.activation_functions_mapping["relu"]())
        self.layers.append(torch.nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)))


        # global pooling, flattering and dense
        self.layers.append(torch.nn.AvgPool2d(kernel_size=(8,8)))
        self.layers.append(torch.nn.Flatten(1,-1))
        self.layers.append(torch.nn.Linear(in_features=256, out_features=64))
        self.layers.append(self.activation_functions_mapping["relu"]())
        self.layers.append(torch.nn.Linear(in_features=64, out_features=10178))
        #self.layers.append(self.activation_functions_mapping["softmax"](dim=-1))





def load_data():
    path_to_data = r"C:\Users\Professional\Desktop\CelebA\100k\100k"
    path_to_labels=r"C:\Users\Professional\Desktop\CelebA\identity_CelebA.txt"
    labels=pd.read_csv(path_to_labels,sep=" ",header=None,names=["frame","label"])
    prefix=r"E:\Databases\CelebA\100k\100k"
    data=pd.DataFrame(labels['frame'])
    print(data.shape)
    # clean data to delete frames that do not exist
    data['frame']=data['frame'].apply(lambda x:x if os.path.exists(os.path.join(prefix, x)) else None)
    data = data.dropna()
    labels = labels[labels.set_index("frame").index.isin(data.set_index('frame').index)]
    labels = labels.drop(columns=['frame'])
    print(data.shape)
    return data, labels


if __name__=="__main__":
    prefix = r"C:\Users\Professional\Desktop\CelebA\100k\100k"
    paths, labels = load_data()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    augmentation_functions = {
        pad_image_random_factor: 0.05,
        grayscale_image: 0.05,
        partial(collor_jitter_image_random, brightness=0.5, hue=0.3, contrast=0.3, saturation=0.3): 0.05,
        partial(gaussian_blur_image_random, kernel_size=(5, 9), sigma=(0.1, 5)): 0.05,
        random_perspective_image: 0.05,
        random_rotation_image: 0.05,
        partial(random_crop_image, cropping_factor_limits=(0.7, 0.9)): 0.05,
        random_posterize_image: 0.05,
        partial(random_adjust_sharpness_image, sharpness_factor_limits=(0.1, 3)): 0.05,
        random_equalize_image: 0.05,
        random_horizontal_flip_image: 0.05,
        random_vertical_flip_image: 0.05
    }

    image_generator=ImageDataLoader(labels=labels, paths_to_images=paths, paths_prefix=prefix,
                                    preprocessing_functions=[T.Resize(size=(128,128))],
                                    augment=True,
                                    augmentation_functions=augmentation_functions)

    data_loader = torch.utils.data.DataLoader(image_generator, batch_size=48, shuffle=True,
                                              num_workers=8, pin_memory=False)

    model = Conv2dModel((3,128,128)).to(device)
    print('model structure:')
    print(model)
    print('-----------------------------------')
    print('model summary:')
    print(summary(model, input_size=(3,128,128)))

    # training process
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(10):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(data_loader, 0):
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
            if i % 100 == 99:  # print every 10 mini-batches
                print("Epoch: %i, mini-batch: %i, loss: %.10f" % (epoch, i, running_loss / 100.))
                running_loss = 0.0

    print('Finished Training')