import os

from SimpleHRNet import SimpleHRNet

os.environ["CUDA_VISIBLE_DEVICES"] = ""
import sys
from collections import OrderedDict

import torch
torch.cuda.is_available = lambda : False
from torchinfo import summary
import numpy as np



def load_HRNet_model(path_to_weights:str=r"C:\Users\Professional\PycharmProjects\simple-HRNet-master\pose_hrnet_w32_256x192.pth")->torch.nn.Module:
    model = SimpleHRNet(c=32, nof_joints=17,
                              checkpoint_path=path_to_weights, multiperson=False,
                              return_heatmaps=False, return_bounding_boxes=False,
                              device="cpu")
    model = model.model
    return model



class modified_HRNet(torch.nn.Module):

    def __init__(self, HRNet_model:torch.nn.Module, num_classes:int=4):
        super(modified_HRNet, self).__init__()
        self.HRNet = HRNet_model
        self.HRNet.final_layer = torch.nn.Identity()
        self.num_classes = num_classes
        self._build_additional_layers()

    def _build_additional_layers(self):
        """self.additional_layers = torch.nn.Sequential(
            OrderedDict([
                    # first block
                    ("conv1_add",torch.nn.Conv2d(17, 256, kernel_size=(3,3), stride=(1,1), padding="same")),
                     ("relu1_add",torch.nn.ReLU()),
                     ("maxpool1_add",torch.nn.MaxPool2d(kernel_size=2, stride=2)), # 64x64

                    # global average pooling
                     ("glovalpool_add",torch.nn.AdaptiveAvgPool2d(output_size=(1, 1))),
                    # Flatten
                     ("flatten1_add",torch.nn.Flatten()),
                    # dense layer
                     ("linear1_add",torch.nn.Linear(256, 256)),
                     ("relu2_add", torch.nn.ReLU()),
                    # last layer
                     ("lastlayer_add",torch.nn.Linear(256, self.num_classes))]
                )
        )"""
        """self.additional_layers = torch.nn.Sequential(
            OrderedDict([
                    # first block
                    ("conv1_add",torch.nn.Conv2d(17, 256, kernel_size=(3,3), stride=(1,1), padding="same")),
                     ("relu1_add",torch.nn.ReLU()),
                     ("maxpool1_add",torch.nn.MaxPool2d(kernel_size=2, stride=2)), # 64x64
                    ]
                )
        )
        """
        #self.additional_layers = torch.nn.Sequential(torch.nn.AdaptiveAvgPool2d(output_size=(1, 1)))

    def forward(self, x):
        x = self.HRNet(x)
        #x = self.additional_layers(x)
        return x



if __name__ == "__main__":
    model = load_HRNet_model()
    a = model(torch.ones((1,3,256,256)))
    model = modified_HRNet(model)
    device = torch.device("cpu")
    model.to(device)
    print("DEVICE:", device)
    #print(model)
    summary(model, input_size=(1, 3, 256, 256))



"""# second block
                torch.nn.Conv2d(256, 256, kernel_size=(3,3), stride=(1,1), padding="same"),
                torch.nn.ReLU(),
                torch.nn.MaxPool2d(kernel_size=2, stride=2), # 32x32
                # third block
                torch.nn.Conv2d(256, 256, kernel_size=(3,3), stride=(1,1), padding="same"),
                torch.nn.ReLU(),
                torch.nn.MaxPool2d(kernel_size=2, stride=2),  # 16x16
                # forth block
                torch.nn.Conv2d(256, 256, kernel_size=(3,3), stride=(1,1), padding="same"),
                torch.nn.ReLU(),
                torch.nn.MaxPool2d(kernel_size=2, stride=2),  # 8x8"""