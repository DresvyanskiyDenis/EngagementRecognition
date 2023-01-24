from SimpleHRNet import SimpleHRNet
from collections import OrderedDict
import torch


def load_HRNet_model(device:str="cuda",
    path_to_weights: str = r"C:\Users\Professional\PycharmProjects\simple-HRNet-master\pose_hrnet_w32_256x192.pth") -> torch.nn.Module:

    model = SimpleHRNet(c=32, nof_joints=17,
                        checkpoint_path=path_to_weights, multiperson=False,
                        return_heatmaps=False, return_bounding_boxes=False,
                        device=device)
    model = model.model
    return model


class modified_HRNet(torch.nn.Module):

    def __init__(self, HRNet_model: torch.nn.Module, num_classes: int = 4):
        super(modified_HRNet, self).__init__()
        self.HRNet = HRNet_model
        self.HRNet.final_layer = torch.nn.Identity()
        self._freeze_hrnet_parts()
        self.num_classes = num_classes
        self._build_additional_layers()

    def _freeze_hrnet_parts(self):
        for name, param in self.HRNet.named_parameters():
            param.requires_grad = False

    def _build_additional_layers(self):
        self.additional_layers = torch.nn.Sequential(
            OrderedDict([
                # block 1
                ("conv1_new", torch.nn.Conv2d(17, 128, kernel_size=(3, 3), stride=(1, 1), padding="same")),
                ("BatchNormalization1_new", torch.nn.BatchNorm2d(128)),
                ("relu1_new", torch.nn.ReLU()),
                ("maxpool1_new", torch.nn.MaxPool2d(kernel_size=2, stride=2)),  # 64x64
                # block 2
                ("conv2_new", torch.nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding="same")),
                ("BatchNormalization2_new", torch.nn.BatchNorm2d(128)),
                ("relu2_new", torch.nn.ReLU()),
                ("maxpool2_new", torch.nn.MaxPool2d(kernel_size=2, stride=2)),  # 32x32
                # block 3
                ("conv3_new", torch.nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding="same")),
                ("BatchNormalization3_new", torch.nn.BatchNorm2d(256)),
                ("relu3_new", torch.nn.ReLU()),
                ("maxpool3_new", torch.nn.MaxPool2d(kernel_size=2, stride=2)),  # 16x16
                # block 4
                ("conv4_new", torch.nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding="same")),
                ("BatchNormalization4_new", torch.nn.BatchNorm2d(256)),
                ("relu4_new", torch.nn.ReLU()),
                ("maxpool4_new", torch.nn.MaxPool2d(kernel_size=2, stride=2)),  # 8x8
                # Global avg pool
                ("globalpool_new", torch.nn.AdaptiveAvgPool2d(output_size=(1, 1))),
                # Flatten
                ("flatten_new", torch.nn.Flatten()),
                # dense layer
                ("linear1_new", torch.nn.Linear(256, 256)),
                ("relu2_new", torch.nn.ReLU()),
                # last layer
                ("lastlayer_new", torch.nn.Linear(256, self.num_classes))
            ]
            )
        )

    def forward(self, x):
        x = self.HRNet(x)
        x = self.additional_layers(x)
        return x