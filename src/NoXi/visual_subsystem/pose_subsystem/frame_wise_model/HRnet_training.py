from collections import Callable

import torch
from torchinfo import summary

from src.NoXi.visual_subsystem.pose_subsystem.frame_wise_model.HRNet import load_HRNet_model, modified_HRNet


def train_step(model:torch.nn.Module, train_generator:torch.utils.data.DataLoader,
               optimizer:torch.optim.Optimizer, criterion:torch.nn.Module,
               device:torch.device, print_step:int=100):

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
        if i % print_step == (print_step-1):  # print every 10 mini-batches
            print("Mini-batch: %i, loss: %.10f" % (i, running_loss / print_step))
            running_loss = 0.0







if __name__ == "__main__":
    torch.cuda_available=False
    model = load_HRNet_model("cpu")
    a = model(torch.ones((1,3,256,256)))
    model = modified_HRNet(model)
    device = torch.device("cpu")
    model.to(device)
    print("DEVICE:", device)
    summary(model, input_size=(1, 3, 256, 256))

