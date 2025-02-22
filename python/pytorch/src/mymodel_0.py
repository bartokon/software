import os
os.environ["HSA_OVERRIDE_GFX_VERSION"] = "10.3.0"

import torch
from torch.nn import LazyLinear, Conv1d, BatchNorm1d, Flatten, LeakyReLU, Sequential, MaxPool1d, Dropout, LazyBatchNorm1d
from torch.nn.functional import relu, leaky_relu
from torch.autograd import Variable
import numpy as np

from torch_geometric.loader import DataLoader
from model_pn import MLP
from dataset import ModelNet40

class CONV_NORM(torch.nn.Module):

    def __init__(self):
        super().__init__()

        self.s0 = Sequential(
            Conv1d(3, 128, 1),
            LazyBatchNorm1d(),
            Dropout(),
            LeakyReLU()
        )

        self.s1 = Sequential(
            Conv1d(128, 128, 1),
            LazyBatchNorm1d(),
            Dropout(),
            LeakyReLU()
        )

        self.s2 = Sequential(
            Conv1d(128, 256, 1),
            LazyBatchNorm1d(),
            Dropout(),
            LeakyReLU()
        )

        self.s3 = Sequential(
            Conv1d(256, 512, 1),
            LazyBatchNorm1d(),
            Dropout(),
            LeakyReLU()
        )

    def forward(self, x):

        x = self.s0(x)
        x = self.s1(x)
        x = self.s2(x)
        x = self.s3(x)
        return x

class PointNetWeird(torch.nn.Module):

    def __init__(self):
        super().__init__()

        self.CN_0 = CONV_NORM()

        self.CN_2 = Sequential(
            Conv1d(1024, 128, 3),
            LazyBatchNorm1d(),
            Dropout(),
            LeakyReLU()
        )

        self.s0 = Sequential(
            LazyLinear(128),
            LeakyReLU(),
            LazyLinear(3)
        )

    def forward(self, x):
        xx, xy = torch.split(x, split_size_or_sections=int(x.shape[1]/2), dim=1)

        xx = self.CN_0(xx)
        xy = self.CN_0(xy)

        x = torch.cat((xx, xy), dim=0)
        x = self.CN_2(x)
        x = self.s0(x)
        x, _ = torch.max(x, dim=0)
        return x

def m0():
    model = CONV_NORM()
    model.to('cuda')
    for data in train_dataloader:
        logits = model(data.pos)
        print(logits.shape)
        break

def m1():
    model = PointNetWeird()
    model.to('cuda')

    for data in train_dataloader:
        logits = model(data.pos)
        #print(logits.shape)

    for data in test_dataloader:
        logits = model(data.pos)
        #print(logits.shape)
        #break

if __name__=="__main__":
    train_dataset = ModelNet40(root="data_train")
    test_dataset = ModelNet40(root="data_test")
    train_dataloader = DataLoader(train_dataset, batch_size = 1)
    test_dataloader = DataLoader(test_dataset, batch_size = 1)
    #m0()
    m1()