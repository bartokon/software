import os

import torch_geometric.nn
os.environ["HSA_OVERRIDE_GFX_VERSION"] = "10.3.0"

import torch
import torch_geometric
from torch.nn import ReLU, Dropout, LeakyReLU, MaxPool1d, Linear
from torch_geometric.nn import Sequential, GCNConv, BatchNorm
from torch_geometric.nn import EdgeConv, EdgeCNN

from torch_geometric.loader import DataLoader
from dataset import ModelNet40graph
import math

class GCN(torch.nn.Module):

    def __init__(self, in_channels = 3, hidden_channels = 64, out_channels = 3):
        super().__init__()

        self.s0 = Sequential(
            'x, edge_index',
            [
                (GCNConv(in_channels = in_channels, out_channels = hidden_channels, improved=True), 'x, edge_index -> x'),
                BatchNorm(hidden_channels),
                MaxPool1d(kernel_size=3, stride=1, padding=1),
                LeakyReLU(),
                Linear(hidden_channels, hidden_channels),
                LeakyReLU(),
                Dropout(),
            ]
        )
        self.s1 = Sequential(
            'x, edge_index',
            [
                (GCNConv(in_channels = hidden_channels, out_channels = hidden_channels, improved=True), 'x, edge_index -> x'),
                BatchNorm(hidden_channels),
                MaxPool1d(kernel_size=5, stride=1, padding=2),
                LeakyReLU(),
                Linear(hidden_channels, hidden_channels),
                LeakyReLU(),
                Dropout(),
            ]
        )

        self.s2 = Sequential(
            'x, edge_index',
            [
                (GCNConv(in_channels = hidden_channels, out_channels = out_channels, improved=True), 'x, edge_index -> x'),
                BatchNorm(out_channels),
                MaxPool1d(kernel_size=3, stride=1, padding=1),
                LeakyReLU(),
                Linear(out_channels, out_channels),
                LeakyReLU(),
                Dropout(),
            ]
        )

    def forward(self, x, edge_index):
        x = self.s0(x, edge_index)
        x = self.s1(x, edge_index)
        x = self.s2(x, edge_index)
        return x

class SIAMESE_GCN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.n = 1024
        self.s0 = GCN(in_channels = 3, hidden_channels=self.n, out_channels = 1024)

    def forward(self, x_x, x_edge_index, y_x, y_edge_index):
        x_x = self.s0(x_x, x_edge_index)
        y_x = self.s0(y_x, y_edge_index)
        x = torch.cat((x_x, y_x))
        return x

class PointNet_GCN(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.s0 = SIAMESE_GCN()
        self.s1 = torch.nn.Sequential(
            torch.nn.LazyConv1d(512, 1),
            torch.nn.LazyBatchNorm1d(),
            MaxPool1d(kernel_size=3, stride=1, padding=1),
            torch.nn.LeakyReLU(),
            torch.nn.Dropout(),
            torch.nn.LazyLinear(256),
            torch.nn.LazyBatchNorm1d(),
            MaxPool1d(kernel_size=3, stride=1, padding=1),
            torch.nn.LeakyReLU(),
            torch.nn.Dropout(),
            torch.nn.LazyLinear(3)
        )

    def forward(self, x_x, x_edge_index, y_x, y_edge_index):
        x = self.s0(x_x, x_edge_index, y_x, y_edge_index)
        x = self.s1(x)
        x = torch.mean(x, dim=0)
        return x

class PointNet_GCN_rotation(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.s0 = PointNet_GCN()

    def forward(self, x_x, x_edge_index, y_x, y_edge_index):
        x = self.s0(x_x, x_edge_index, y_x, y_edge_index)
        #print(x)
        sin, cos = math.sin(x[0]), math.cos(x[0])
        m = torch.tensor(([[1, 0, 0], [0, cos, sin], [0, -sin, cos]]), device='cuda', requires_grad=True)
        yy = y_x @ m
        sin, cos = math.sin(x[1]), math.cos(x[1])
        m = torch.tensor(([[cos, 0, -sin], [0, 1, 0], [sin, 0, cos]]), device='cuda', requires_grad=True)
        yy = yy @ m
        sin, cos = math.sin(x[2]), math.cos(x[2])
        m = torch.tensor(([[cos, sin, 0], [-sin, cos, 0], [0, 0, 1]]), device='cuda', requires_grad=True)
        yy = yy @ m
        return yy

def m0():
    model = GCN()
    model.to('cuda')
    for data in train_dataloader:
        logits = model(data[0].pos, data[0].edge_index)
        print(logits.shape)
        break

def m1():
    model = SIAMESE_GCN()
    model.to('cuda')
    for data in train_dataloader:
        logits = model(
            data[0].pos, data[0].edge_index,
            data[1].pos, data[1].edge_index,
        )
        print(logits.shape)
        break

def m2():
    model = PointNet_GCN()
    model.to('cuda')
    for data in train_dataloader:
        logits = model(
            data[0].pos, data[0].edge_index,
            data[1].pos, data[1].edge_index,
        )
        print(logits.shape)
        break

if __name__=="__main__":
    train_dataset = ModelNet40graph(root="data_train")
    test_dataset = ModelNet40graph(root="data_test")
    train_dataloader = DataLoader(train_dataset, batch_size = 1)
    test_dataloader = DataLoader(test_dataset, batch_size = 1)
    m0()
    m1()
    m2()