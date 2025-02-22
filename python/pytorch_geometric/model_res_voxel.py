import torch
from torch.nn import Sequential, ModuleList
from torch.nn import ReLU, ReLU6, Mish, Sigmoid, Conv3d, MaxPool3d, MaxPool1d, Bilinear, Linear, LayerNorm, Flatten, Dropout3d, Dropout, AvgPool3d, AvgPool2d, AvgPool1d
from torch.nn import LazyLinear, LazyConv3d, LazyInstanceNorm3d, LazyBatchNorm3d, LazyBatchNorm1d, AdaptiveMaxPool3d
from torch_geometric.loader import DataLoader
from torch_geometric.nn import MLP

from torch_geometric.typing import WITH_TORCH_CLUSTER
from dataset_torchstudio_small_voxel import ModelNet40_n3
import math

if not WITH_TORCH_CLUSTER:
    quit("This example requires 'torch-cluster'")

class MyLayer(torch.nn.Module):
    def __init__(self, channels = 1, kernel = 3):
        super().__init__()
        self.layer = ModuleList([
            LazyConv3d(out_channels = channels, kernel_size = kernel, stride = 1, padding='same'),
            Mish(inplace = True),
            LazyConv3d(out_channels = channels, kernel_size = kernel, stride = 1, padding='same'),
            Mish(inplace = True),
        ])

    def forward(self, x):
        xl = x
        for l in self.layer:
            xl = l(xl)
        return torch.add(x, xl)

class MyBottleneck(torch.nn.Module):
    def __init__(self, channels = 1):
        super().__init__()
        self.layer = Sequential(
            LazyConv3d(out_channels = channels, kernel_size = 3, stride = 1, padding = "same"),
            Mish(True),
        )

    def forward(self, x):
        return self.layer(x)


class MyNorm(torch.nn.Module):

    def __init__(self, C_O = 2):
        super().__init__()
        self.layer = Sequential(
            AdaptiveMaxPool3d(output_size = C_O),
            LayerNorm((C_O, C_O, C_O), elementwise_affine = False, bias = False),
        )

    def forward(self, x):
        return self.layer(x)

class MyCompund(torch.nn.Module):
    def __init__(self, C_I = 1, C_O = 1, kernel = 3):
        super().__init__()
        self.layer = Sequential(
            MyLayer(C_I, kernel),
            MyNorm(C_O)
        )

    def forward(self, x):
        return self.layer(x)

class FullNet(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.shared_layers_0 = Sequential(
            MyCompund(C_I = 64, C_O = 16, kernel = 3),
            MyCompund(C_I = 64, C_O = 8, kernel = 3),
            MyCompund(C_I = 64, C_O = 4, kernel = 3),
            MyCompund(C_I = 64, C_O = 2, kernel = 3),
            #MyCompund(C_I = 64, C_O = 1, kernel = 3),
        )

        self.shared_layers_1 = Sequential(
            MyCompund(C_I = 8, C_O = 16, kernel = 9),
            MyCompund(C_I = 8, C_O = 8, kernel = 9),
            MyCompund(C_I = 8, C_O = 4, kernel = 9),
            MyCompund(C_I = 8, C_O = 2, kernel = 9),
            #MyCompund(C_I = 8, C_O = 1, kernel = 9),
        )

        self.common_layers = Sequential(
            Flatten(start_dim = 1),
            LazyLinear(out_features = 2**12),
            Mish(inplace = True),
            Dropout(p = 0.5, inplace = False),
            LazyLinear(out_features = 2**12),
            Mish(inplace = True),
            Dropout(p = 0.5, inplace = False),
            LazyLinear(out_features = 3),
        )

    def shared(self, data):
        x_0 = torch.unsqueeze(data, dim=1)
        x_1 = torch.unsqueeze(data, dim=1)
        #print(x.shape)
        for layer in self.shared_layers_0:
            x_0 = layer(x_0)
            #print(x.shape)
        for layer in self.shared_layers_1:
            x_1 = layer(x_1)
        return torch.cat((x_0, x_1), dim = 1)

    def common(self, data_0, data_1):
        #print(data_0.shape)
        #x = torch.cat((data_0, data_1), dim=1)
        x = torch.subtract(data_0, data_1)
        #print(x.shape)
        x = torch.transpose(x, 1, 4)
        #print(x.shape)
        for layer in self.common_layers:
            x = layer(x)
        return x

    def forward(self, data_0):
        x = data_0[:, 0]
        y = data_0[:, 1]
        x = self.shared(x)
        y = self.shared(y)
        #print(x.shape)
        xy = self.common(x, y)
        #xy = torch.squeeze(xy, dim=(1, 2, 3))
        return xy

def m1():
    model = FullNet()
    print(model)
    model.to('cuda')
    #model_ = torch.nn.DataParallel(model)
    for input_0, output_0, raw_0 in train_dataloader:
        logits = model(
            input_0
        )
        print(logits.shape)
        #print(logits)
        break

if __name__=="__main__":
    train_dataset = ModelNet40_n3(root="data_test")
    train_dataloader = DataLoader(train_dataset, batch_size = 4)
    m1()