import torch
from torch.nn import Sequential, ReLU, Conv3d, MaxPool3d, MaxPool1d, Bilinear, Linear, LayerNorm, Flatten, Dropout3d, Dropout
from torch.nn import LazyLinear, LazyConv3d, LazyInstanceNorm3d, LazyBatchNorm3d
from torch_geometric.loader import DataLoader
from torch_geometric.nn import MLP

from torch_geometric.typing import WITH_TORCH_CLUSTER
from dataset_torchstudio_small_voxel import ModelNet40_n3

if not WITH_TORCH_CLUSTER:
    quit("This example requires 'torch-cluster'")

class FullNet(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.shared_stage_0 = Sequential(
            LazyConv3d(
                out_channels=2,
                kernel_size=3,
                stride=2,
            ),
            ReLU(inplace=True),
        )

        self.shared_stage_1 = Sequential(
            #MaxPool3d(kernel_size=2),
            #LazyInstanceNorm3d(),
            LazyBatchNorm3d(),
            Dropout3d(0.2),
            LazyConv3d(
                out_channels=4,
                kernel_size=3,
                stride=2,
            ),
            Dropout3d(0.2),
            ReLU(inplace=True),
            LazyConv3d(
                out_channels=4,
                kernel_size=3
            ),
            ReLU(inplace=True),
        )

        self.shared_stage_2 = Sequential(
            #MaxPool3d(kernel_size=2),
            #LazyInstanceNorm3d(),
            LazyBatchNorm3d(),
            Dropout3d(0.2),
            LazyConv3d(
                out_channels=8,
                kernel_size=3,
                stride=2,
            ),
            Dropout3d(0.2),
            ReLU(inplace=True),
            LazyConv3d(
                out_channels=8,
                kernel_size=3,
            ),
            ReLU(inplace=True),
        )

        self.shared_stage_3 = Sequential(
            #MaxPool3d(kernel_size=2),
            #LazyInstanceNorm3d(),
            LazyBatchNorm3d(),
            Dropout3d(0.2),
            LazyConv3d(
                out_channels=16,
                kernel_size=3,
                stride=2,
            ),
            Dropout3d(0.2),
            ReLU(inplace=True),
            LazyConv3d(
                out_channels=16,
                kernel_size=3,
            ),
            ReLU(inplace=True),
        )

        self.shared_stage_4 = Sequential(
            #MaxPool3d(kernel_size=2),
            #LazyInstanceNorm3d(),
            LazyBatchNorm3d(),
            Dropout3d(0.2),
            LazyConv3d(
                out_channels=32,
                kernel_size=2,
            ),
            Dropout3d(0.2),
            ReLU(inplace=True),
            LazyConv3d(
                out_channels=32,
                kernel_size=2,
            ),
            ReLU(inplace=True),
        )

        self.shared_layers = Sequential(
            self.shared_stage_0,
            self.shared_stage_1,
            self.shared_stage_2,
            self.shared_stage_3,
            self.shared_stage_4,
        )

        self.common_stage_0 = Sequential(
            LazyBatchNorm3d(),
            Dropout3d(0.2),
            Flatten(start_dim=1, end_dim=-1),
        )

        self.common_stage_1 = Sequential(
            LazyLinear(out_features=16),
            ReLU(),
            LazyLinear(out_features=16),
            ReLU(),
            LazyLinear(out_features=3),
        )

        self.common_layers = Sequential(
            self.common_stage_0,
            self.common_stage_1,
        )

    def shared(self, data):
        x = torch.unsqueeze(data, dim=1)
        for layer in self.shared_layers:
            x = layer(x)
            #print(x.shape)
        return x

    def common(self, data_0, data_1):
        x = torch.cat((data_0, data_1), dim=1)
        for layer in self.common_layers:
            x = layer(x)
            #print(x.shape)
        return x

    def forward(self, data_0):
        x = data_0[:, 0]
        y = data_0[:, 1]
        x = self.shared(x)
        y = self.shared(y)
        xy = self.common(x, y)
        #xy = torch.squeeze(xy, dim=0)
        return xy

def m1():
    model = FullNet()
    print(model)
    model.to('cuda')
    for input_0, output_0, raw_0 in train_dataloader:
        input_0.to(device='cuda')
        output_0.to(device='cuda')
        logits = model(
            input_0
        )
        print(logits.shape)
        #print(logits)
        break

if __name__=="__main__":
    train_dataset = ModelNet40_n3(root="data_test")
    train_dataloader = DataLoader(train_dataset, batch_size = 64)
    m1()