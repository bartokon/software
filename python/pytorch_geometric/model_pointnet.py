import os.path as osp

import torch
import torch.nn.functional as F

import torch_geometric.transforms as T
from torch_geometric.datasets import ModelNet
from torch_geometric.loader import DataLoader
from torch_geometric.nn import MLP, PointNetConv, fps, global_max_pool, radius, knn, InstanceNorm
from torch_geometric.typing import WITH_TORCH_CLUSTER
from dataset_torchstudio import ModelNet40_n3
if not WITH_TORCH_CLUSTER:
    quit("This example requires 'torch-cluster'")

class SAModule(torch.nn.Module):
    def __init__(self, ratio, r, nn):
        super().__init__()
        self.ratio = ratio
        self.r = r
        self.conv = PointNetConv(nn, add_self_loops=False)

    def forward(self, x, pos, batch):
        idx = fps(pos, batch, ratio=self.ratio)
        row, col = radius(pos, pos[idx], self.r, batch, batch[idx], max_num_neighbors=64)
        #row, col = knn(pos, pos[idx], 32, batch, batch[idx], num_workers = 8)
        edge_index = torch.stack([col, row], dim=0)
        x_dst = None if x is None else x[idx]
        x = self.conv((x, x_dst), (pos, pos[idx]), edge_index)
        pos, batch = pos[idx], batch[idx]
        return x, pos, batch


class GlobalSAModule(torch.nn.Module):
    def __init__(self, nn):
        super().__init__()
        self.nn = nn

    def forward(self, x, pos, batch):
        x = self.nn(torch.cat([x, pos], dim=1))
        x = global_max_pool(x, batch)
        pos = pos.new_zeros((x.size(0), 3))
        batch = torch.arange(x.size(0), device=batch.device)
        return x, pos, batch


class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()

        # Input channels account for both `pos` and node features.
        self.sa1_module = SAModule(0.5, 0.2, MLP([3, 2048], norm="BatchNorm", act="leaky_relu"))
        self.sa2_module = SAModule(0.25, 0.4, MLP([2048 + 3, 256], norm=None, act="leaky_relu"))
        self.sa3_module = GlobalSAModule(MLP([256 + 3, 512], norm=None, act="leaky_relu"))

    def forward(self, data):
        sa0_out = (data.x, data.pos, data.batch)
        sa1_out = self.sa1_module(*sa0_out)
        sa2_out = self.sa2_module(*sa1_out)
        sa3_out = self.sa3_module(*sa2_out)
        x, pos, batch = sa3_out

        return x

class FullNet(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.feat = Net()
        self.mlp = MLP([1024, 256, 3], dropout=0.25, norm=None, act="leaky_relu")

    def forward(self, data_0, data_1):
        x = self.feat(data_0)
        y = self.feat(data_1)
        xy = torch.cat((x, y), dim = 1)
        return self.mlp(xy)

def m1():
    model = FullNet()
    #print(model)
    model.to('cuda')
    for input_0, input_1, output_0 in train_dataloader:
        input_0.to(device='cuda')
        input_1.to(device='cuda')
        output_0.to(device='cuda')
        logits = model(
            input_0,
            input_1
        )
        print(logits.shape)
        print(logits)
        break

if __name__=="__main__":
    train_dataset = ModelNet40_n3(root="data_test")
    train_dataloader = DataLoader(train_dataset, batch_size = 10)
    m1()