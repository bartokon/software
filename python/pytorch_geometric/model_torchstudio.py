import os
os.environ["HSA_OVERRIDE_GFX_VERSION"] = "10.3.0"

import torch
from torch import Tensor
from torch.nn import Sequential, Linear, ReLU, LeakyReLU
from torch_geometric.loader import DataLoader
from torch_geometric.nn import MessagePassing, global_max_pool
from dataset_torchstudio import ModelNet40_n3
from torch.nn.functional import relu, leaky_relu

class PointNetLayer(MessagePassing):
    def __init__(self, in_channels: int, out_channels: int):
        # Message passing with "max" aggregation.
        super().__init__(aggr='max')

        # Initialization of the MLP:
        # Here, the number of input features correspond to the hidden
        # node dimensionality plus point dimensionality (=3).
        self.mlp = Sequential(
            Linear(in_channels + 3, out_channels),
            LeakyReLU(),
            Linear(out_channels, out_channels),
        )

    def forward(self,
        h: Tensor,
        pos: Tensor,
        edge_index: Tensor,
    ) -> Tensor:
        # Start propagating messages.
        return self.propagate(edge_index, h=h, pos=pos)

    def message(self,
        h_j: Tensor,
        pos_j: Tensor,
        pos_i: Tensor,
    ) -> Tensor:
        # h_j: The features of neighbors as shape [num_edges, in_channels]
        # pos_j: The position of neighbors as shape [num_edges, 3]
        # pos_i: The central node position as shape [num_edges, 3]

        edge_feat = torch.cat([h_j, pos_j - pos_i], dim=-1)
        return self.mlp(edge_feat)

class PointNet(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = PointNetLayer(3, 32)
        self.conv2 = PointNetLayer(32, 32)
        self.classifier = Linear(32, 3)

    def forward(self,
        pos: Tensor,
        edge_index: Tensor,
        batch: Tensor,
    ) -> Tensor:

        # Perform two-layers of message passing:
        h = self.conv1(h=pos, pos=pos, edge_index=edge_index)
        h = h.relu()
        h = self.conv2(h=h, pos=pos, edge_index=edge_index)
        h = h.relu()
        print(h)
        # Global Pooling:
        h = global_max_pool(h, batch)  # [num_examples, hidden_channels]

        # Classifier:
        return self.classifier(h)

class SingleNet(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = PointNetLayer(3, 32)
        self.conv2 = PointNetLayer(32, 64)
        self.fc0 = Linear(64, 64)

    def forward(self,
        pos: Tensor,
        edge_index: Tensor,
        batch: Tensor,
    ) -> Tensor:

        # Perform two-layers of message passing:
        h = leaky_relu(self.conv1(h=pos, pos=pos, edge_index=edge_index))
        h = leaky_relu(self.conv2(h=h, pos=pos, edge_index=edge_index))

        # Global Pooling:
        h = global_max_pool(h, batch)  # [num_examples, hidden_channels]

        # Classifier:
        return self.fc0(h)

class FullNet(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.feat = SingleNet()
        self.fc0 = Linear(128, 256)
        self.fc1 = Linear(256, 256)
        self.fc2 = Linear(256, 128)
        self.fc3 = Linear(128, 3)

    def forward(self,
        pos_0,
        edge_index_0,
        batch_0,
        pos_1,
        edge_index_1,
        batch_1
    ):
        x = self.feat(pos_0, edge_index_0, batch_0)
        y = self.feat(pos_1, edge_index_1, batch_1)
        xy = torch.cat((x, y), dim = 1)
        #print(xy)
        #print(xy.shape)
        #exit(-1)
        xy = leaky_relu(self.fc0(xy))
        xy = leaky_relu(self.fc1(xy))
        xy = leaky_relu(self.fc2(xy))
        xy = self.fc3(xy)

        return xy

def m0():
    model = PointNet()
    print(model)
    model.to('cuda')
    for input_0, input_1, output_0 in train_dataloader:
        input_0.to(device='cuda')
        print(input_0)
        logits = model(
            pos = input_0.pos,
            edge_index = input_0.edge_index,
            batch = input_0.batch
        )
        print(logits.shape)
        print(logits)
        break

def m1():
    model = FullNet()
    print(model)
    model.to('cuda')
    for input_0, input_1, output_0 in train_dataloader:
        input_0.to(device='cuda')
        input_1.to(device='cuda')
        output_0.to(device='cuda')
        logits = model(
            pos_0 = input_0.pos,
            edge_index_0 = input_0.edge_index,
            batch_0 = input_0.batch,
            pos_1 = input_1.pos,
            edge_index_1 = input_1.edge_index,
            batch_1 = input_1.batch
        )
        print(logits.shape)
        print(logits)
        break

if __name__=="__main__":
    train_dataset = ModelNet40_n3(root="data")
    train_dataloader = DataLoader(train_dataset, batch_size = 10)
    #m0()
    m1()