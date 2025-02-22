import torch
from torch_geometric.nn import global_max_pool, global_mean_pool
from torch import Tensor
from torch.nn import Sequential, Linear, Softmax, ReLU, LazyLinear, LazyConv3d, AvgPool1d, MaxPool1d
from torch_geometric.nn import MessagePassing, GCNConv

class PointNetLayer(MessagePassing):
    def __init__(self, in_channels: int, out_channels: int):
        # Message passing with "max" aggregation.
        super().__init__(aggr='max')

        # Initialization of the MLP:
        # Here, the number of input features correspond to the hidden
        # node dimensionality plus point dimensionality (=3).
        self.mlp = Sequential(
            LazyLinear(out_channels),
            ReLU(),
            LazyLinear(out_channels, out_channels),
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
        number = 2**8
        self.conv1_0 = PointNetLayer(3, number)
        self.conv1_1 = PointNetLayer(3, number)
        self.conv2_0 = GCNConv(2*number, int(number / 2))
        self.conv2_1 = GCNConv(2*number, int(number / 2))
        self.conv3 = LazyLinear(int(number / 2))
        self.conv4 = torch.nn.Flatten(0)
        self.classifier = LazyLinear(3)

    def forward(self,
        pos_0: Tensor,
        edge_index_0: Tensor,
        batch_0: Tensor,
        pos_1: Tensor,
        edge_index_1: Tensor,
        batch_1: Tensor,
    ) -> Tensor:

        # Perform two-layers of message passing:
        h_0 = self.conv1_0(h=pos_0, pos=pos_0, edge_index=edge_index_0)
        h_0 = h_0.relu()
        h_1 = self.conv1_1(h=pos_1, pos=pos_1, edge_index=edge_index_1)
        h_1 = h_1.relu()

        h = torch.cat((h_0, h_1), dim=1)
        h_0 = self.conv2_0(x=h, edge_index=edge_index_0)
        h_0 = h_0.relu()
        h_1 = self.conv2_1(x=h, edge_index=edge_index_1)
        h_1 = h_1.relu()

        h = torch.cat((h_0, h_1), dim=1)
        h = self.conv3(h)
        h = h.relu()
        h = self.conv4(h)
        h = h.relu(0)
        h = self.classifier(h)
        #print(h.shape)
        #exit(-1)
        return h

if __name__=="__main__":
    model = PointNet()
    print(model)