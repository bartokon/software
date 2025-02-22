import torch
from torch.nn import Linear, RNN, MaxPool1d, LSTM, Transformer, TransformerEncoder, TransformerEncoderLayer, TransformerDecoder, TransformerDecoderLayer
from torch_geometric.loader import DataLoader
from torch_geometric.nn import MLP, PointTransformerConv, PointNetConv, global_mean_pool, fps, radius, global_max_pool, knn, knn_graph
from torch_geometric.utils import scatter

from torch_geometric.typing import WITH_TORCH_CLUSTER
from dataset_torchstudio_small import ModelNet40_n3

if not WITH_TORCH_CLUSTER:
    quit("This example requires 'torch-cluster'")

class FullNet(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder_layer = TransformerEncoderLayer(
            d_model=2048,
            dim_feedforward=2048,
            nhead=4,
            batch_first=True,
            dropout=0.5
        )

        self.decoder_layer = TransformerDecoderLayer(
            d_model=2048,
            dim_feedforward=2048,
            nhead=4,
            batch_first=True,
            dropout=0.5
        )

        self.decoder = TransformerDecoder(
            decoder_layer=self.decoder_layer,
            num_layers=4,
            norm=torch.nn.LayerNorm(2048)
        )

        self.encoder = TransformerEncoder(
            encoder_layer=self.encoder_layer,
            num_layers=4,
            norm=torch.nn.LayerNorm(2048)
        )

        self.feat = Transformer(
            d_model=2048,
            nhead=4,
            dropout=0.5,
            batch_first=True,
            custom_encoder=self.encoder,
            custom_decoder=self.decoder
        )

        self.mlp_0 = MLP(
            channel_list=None,
            in_channels=512,
            hidden_channels=512,
            num_layers=4,
            out_channels=64,
            dropout=0.5,
            norm="LayerNorm",
            act="relu"
        )

        self.norm = torch.nn.LayerNorm(2048)

        self.max_pool_0 = MaxPool1d(
            kernel_size=4
        )

        self.max_pool_1 = MaxPool1d(
            kernel_size=64
        )

    def forward(self, data_0):
        x = data_0.view(-1, 3, 2048)
        x = self.norm(x)
        xy = self.feat(x, x)
        xy = self.max_pool_0(xy)
        xy = self.mlp_0(xy)
        xy = self.max_pool_1(xy)
        #xy = self.mlp_1(xy.view(data_0.shape[:-1]))
        xy = torch.squeeze(xy, dim=2)
        return xy
        xy = xy.view(-1)
        xy = self.mlp_0(xy)
        #y = torch.flatten(xy)
        #xy = torch.mean(xy, dim=0)
        #xy = xy.view(-1, 3)
        return xy

class FullNet_voxel(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.mlp_0 = MLP(
            channel_list=None,
            in_channels=512,
            hidden_channels=512,
            num_layers=4,
            out_channels=64,
            dropout=0.5,
            norm="LayerNorm",
            act="relu"
        )

        self.norm = torch.nn.LayerNorm(2048)

        self.max_pool_0 = MaxPool1d(
            kernel_size=4
        )

        self.max_pool_1 = MaxPool1d(
            kernel_size=64
        )

    def forward(self, data_0):
        x = data_0.view(-1, 3, 2048)
        x = self.norm(x)
        xy = self.feat(x, x)
        xy = self.max_pool_0(xy)
        xy = self.mlp_0(xy)
        xy = self.max_pool_1(xy)
        #xy = self.mlp_1(xy.view(data_0.shape[:-1]))
        xy = torch.squeeze(xy, dim=2)
        return xy
        xy = xy.view(-1)
        xy = self.mlp_0(xy)
        #y = torch.flatten(xy)
        #xy = torch.mean(xy, dim=0)
        #xy = xy.view(-1, 3)
        return xy

def m1():
    model = FullNet()
    #print(model)
    #exit(-1)
    model.to('cuda')
    for input_0, output_0 in train_dataloader:
        input_0.to(device='cuda')
        output_0.to(device='cuda')
        logits = model(
            input_0
        )
        print(logits.shape)
        print(logits)
        break

if __name__=="__main__":
    train_dataset = ModelNet40_n3(root="data_test")
    train_dataloader = DataLoader(train_dataset, batch_size = 10)
    m1()