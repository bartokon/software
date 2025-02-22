import os
os.environ["HSA_OVERRIDE_GFX_VERSION"] = "10.3.0"
import torch
import torch.nn as nn
from torch.nn.functional import relu

from torch.utils.data import DataLoader
from dataset_pytorch import ModelNet40_n3

class ConvNet(nn.Module):
    def __init__(self, scale = 1):
        super(ConvNet, self).__init__()
        self.conv1 = torch.nn.Conv1d(3, 256 * scale, 2)
        self.conv2 = torch.nn.Conv1d(256 * scale, 512 * scale, 4)
        self.conv3 = torch.nn.Conv1d(512 * scale, 1024 * scale, 8)

        self.bn1 = nn.BatchNorm1d(256 * scale)
        self.bn2 = nn.BatchNorm1d(512 * scale)
        self.bn3 = nn.BatchNorm1d(1024 * scale)

        self.mx1 = nn.MaxPool1d(kernel_size = 2, stride = 1)

    def forward(self, x):
        x = (relu(self.bn1(self.conv1(x))))
        x = (relu(self.bn2(self.conv2(x))))
        x = self.mx1(relu(self.bn3(self.conv3(x))))
        return x

class FullNet(nn.Module):
    def __init__(self, scale = 1):
        super(FullNet, self).__init__()
        self.ConvNetScale = 4

        #self.feat = ConvNet(scale = self.ConvNetScale)

        self.feat = nn.Transformer(d_model=1024, nhead=8)

        self.fc1 = nn.Linear(2048, 512 * scale)
        self.fc2 = nn.Linear(512 * scale, 256 * scale)
        self.fc3 = nn.Linear(256 * scale, 128 * scale)
        self.fc4 = nn.Linear(128 * scale, 3)

        self.bn1 = nn.BatchNorm1d(3)
        self.bn2 = nn.BatchNorm1d(3)
        self.bn3 = nn.BatchNorm1d(3)
        self.bn4 = nn.BatchNorm1d(3)

        self.dr1 = nn.Dropout(p=0.3)
        self.dr2 = nn.Dropout(p=0.3)
        self.dr3 = nn.Dropout(p=0.3)
        self.dr4 = nn.Dropout(p=0.3)


    def forward(self, x, y):
        #x = self.feat(x, y)
        #y = self.feat(y, x)
        xy = torch.cat((x, y), dim = 2)
        #print(x.shape)
        xy = relu(self.dr1(self.bn1(self.fc1(xy))))
        xy = relu(self.dr2(self.bn2(self.fc2(xy))))
        xy = relu(self.dr3(self.bn3(self.fc3(xy))))
        #xy = relu(self.dr4(self.bn4(self.fc4(xy))))
        xy = self.fc4(xy)
        #print(xy.shape)
        xy = torch.max(xy, dim = 1)[0]
        return xy

def m0():
    model = FullNet()
    model.to('cuda')
    for pc_0, pc_1, y in train_dataloader:
        logits = model(pc_0, pc_1)
        print(logits.shape)
        break

if __name__=="__main__":
    train_dataset = ModelNet40_n3(root="data_train")
    test_dataset = ModelNet40_n3(root="data_test")
    train_dataloader = DataLoader(train_dataset, batch_size = 10)
    test_dataloader = DataLoader(test_dataset, batch_size = 10)
    m0()