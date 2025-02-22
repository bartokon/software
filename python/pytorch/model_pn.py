import torch
from torch.nn import Linear, LazyLinear, Softmax
from torch.nn.functional import relu, leaky_relu, sigmoid

class MLP(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.layer = []
        self.layer.append(Linear(3, 1024))
        self.layer.append(Linear(1024, 1024))
        self.layer.append(Linear(1024, 1024))
        self.layer.append(Linear(1024, 512))
        self.layer.append(Linear(512, 3))
        self.layer = torch.nn.ModuleList(self.layer)

    def forward(self, x):
        for i in range(0, len(self.layer)):
            x = self.layer[i](x)
            x = leaky_relu(x)

        x, _ = torch.max(x, dim=0)
        return x

if __name__=="__main__":
    model = MLP()
    print(model)