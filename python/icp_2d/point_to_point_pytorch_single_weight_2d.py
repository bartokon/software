import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
import point_to_point_utils_2d as utils

def torch_create_r(fi):
    R = torch.tensor((
        [torch.cos(fi), -torch.sin(fi)],
        [torch.sin(fi), torch.cos(fi)]
    ))
    return R

class myLayerRotation(nn.Module):
    def __init__(self):
        super().__init__()
        weight = torch.Tensor(1)
        self.weight = nn.Parameter(weight, requires_grad=True)
        torch.nn.init.zeros_(self.weight)

    def forward(self, x):
        x0 = torch.cos(self.weight) * x[0] - torch.sin(self.weight) * x[1]
        x1 = torch.sin(self.weight) * x[0] + torch.cos(self.weight) * x[1]
        return torch.cat((x0, x1), 0)

class myLayerTranslation(nn.Module):
    def __init__(self):
        super().__init__()
        weight = torch.Tensor(2)
        self.weight = nn.Parameter(weight, requires_grad=True)
        torch.nn.init.zeros_(self.weight)

    def forward(self, x):
        return x + self.weight

if __name__ == '__main__':
    pc_fixed = (torch.rand(1000, 2, requires_grad=False) - 0.5) * 10
    R = torch.tensor([1])
    T = torch.from_numpy(np.array([-5, 5])).float()

    pc_rotated = torch.zeros_like(pc_fixed, requires_grad=False)
    for i in range(len(pc_fixed)):
        pc_rotated[i] = torch_create_r(R) @ pc_fixed[i] + T

    model = nn.Sequential(
        myLayerRotation(),
        myLayerTranslation()
    )
    mse_loss = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    #print(model[0].weight)
    #exit()
    for _ in range(10):
        for i in range(len(pc_rotated)):
            optimizer.zero_grad()
            model_output = model.forward(pc_rotated[i])
            loss = mse_loss(pc_fixed[i], model_output)
            loss.backward()
            optimizer.step()
        print(f"Loss: {loss}")
        if (loss < 1e-10):
            break

    R = model[0].weight
    T = model[1].weight
    print(f"Degres: {utils.rad_to_deg(R)}")
    pc_corrected = torch.zeros_like(pc_fixed, requires_grad=False)
    for i in range(len(pc_fixed)):
        pc_corrected[i] = torch_create_r(R) @ pc_rotated[i] + T
    utils.plot_2d_point_clouds(
        (np.array(pc_fixed.detach().numpy()), "fixed", "*", 100),
        (np.array(pc_corrected.detach().numpy()), "corrected", "x", 100),
        (np.array(pc_rotated.detach().numpy()), "rotated", "o", 100)
    )