import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
import point_to_point_utils_3d as utils

class myLayerRotation(nn.Module):
    def __init__(self):
        super().__init__()
        weight = torch.Tensor(3,1) #RADIANS
        self.weight = nn.Parameter(weight, requires_grad=True)
        torch.nn.init.zeros_(self.weight)

    def forward(self, x):
        a = self.weight[0]
        b = self.weight[1]
        c = self.weight[2]

        r00 = torch.cos(c) * torch.cos(b)
        r01 = torch.sin(a) * torch.sin(b) * torch.cos(c) - torch.cos(a) * torch.sin(c)
        r02 = torch.cos(a) * torch.sin(b) * torch.cos(c) + torch.sin(a) * torch.sin(c)
        r10 = torch.cos(b) * torch.sin(c)
        r11 = torch.sin(a) * torch.sin(b) * torch.sin(c) + torch.cos(a) * torch.cos(c)
        r12 = torch.cos(a) * torch.sin(b) * torch.sin(c) - torch.sin(a) * torch.cos(c)
        r20 = -torch.sin(b)
        r21 = torch.sin(a) * torch.cos(b)
        r22 = torch.cos(a) * torch.cos(b)

        x_0 = r00 * x[0] + r01 * x[1] + r02 * x[2]
        x_1 = r10 * x[0] + r11 * x[1] + r12 * x[2]
        x_2 = r20 * x[0] + r21 * x[1] + r22 * x[2]

        return torch.cat((x_0, x_1, x_2), 0)

class myLayerTranslation(nn.Module):
    def __init__(self):
        super().__init__()
        weight = torch.Tensor(3)
        self.weight = nn.Parameter(weight, requires_grad=True)
        torch.nn.init.zeros_(self.weight)

    def forward(self, x):
        return x + self.weight

if __name__ == '__main__':
    pc_fixed = (torch.rand(1000, 3, requires_grad=False) - 0.5) * 10
    R = torch.from_numpy(utils.euler_angles_to_rotation_matrix(np.array([utils.deg_to_rad(30), utils.deg_to_rad(40), utils.deg_to_rad(20)]))).float()
    T = torch.from_numpy(np.array([5, -5, 20])).float()

    pc_rotated = torch.zeros_like(pc_fixed, requires_grad=False)
    for i in range(len(pc_fixed)):
        pc_rotated[i] = R @ pc_fixed[i] + T

    model = nn.Sequential(
        myLayerRotation(),
        myLayerTranslation()
    )
    mse_loss = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    for _ in range(100):
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

    alpha1 = R[0][0]
    alpha2 = R[1][0]
    alpha3 = R[2][0]
    print(f"DEG: {utils.rad_to_deg(alpha1)}, {utils.rad_to_deg(alpha2)} {utils.rad_to_deg(alpha3)}")

    pc_corrected = torch.zeros_like(pc_fixed, requires_grad=False)
    with torch.no_grad():
        R = torch.from_numpy(utils.euler_angles_to_rotation_matrix([alpha1, alpha2, alpha3]))
        for i in range(len(pc_fixed)):
            pc_corrected[i] = R @ pc_rotated[i] + T
    utils.plot_3d_point_clouds(
        (np.array(pc_fixed.detach().numpy()), "fixed"),
        (np.array(pc_corrected.detach().numpy()), "corrected"),
        (np.array(pc_rotated.detach().numpy()), "rotated")
    )