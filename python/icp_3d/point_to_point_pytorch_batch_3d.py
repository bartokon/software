import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
import point_to_point_utils_3d as utils

class myLayerRotation(nn.Module):
    def __init__(self):
        super().__init__()
        weight = torch.Tensor(3,3)
        self.weight = nn.Parameter(weight, requires_grad=True)
        torch.nn.init.zeros_(self.weight)

    def forward(self, x):
        a = torch.zeros_like(x)
        for i in range(len(x)):
            a[i] = self.weight @ x[i]
        return a

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

    scm = torch.jit.script(model)
    mse_loss = nn.MSELoss()

    #optimizer = optim.AdamW(scm.parameters(), lr=0.1, amsgrad=True)
    optimizer = optim.Adam(scm.parameters(), lr=0.1, amsgrad=True)

    with torch.jit.optimized_execution(True):
        for _ in range(1000):
            optimizer.zero_grad()
            model_output = scm.forward(pc_rotated)
            loss = mse_loss(pc_fixed, model_output)
            loss.backward()
            optimizer.step()
            print(f"Loss: {loss}")
            if (loss < 1e-10):
                break

    R = model[0].weight
    T = model[1].weight

    alpha1 = torch.atan2(R[2, 1], R[2, 2])
    alpha2 = torch.atan2(-R[2, 0], torch.sqrt(R[2, 1]**2 + R[2, 2]**2))
    alpha3 = torch.atan2(R[1, 0], R[0, 0])
    print(f"DEG: {utils.rad_to_deg(alpha1)}, {utils.rad_to_deg(alpha2)} {utils.rad_to_deg(alpha3)}")

    pc_corrected = torch.zeros_like(pc_fixed, requires_grad=False)
    for i in range(len(pc_fixed)):
        pc_corrected[i] = R @ pc_rotated[i] + T
    utils.plot_3d_point_clouds(
        (np.array(pc_fixed.detach().numpy()), "fixed"),
        (np.array(pc_corrected.detach().numpy()), "corrected"),
        (np.array(pc_rotated.detach().numpy()), "rotated")
    )