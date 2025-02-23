import torch
import matplotlib.pyplot as plt

def draw_voxel(voxels: list[int]):
    def get_xs_ys_cs(voxel: list[int]):
        xs = []
        ys = []
        zs = []
        cs = []
        for x in range(voxel.shape[0]):
            for y in range(voxel.shape[1]):
                for z in range(voxel.shape[2]):
                    if (voxels[x,y,z] != 0):
                        xs.append(x)
                        ys.append(y)
                        zs.append(z)
                        cs.append(voxel[x, y, z])
        return xs, ys, zs, cs

    # Do the plotting in a single call.
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    xs, ys, zs, cs = get_xs_ys_cs(voxels)
    cs = torch.tensor(cs) / max(cs)
    ax.scatter(
        xs,
        ys,
        zs,
        c = cs,
        #alpha = cs,
        marker = "s",
        s = 64
    )
    plt.show(block=False)

def draw_voxel_pair(voxels_0: list[int], voxels_1: list[int], epoch):
    def get_xs_ys_cs(voxel: list[int]):
        xs = []
        ys = []
        zs = []
        cs = []
        for x in range(voxel.shape[0]):
            for y in range(voxel.shape[1]):
                for z in range(voxel.shape[2]):
                    if (voxel[x,y,z] != 0):
                        xs.append(x)
                        ys.append(y)
                        zs.append(z)
                        cs.append(voxel[x, y, z])
        return xs, ys, zs, cs

    # Do the plotting in a single call.
    fig = plt.figure(figsize=(11.69,8.27))
    ax = fig.add_subplot(projection='3d')
    xs, ys, zs, cs = get_xs_ys_cs(voxels_0)
    ax.scatter(
        xs,
        ys,
        zs,
        marker="s",
        s=64,
        alpha=0.5
    )
    xs, ys, zs, cs = get_xs_ys_cs(voxels_1)
    ax.scatter(
        xs,
        ys,
        zs,
        marker="s",
        s=64,
        alpha=1
    )
    fig.savefig(f"logs/latest_{epoch}.pdf")