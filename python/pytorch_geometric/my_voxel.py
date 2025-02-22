import torch_geometric
import torch
from torch_geometric.loader import DataLoader
from dataset_torchstudio import ModelNet40_n3
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from torch_geometric.io import read_off
from torch_geometric.transforms import SamplePoints, KNNGraph, Compose
import numpy as np

from rotate import Rotate

def voxelize(pos, vox_count: int = 1):
    #POS is 0 - 1
    vox_array = torch.zeros((vox_count, vox_count, vox_count))
    #print(vox_array.shape)

    points = pos
    # Define bin edges
    bin_edges = np.linspace(0, 1, num=vox_count, endpoint=False)

    # Bucketize each dimension
    #print(points[:, 0])
    #print(points[:, 1])
    #print(points[:, 2])
    x_buckets = np.digitize(points[:, 0], bin_edges) - 1
    y_buckets = np.digitize(points[:, 1], bin_edges) - 1
    z_buckets = np.digitize(points[:, 2], bin_edges) - 1

    #print(bin_edges)
    #print(x_buckets)

    for a, b, c in zip(x_buckets, y_buckets, z_buckets):
        vox_array[a, b, c] += 1
    #print(vox_array.shape)
    vox_array = torch.tensor(vox_array, dtype=torch.float32)
    vox_array = vox_array / torch.max(vox_array)
    return vox_array

def rotate_and_sample_and_voxelize(data, degs: list[int] = [0, 0, 0], points: int = 1024, vox_count: int = 64):
    base_transform = Compose([
    SamplePoints(num = points, include_normals = True),
    KNNGraph(k=6)
    ])
    rotate_transform = Compose([
        Rotate(degrees=degs[0], axis=0),
        Rotate(degrees=degs[1], axis=1),
        Rotate(degrees=degs[2], axis=2),
        base_transform
    ])
    r = rotate_transform(data)
    r.pos -= r.pos.min()
    r.pos /= r.pos.max()
    return voxelize(r.pos, vox_count)

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
    ax.scatter(
        xs,
        ys,
        zs,
        marker="s",
        s=64
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
    #plt.show(block=False)

if __name__ == "__main__":
    data = read_off("data_train/raw/airplane_0002.off")
    draw_voxel(rotate_and_sample_and_voxelize(data, degs=[0, 0, 0], points=2**18, vox_count=2**7))
    draw_voxel(rotate_and_sample_and_voxelize(data, degs=[0, 0, 0], points=2**18, vox_count=2**7))
    plt.show()