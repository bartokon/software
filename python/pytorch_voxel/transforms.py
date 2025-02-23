import torch
from torch_geometric.io import read_off
from torch_geometric.transforms import SamplePoints, KNNGraph, Compose
import numpy as np

from rotate import Rotate

@functional_transform('voxelize')
def voxelize(pos, vox_count: int = 1):
    vox_array = torch.zeros((vox_count, vox_count, vox_count))
    points = pos
    bin_edges = np.linspace(0, 1, num=vox_count, endpoint=False)
    x_buckets = np.digitize(points[:, 0], bin_edges) - 1
    y_buckets = np.digitize(points[:, 1], bin_edges) - 1
    z_buckets = np.digitize(points[:, 2], bin_edges) - 1
    for a, b, c in zip(x_buckets, y_buckets, z_buckets):
        vox_array[a, b, c] += 1
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