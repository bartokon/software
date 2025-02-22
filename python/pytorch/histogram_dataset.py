import os
os.environ["HSA_OVERRIDE_GFX_VERSION"] = "10.3.0"

from os import listdir
from os.path import isfile, join
from copy import deepcopy

from tqdm import tqdm
from random import uniform

import torch
from torch.utils.data import Dataset, DataLoader
from torch_geometric.transforms import SamplePoints, Compose
from torch_geometric.io import read_off

from rotate import Rotate
from rotate import rotate
import multiprocessing
from pointnet_model import STN3d, PointNetfeat, PointNetDenseRot
from scipy.spatial import KDTree
import point_angles
from point_information import point_information
import matplotlib.pyplot as plt
import point_to_plane_utils_3d as utils
import numpy as np

class ModelNet40_n3(Dataset):
    def __init__(self, root = "data_test", max_rot = 45):
        super(ModelNet40_n3, self).__init__()
        self.root = root
        self.raw_dir = root + "/raw"
        self.processed_dir = root + "/processed"
        self.max_rot = max_rot
        self.process()

    @property
    def raw_file_names(self):
        file_names = [f for f in listdir(self.raw_dir) if isfile(join(self.raw_dir, f))]
        return file_names

    @property
    def processed_file_names(self):
        file_names = [f for f in listdir(self.raw_dir) if isfile(join(self.raw_dir, f))]
        file_names = [f[:-3]+ f"{self.max_rot}" + ".pt" for f in file_names]
        return file_names

    def download(self):
        pass

    def tree_and_neighbors(self, points, normals):
        def po(i):
            i[0] *= i[0]
            i[1] *= i[1]
            i[2] *= i[2]
            return torch.sum(i)

        def angle_between_planes(a, b):
            cos_fi_top = torch.abs(a[0]*b[0] + a[1]*b[1] + a[2]*b[2])
            cos_fi_bot = torch.sqrt(po(a)) * torch.sqrt(po(b))
            if (cos_fi_bot == 0):
                cos_fi_bot = torch.tensor(1e-6)
            cos_fi = cos_fi_top / cos_fi_bot
            if (cos_fi > 1):
                cos_fi = torch.tensor(1)
            if (cos_fi < -1):
                cos_fi = torch.tensor(-1)
            fi = torch.arccos(cos_fi)
            return fi * 180 / torch.pi

        def angle_between_vectors(a, b):
            cos_fi_top = torch.dot(a, b)
            cos_fi_bot = torch.sqrt(po(a)) * torch.sqrt(po(b))
            if (cos_fi_bot == 0):
                cos_fi_bot = torch.tensor(1e-6)
            cos_fi = cos_fi_top / cos_fi_bot
            if (cos_fi > 1):
                #print(cos_fi)
                cos_fi = torch.tensor(1)
            if (cos_fi < -1):
                #print(cos_fi)
                cos_fi = torch.tensor(-1)
            fi = torch.arccos(cos_fi)
            return fi * 180 / torch.pi

        tree = KDTree(points)
        alphas = []
        bethas = []
        thetas = []
        for p in points:
            dd, ii = tree.query(p, k = 5)
            nn4p = []
            nn4n = []
            for i in ii:
                nn4p.append(points[i])
                nn4n.append(normals[i])
            alpha = []
            theta = []
            for n, p in zip(nn4n[1:], nn4p[1:]):
                alpha.append(angle_between_planes(nn4n[0], n))
                theta.append(angle_between_vectors(n, torch.cross((nn4p[0] - p), nn4n[0])))
                #print(f"{alpha=}\n{beta=}\n{theta=}")
            alphas.append(torch.stack(alpha))
            thetas.append(torch.stack(theta))

        alphas = torch.stack(alphas)
        thetas = torch.stack(thetas)
        return torch.cat((alphas, thetas), dim = 1)
        return torch.tensor([alphas, bethas, thetas])
            #exit(1)

    def mp(self, index):
        if (not isfile(self.processed_dir + "/" + self.raw_file_names[index][:-3] + f"{self.max_rot}" + ".pt")):
            #print(self.raw_file_names[index])
            points = 1024
            degs = [uniform(0, self.max_rot), uniform(0, self.max_rot), uniform(0, self.max_rot)]
            base_transform = SamplePoints(num = points, include_normals = True)
            rotate_transform = Compose([
                    Rotate(degrees=degs[0], axis=0),
                    Rotate(degrees=degs[1], axis=1),
                    Rotate(degrees=degs[2], axis=2)
                ])
            data = read_off(self.raw_dir + "/" + self.raw_file_names[index])

            data_transformed = base_transform(data)
            a_t_original = self.tree_and_neighbors(data_transformed.pos, data_transformed.normal)
            a_t_original = a_t_original.transpose(1, 0)

            data_transformed = rotate_transform(data)
            data_transformed = base_transform(data)
            a_t_rotated = self.tree_and_neighbors(data_transformed.pos, data_transformed.normal)
            a_t_rotated = a_t_rotated.transpose(1, 0)

            #print(a_t_original.shape)
            #print(a_t_rotated.shape)
            #print(torch.sort(a_t_original)[0])
            #print(torch.sort(a_t_rotated)[0])
            torch.save([a_t_original, a_t_rotated, torch.tensor(degs)], self.processed_dir + "/" + self.raw_file_names[index][:-3] + f"{self.max_rot}" + ".pt")

    def process(self):
        from p_tqdm import p_umap
        pool = p_umap(
            self.mp,
            range(0, len(self.raw_file_names))
        )
        print("MP process done.")

    def __len__(self):
        return len(self.processed_file_names)

    def __getitem__(self, index):
        #data = torch.load(self.processed_dir + "/" + self.processed_file_names[index], map_location='cuda')
        data = torch.load(self.processed_dir + "/" + self.processed_file_names[index])
        return data[0], data[1], data[2]

if __name__ == "__main__":
    dataset_test = ModelNet40_n3(root="data_test")
    dataset_train = ModelNet40_n3(root="data_train")
    train_dataloader = DataLoader(dataset_train, batch_size = 10)
    test_dataloader = DataLoader(dataset_test, batch_size = 10)
    print("/********************************************/")
