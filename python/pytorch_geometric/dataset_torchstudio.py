import os
os.environ["HSA_OVERRIDE_GFX_VERSION"] = "10.3.0"

from os import listdir
from os.path import isfile, join
from copy import deepcopy

from tqdm import tqdm
from p_tqdm import p_umap
from random import uniform
from numpy import linspace

import torch
from torch.utils.data import Dataset, DataLoader
from torch_geometric.transforms import SamplePoints, KNNGraph, Compose
from torch_geometric.io import read_off

from rotate import Rotate
from functools import partial

class ModelNet40_n3(Dataset):
    def __init__(self, root = "data"):
        super(ModelNet40_n3, self).__init__()
        self.root = root
        self.raw_dir = root + "/raw"
        self.processed_dir = root + "/processed"
        self.points = 1024
        self.min_rot = 20
        self.x_rot = 45
        self.y_rot = 45
        self.z_rot = 45
        self.x_pts = 1
        self.y_pts = 1
        self.z_pts = 2
        self.process()

    @property
    def raw_file_names(self):
        file_names = [f for f in listdir(self.raw_dir) if isfile(join(self.raw_dir, f))]
        return file_names

    @property
    def processed_file_names(self):
        file_names_stock = [f for f in listdir(self.raw_dir) if isfile(join(self.raw_dir, f))]
        file_names = []
        for f in file_names_stock:
            for i in linspace(0, self.x_rot, self.x_pts, dtype=int):
                for j in linspace(0, self.y_rot, self.y_pts, dtype=int):
                    for k in linspace(0, self.z_rot, self.z_pts, dtype=int):
                        file_names.append(f[:-3]+ f"x{i}y{j}z{k}" + ".pt")
        return file_names

    def download(self):
        pass

    def rotate_and_save(self, degs, name, data, base_transform):
        i, j, k = degs
        if (not isfile(self.processed_dir + "/" + name[:-3] + f"x{i}y{j}z{k}" + ".pt")):
            rotate_transform = Compose([
                Rotate(degrees=degs[0], axis=0),
                Rotate(degrees=degs[1], axis=1),
                Rotate(degrees=degs[2], axis=2),
                base_transform
            ])
            b = base_transform(data)
            b.pos -= b.pos.min()
            b.pos /= b.pos.max()
            r = rotate_transform(data)
            r.pos -= r.pos.min()
            r.pos /= r.pos.max()
            torch.save(
                [torch.cat((b.pos, r.pos)), torch.tensor(degs, dtype=torch.float32)],
                self.processed_dir + "/" + name[:-3] + f"x{i}y{j}z{k}" + ".pt"
            )

    def process(self):
        base_transform = Compose([
            SamplePoints(num = self.points, include_normals = True),
            KNNGraph(k=6)
        ])

        for name in self.raw_file_names:
            data = read_off(self.raw_dir + "/" + name)
            pool = p_umap(
                partial(self.rotate_and_save, name=name, data=data, base_transform=base_transform),
                [[i, j, k] for i in linspace(self.min_rot, self.x_rot, self.x_pts, dtype=int, endpoint=True)
                for j in linspace(self.min_rot, self.y_rot, self.y_pts, dtype=int, endpoint=True)
                for k in linspace(self.min_rot, self.z_rot, self.z_pts, dtype=int, endpoint=True)]
            )

        #for name in self.raw_file_names:
            #data = read_off(self.raw_dir + "/" + name)
            #for i in linspace(0, self.x_rot, self.x_pts, dtype=int):
                #for j in linspace(0, self.y_rot, self.y_pts, dtype=int):
                    #for k in linspace(0, self.z_rot, self.z_pts, dtype=int):
                        #degs = [i, j, k]
                        #if (not isfile(self.processed_dir + "/" + name[:-3] + f"x{i}y{j}z{k}" + ".pt")):
                            #rotate_transform = Compose([
                                #Rotate(degrees=degs[0], axis=0),
                                #Rotate(degrees=degs[1], axis=1),
                                #Rotate(degrees=degs[2], axis=2),
                                #base_transform
                            #])
                            #torch.save(
                                #[base_transform(data), rotate_transform(data), torch.tensor(degs, dtype=torch.float32)],
                                #self.processed_dir + "/" + name[:-3] + f"x{i}y{j}z{k}" + ".pt"
                            #)
        #pool = p_umap(
            #self.mp,
            #range(0, 45)
        #)
        print("MP process done.")

    def __len__(self):
        return len(self.processed_file_names)

    def __getitem__(self, index):
        data = torch.load(self.processed_dir + "/" + self.processed_file_names[index])
        return data[0].to(device='cuda'), data[1].to(device='cuda')

if __name__ == "__main__":
    dataset_test = ModelNet40_n3(root="data_train")
    dataset_test = ModelNet40_n3(root="data_test")
    data = dataset_test[0]
    print(data)
    print("/********************************************/")
