import os
os.environ["HSA_OVERRIDE_GFX_VERSION"] = "10.3.0"

from os import listdir
from os.path import isfile, join
from copy import deepcopy

from tqdm import tqdm
from random import uniform

import torch
from torch.utils.data import Dataset, DataLoader
from torch_geometric.transforms import SamplePoints
from torch_geometric.io import read_off

from rotate import rotate
import multiprocessing
from pointnet_model import STN3d, PointNetfeat, PointNetDenseRot

class ModelNet40_n3(Dataset):
    def __init__(self, root, max_rot = 45):
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

    def mp(self, index):
        if (not isfile(self.processed_dir + "/" + self.raw_file_names[index][:-3] + f"{self.max_rot}" + ".pt")):
            #print(self.raw_file_names[index])
            points = 1024
            degs = [uniform(0, self.max_rot), uniform(0, self.max_rot), uniform(0, self.max_rot)]
            base_transform = SamplePoints(num=points)

            data = read_off(self.raw_dir + "/" + self.raw_file_names[index])
            data = base_transform(data)

            data.pos = data.pos - torch.mean(data.pos, dim = 0)
            #minimum, _ = torch.min(data.pos, dim = 0)
            #maximum, _ = torch.max(data.pos, dim = 0)
            #data.pos = (data.pos - minimum) / (maximum - minimum)
            #data.pos = data.pos * (1 - -1) + -1
            #minimum = torch.min(data.pos)
            #maximum = torch.max(data.pos)
            #if(minimum < -1):
                #print(f"Fail min {minimum}")
                #exit(-1)
            #if(maximum > 1):
                #print(f"Fail max {maximum}")
                #exit(-1)


            r = deepcopy(data.pos)
            r = rotate(r, degs[0], axis=0, device='cpu')
            r = rotate(r, degs[1], axis=1, device='cpu')
            r = rotate(r, degs[2], axis=2, device='cpu')

            r = r - torch.mean(r, dim = 0)
            #minimum, _ = torch.min(r, dim = 0)
            #maximum, _ = torch.max(r, dim = 0)
            #r = (r - minimum) / (maximum - minimum)
            #r = r * (1 - -1) + -1
            #minimum = torch.min(r)
            #maximum = torch.max(r)
            #if(minimum < -1):
                #print(f"Fail min {minimum}")
                #exit(-1)
            #if(maximum > 1):
                #print(f"Fail max {maximum}")
                #exit(-1)

            data.pos = data.pos.transpose(1, 0)
            r = r.transpose(1, 0)
            torch.save([data.pos, r, torch.tensor(degs)], self.processed_dir + "/" + self.raw_file_names[index][:-3] + f"{self.max_rot}" + ".pt")

    def process(self):
        from p_tqdm import p_umap
        pool = p_umap(
            self.mp,
            range(0, len(self.raw_file_names))
        )
        #pool = multiprocessing.Pool(processes=8)
        #pool.map(
            #self.mp,
            #range(0, len(self.raw_file_names))
        #)
        #pool.close()
        #pool.join()
        print("MP process done.")

    def __len__(self):
        return len(self.processed_file_names)

    def __getitem__(self, index):
        data = torch.load(self.processed_dir + "/" + self.processed_file_names[index], map_location='cuda')
        return data[0], data[1], data[2]

if __name__ == "__main__":
    dataset_test = ModelNet40_n3(root="data_test")
    dataset_train = ModelNet40_n3(root="data_train")
    train_dataloader = DataLoader(dataset_train, batch_size = 10)
    test_dataloader = DataLoader(dataset_test, batch_size = 10)
    print("/********************************************/")
    #data = dataset_train[0]
    #print(data)

    #model = STN3d()
    #model.to('cuda')
    #for data in train_dataloader:
        #logits = model(data[0])
        #print(logits.shape)
        #break

    #model = PointNetfeat()
    #model.to('cuda')
    #for data in train_dataloader:
        #logits = model(data[0])
        #print(logits.shape)
        #break

    model = PointNetDenseRot()
    model.to('cuda')
    for pc_0, pc_1, y in train_dataloader:
        logits = model(pc_0, pc_1)
        print(logits.shape)
        break