import os
os.environ["HSA_OVERRIDE_GFX_VERSION"] = "10.3.0"

from os import listdir
from os.path import isfile, join
from copy import deepcopy

from tqdm import tqdm
from random import uniform

import torch
from torch_geometric.data import Dataset
from torch_geometric.transforms import SamplePoints, Compose, KNNGraph
from torch_geometric.io import read_off
from torch_geometric.explain import Explanation
from torch.nn.functional import normalize

from rotate import Rotate
from rotate import rotate
import multiprocessing
from p_tqdm import p_umap

class ModelNet40(Dataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(ModelNet40, self).__init__(root, transform, pre_transform)

    @property
    def raw_file_names(self):
        file_names = [f for f in listdir(self.raw_dir) if isfile(join(self.raw_dir, f))]
        return file_names

    @property
    def processed_file_names(self):
        file_names = [f for f in listdir(self.raw_dir) if isfile(join(self.raw_dir, f))]
        file_names = [f[:-3]+"pt" for f in file_names]
        return file_names

    def download(self):
        pass

    def mp(self, index):
        if (not isfile(self.processed_dir + "/" + self.raw_file_names[index][:-3] + "pt")):
            data = read_off(self.raw_paths[index])
            points = 1024
            degs = [uniform(0, 10), uniform(0, 10), uniform(0, 10)]

            base_transform = SamplePoints(num=points)
            rotate_transform = Compose([
                Rotate(degrees=degs[0], axis=0),
                Rotate(degrees=degs[1], axis=1),
                Rotate(degrees=degs[2], axis=2)
            ])

            data = base_transform(data)
            data.pos = data.pos - torch.mean(data.pos, dim=0) #CENTER PC

            data_cpy = deepcopy(data)
            knn = KNNGraph(k=6)
            data_cpy = knn(data)

            data_cpy = rotate_transform(data_cpy)
            knn = KNNGraph(k=6)
            data_cpy = knn(data_cpy)
            data_cpy.pos = data_cpy.pos - torch.mean(data_cpy.pos, dim=0) #CENTER PC

            data.pos = normalize(data.pos)
            data_cpy.pos = normalize(data_cpy.pos)

            data.pos = torch.cat((data.pos, data_cpy.pos), dim=0)
            data.pos = data.pos.transpose(0,1)
            data.y = torch.tensor(degs)
            torch.save(data, self.processed_dir + "/" + self.raw_file_names[index][:-3] + "pt")


    def process(self):
        #pool = p_umap(
            #self.mp,
            #range(0, len(self.raw_paths))
        #)
        pool = multiprocessing.Pool(processes=8)
        pool.map(
            self.mp,
            range(0, len(self.raw_paths))
        )
        pool.close()
        pool.join()
        print("MP process done.")

        #for index in tqdm(range(0, len(self.raw_paths))):
            #if (not isfile(self.processed_dir + "/" + self.raw_file_names[index][:-3] + "pt")):
                #data = read_off(self.raw_paths[index])
                #points = 1024
                #degs = [uniform(0, 180), uniform(0, 180), uniform(0, 180)]

                #base_transform = SamplePoints(num=points)
                #rotate_transform = Compose([
                    #Rotate(degrees=degs[0], axis=0),
                    #Rotate(degrees=degs[1], axis=1),
                    #Rotate(degrees=degs[2], axis=2)
                #])

                #data = base_transform(data)
                #data.pos = data.pos - torch.mean(data.pos, dim=0) #CENTER PC

                #data_cpy = deepcopy(data)
                #data_cpy = rotate_transform(data_cpy)
                #data_cpy.pos = data_cpy.pos - torch.mean(data_cpy.pos, dim=0) #CENTER PC

                #data.pos = normalize(data.pos)
                #data_cpy.pos = normalize(data_cpy.pos)

                #data.pos = torch.cat((data.pos, data_cpy.pos), dim=0)
                #data.pos = data.pos.transpose(0,1)
                #data.y = torch.tensor(degs)
                #torch.save(data, self.processed_dir + "/" + self.raw_file_names[index][:-3] + "pt")

    def len(self):
        return len(self.processed_file_names)

    def get(self, index):
        data = torch.load(self.processed_paths[index], map_location='cuda')
        return data


class ModelNet40graph(Dataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(ModelNet40graph, self).__init__(root, transform, pre_transform)

    @property
    def raw_file_names(self):
        file_names = [f for f in listdir(self.raw_dir) if isfile(join(self.raw_dir, f))]
        return file_names

    @property
    def processed_file_names(self):
        file_names = [f for f in listdir(self.raw_dir) if isfile(join(self.raw_dir, f))]
        file_names = [f[:-3]+"pt" for f in file_names]
        return file_names

    def download(self):
        pass

    def mp(self, index):
        if (not isfile(self.processed_dir + "/" + self.raw_file_names[index][:-3] + "pt")):
            points = 1024
            degs = [uniform(0, 89), uniform(0, 89), uniform(0, 89)]
            base_transform = SamplePoints(num=points)
            knn = KNNGraph(k=6)

            data = read_off(self.raw_paths[index])
            data.y = torch.tensor([0.0, 0.0, 0.0])
            minimum, _ = torch.min(data.pos, dim = 0)
            maximum, _ = torch.max(data.pos, dim = 0)
            data.pos = (data.pos - minimum) / (maximum - minimum)
            data.pos = data.pos * (1 - -1) + -1

            minimum = torch.min(data.pos)
            maximum = torch.max(data.pos)
            if(minimum < -1):
                print(f"Fail min {minimum}")
                exit(-1)
            if(maximum > 1):
                print(f"Fail max {maximum}")
                exit(-1)
#  data.pos = data.pos - torch.mean(data.pos, dim=0) #CENTER PC
            data = base_transform(data)
            data_cpy = deepcopy(data)
            data = knn(data)

            #data_cpy = rotate_transform(data_cpy)
            r = data_cpy.pos
            r = rotate(r, degs[0], axis=0, device='cpu')
            r = rotate(r, degs[1], axis=1, device='cpu')
            r = rotate(r, degs[2], axis=2, device='cpu')
            data_cpy.pos = r
#            data_cpy.pos = data_cpy.pos - torch.mean(data_cpy.pos, dim=0) #CENTER PC
            data_cpy = knn(data_cpy)
            data_cpy.y = torch.tensor(degs)

            torch.save([data, data_cpy], self.processed_dir + "/" + self.raw_file_names[index][:-3] + "pt")

    def process(self):
        #pool = p_umap(
            #self.mp,
            #range(0, len(self.raw_paths))
        #)
        pool = multiprocessing.Pool(processes=8)
        pool.map(
            self.mp,
            range(0, len(self.raw_paths))
        )
        pool.close()
        pool.join()
        #for i in range(0, len(self.raw_paths)):
            #self.mp(i)
        print("MP process done.")

    def len(self):
        return len(self.processed_file_names)

    def get(self, index):
        data = torch.load(self.processed_paths[index], map_location='cuda')
        #data = torch.load(self.processed_paths[index])
        return data

class ModelNet40graph_i(Dataset):
    def __init__(self, root, transform=None, pre_transform=None):
        self.i = 1
        super(ModelNet40graph_i, self).__init__(root, transform, pre_transform)

    @property
    def raw_file_names(self):
        file_names = [f for f in listdir(self.raw_dir) if isfile(join(self.raw_dir, f))]
        return file_names

    @property
    def processed_file_names(self):
        file_names = [f for f in listdir(self.raw_dir) if isfile(join(self.raw_dir, f))]
        file_names = [f[:-3]+f"_{i}.pt" for i in range(0, self.i) for f in file_names]
        return file_names

    def download(self):
        pass

    def mp(self, index):
        if (not isfile(self.processed_dir + "/" + self.raw_file_names[index][:-3] + "pt")):
            points = 1024
            base_transform = SamplePoints(num=points)
            knn = KNNGraph(k=6)

            data = read_off(self.raw_paths[index])
            data.pos = data.pos - torch.mean(data.pos, dim=0) #CENTER PC
            data.pos = normalize(data.pos)
            data = base_transform(data)
            data = knn(data)
            data.y = torch.tensor([0.0, 0.0, 0.0])

            for i in range(0, self.i):
                degs = [uniform(0, 89), uniform(0, 89), uniform(0, 89)]
                rotate_transform = Compose([
                    Rotate(degrees=degs[0], axis=0),
                    Rotate(degrees=degs[1], axis=1),
                    Rotate(degrees=degs[2], axis=2)
                ])
                data_cpy = deepcopy(data)
                data_cpy.pos = data_cpy.pos - torch.mean(data_cpy.pos, dim=0) #CENTER PC
                data_cpy.pos = normalize(data_cpy.pos)
                data_cpy = rotate_transform(data_cpy)
                data_cpy = base_transform(data_cpy)
                data_cpy = knn(data_cpy)
                data_cpy.y = torch.tensor(degs)

                torch.save([data, data_cpy], self.processed_dir + "/" + self.raw_file_names[index][:-3] + f"_{i}.pt")

    def process(self):
        #pool = p_umap(
            #self.mp,
            #range(0, len(self.raw_paths))
        #)
        pool = multiprocessing.Pool(processes=8)
        pool.map(
            self.mp,
            range(0, len(self.raw_paths))
        )
        pool.close()
        pool.join()
        print("MP process done.")

    def len(self):
        return len(self.processed_file_names)

    def get(self, index):
        data = torch.load(self.processed_paths[index], map_location='cuda')
        #data = torch.load(self.processed_paths[index])
        return data

class ModelNet40_n3(Dataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(ModelNet40_n3, self).__init__(root, transform, pre_transform)

    @property
    def raw_file_names(self):
        file_names = [f for f in listdir(self.raw_dir) if isfile(join(self.raw_dir, f))]
        return file_names

    @property
    def processed_file_names(self):
        file_names = [f for f in listdir(self.raw_dir) if isfile(join(self.raw_dir, f))]
        file_names = [f[:-3]+"pt" for f in file_names]
        return file_names

    def download(self):
        pass

    def mp(self, index):
        if (not isfile(self.processed_dir + "/" + self.raw_file_names[index][:-3] + "pt")):
            points = 1024
            degs = [uniform(0, 89), uniform(0, 89), uniform(0, 89)]
            base_transform = SamplePoints(num=points)

            data = read_off(self.raw_paths[index])
            data.y = torch.tensor(degs)
            minimum, _ = torch.min(data.pos, dim = 0)
            maximum, _ = torch.max(data.pos, dim = 0)
            data.pos = (data.pos - minimum) / (maximum - minimum)
            data.pos = data.pos * (1 - -1) + -1

            minimum = torch.min(data.pos)
            maximum = torch.max(data.pos)
            if(minimum < -1):
                print(f"Fail min {minimum}")
                exit(-1)
            if(maximum > 1):
                print(f"Fail max {maximum}")
                exit(-1)
            data = base_transform(data)
            data_cpy = deepcopy(data)

            r = data_cpy.pos
            r = rotate(r, degs[0], axis=0, device='cpu')
            r = rotate(r, degs[1], axis=1, device='cpu')
            r = rotate(r, degs[2], axis=2, device='cpu')
            data_cpy.pos = r

            data.pos = data.pos.transpose(1, 0)
            data_cpy.pos = data_cpy.pos.transpose(1, 0)
            data.pos = torch.cat((data.pos, data_cpy.pos), dim = 1)
            torch.save(data, self.processed_dir + "/" + self.raw_file_names[index][:-3] + "pt")

    def process(self):
        pool = multiprocessing.Pool(processes=8)
        pool.map(
            self.mp,
            range(0, len(self.raw_paths))
        )
        pool.close()
        pool.join()
        print("MP process done.")

    def len(self):
        return len(self.processed_file_names)

    def get(self, index):
        data = torch.load(self.processed_paths[index], map_location='cuda')
        #data = torch.load(self.processed_paths[index])
        return data

if __name__ == "__main__":
    dataset_train = ModelNet40_n3(root="data_train")
    print("/********************************************/")
    data = dataset_train[0]
    print(data)

    #g = torch_geometric.utils.convert.to_networkx(data)
    #nx.draw(g, node_size=0.1)
    #plt.show()
    #print(data.pos)
    #print(data_cpy.pos)