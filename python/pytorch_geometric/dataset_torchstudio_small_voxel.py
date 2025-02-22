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
import random
from my_voxel import voxelize, draw_voxel_pair

#TODO: make it as classification problem
#TODO: check classes distribution!

class ModelNet40_n3(Dataset):
    def __init__(self, root = "data"):
        super(ModelNet40_n3, self).__init__()
        self.root = root
        self.raw_dir = root + "/raw"
        self.processed_dir = root + "/processed"
        self.points = 2**18
        self.min_rot = 20
        self.x_rot = 45
        self.y_rot = 45
        self.z_rot = 45
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
            file_names.append(f[:-3] + f"processed" + ".pt")
        return file_names

    def download(self):
        pass

    def __len__(self):
        return len(self.processed_file_names)

    def process(self):
        pass

    def __getitem__(self, index):
        #print(index)
        if (not isfile(self.processed_dir + "/" + self.processed_file_names[index])):
            vox_count = 2**5
            data = read_off(self.raw_dir + "/" + self.raw_file_names[index])
            degs = [random.randrange(0, 90), random.randrange(0, 90), random.randrange(0, 90)]
            base_transform = Compose([
            SamplePoints(num = self.points, include_normals = True),
            KNNGraph(k=6)
            ])
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
                [torch.stack((voxelize(b.pos, vox_count), voxelize(r.pos, vox_count))), torch.tensor(degs, dtype=torch.float32), data],
                self.processed_dir + "/" + self.processed_file_names[index]
            )

        data = torch.load(self.processed_dir + "/" + self.processed_file_names[index])
        return data[0].to(device='cuda'), data[1].to(device='cuda'), data[2].to(device='cpu')
        #return data[0].to(device='cuda'), data[1].to(device='cuda')

def funer_0(i):
    data = dataset_train[i]

def funer_1(i):
    data = dataset_test[i]

def create_histogram(data):
  import matplotlib.pyplot as plt
  """
  Creates a histogram of the x, y, and z coordinates of a given torch array.

  Args:
    data: A torch array of shape (N, 3) where N is the number of elements and
          3 are the coordinates [x, y, z].

  Returns:
    None
  """

  # Extract x, y, and z coordinates
  x_coords = data[:, 0]
  y_coords = data[:, 1]
  z_coords = data[:, 2]

  # Create subplots
  fig, axs = plt.subplots(1, 3, figsize=(15, 5))

  # Plot histograms for each coordinate
  axs[0].hist(x_coords, bins=90, color='blue', alpha=0.7)
  axs[0].set_xlabel('X Coordinate')
  axs[0].set_ylabel('Frequency')
  axs[0].set_title('Histogram of X Coordinates')

  axs[1].hist(y_coords, bins=90, color='green', alpha=0.7)
  axs[1].set_xlabel('Y Coordinate')
  axs[1].set_ylabel('Frequency')
  axs[1].set_title('Histogram of Y Coordinates')

  axs[2].hist(z_coords, bins=90, color='red', alpha=0.7)
  axs[2].set_xlabel('Z Coordinate')
  axs[2].set_ylabel('Frequency')
  axs[2].set_title('Histogram of Z Coordinates')

  plt.tight_layout()
  #plt.show()

def create_histogram(data):
  import matplotlib.pyplot as plt

  # Create subplots
  fig, axs = plt.subplots(1, 2, figsize=(15, 5))

  # Plot histograms for each coordinate
  axs[0].hist(torch.mean(data, dim=0), bins=20, alpha=0.7)
  axs[0].set_xlabel('X Coordinate')
  axs[0].set_ylabel('Frequency')
  axs[0].set_title('Histogram of X Coordinates')

  plt.tight_layout()

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def create_scatter_plot(data):
  """
  Creates a scatter plot of the x, y, and z coordinates of a given torch array
  with color-coding based on local density.

  Args:
    data: A torch array of shape (N, 3) where N is the number of elements and
          3 are the coordinates [x, y, z].

  Returns:
    None
  """

  # Extract x, y, and z coordinates
  x_coords = data[:, 0]
  y_coords = data[:, 1]
  z_coords = data[:, 2]

  # Create a 3D scatter plot
  fig = plt.figure()
  ax = fig.add_subplot(111, projection='3d')

  # Calculate local density (you can adjust the bandwidth)
  from sklearn.neighbors import KernelDensity
  kde = KernelDensity(kernel='gaussian', bandwidth=0.5)  # Adjust bandwidth as needed
  kde.fit(data)
  log_density = kde.score_samples(data)
  print(log_density)
  # Color-code points based on density
  ax.scatter(x_coords, y_coords, z_coords, c=log_density, cmap='viridis')

  # Set labels and title
  ax.set_xlabel('X')
  ax.set_ylabel('Y')
  ax.set_zlabel('Z')
  ax.set_title('3D Scatter Plot with Density')

if __name__ == "__main__":
    dataset_train = ModelNet40_n3(root="data_train")
    dataset_test = ModelNet40_n3(root="data_test")
    #for i in tqdm(range(len(dataset_test))):
        #data = funer(dataset_test, [i])
    #res = p_umap(funer_0, range(len(dataset_train)))
    #res = p_umap(funer_1, range(len(dataset_test)))
    vals = torch.zeros((len(dataset_test), 3))
    for i in range(int(len(dataset_test))):
        vals[i] = dataset_test[i][1]
    create_histogram(vals)
    create_scatter_plot(vals)
    #print(data[0].shape)
    #print(data[1].shape)
    #print(data[2])
    print("/********************************************/")

    plt.show()