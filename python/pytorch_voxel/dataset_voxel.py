import os
os.environ["HSA_OVERRIDE_GFX_VERSION"] = "10.3.0"

import torch
from torch.utils.data import Dataset
from torch_geometric.transforms import Compose
from torch_geometric.io import read_off
from typing import List, Optional
from functools import cached_property
import pathlib
import numpy as np
from torch_geometric.transforms import SamplePoints
from p_tqdm import p_umap
from collections import Counter
import pandas as pd
import filecmp

class ModelNet40_aligned(Dataset):
    def __init__(self,
            root = "dataset",
            classes: List[str] = None,
            train: bool = True,
            download: bool = False,
            transform: Optional[callable] = None,
            target_transform: Optional[callable] = None
        ) -> None:
        super(ModelNet40_aligned, self).__init__()
        self.train = train
        self.root = root
        self.classes = classes
        if (self.classes == None):
            self.classes = [
                f.name for f in pathlib.Path(root).iterdir() if f.is_dir()
            ]
        if (self.classes == []):
            self.untar()
            self.classes = [
                f.name for f in pathlib.Path(root).iterdir() if f.is_dir()
            ]

        self.classes_onehot = torch.nn.functional.one_hot(
            torch.tensor(range(0, len(self.classes))),
            len(self.classes)
        ).bool()
        self.classes_lut = {
            class_name : (class_name, self.classes_onehot[index])
            for index, class_name in enumerate(self.classes)
        }
        if download:
            self.download()
        for class_name in self.classes:
            if (not os.path.isdir(f"{root}/{class_name}")):
                self.untar()
                self.clean()

    @cached_property
    def unprocessed_file_names(self):
        file_dirs = [
            f"{self.root}/{class_name}/train" if self.train else
            f"{self.root}/{class_name}/test"
            for class_name in self.classes
        ]
        full_file_names = [
            f"{directory}/{filename}"
            for directory in file_dirs
            for filename in os.listdir(directory)
            if filename[-4:] == ".off"
        ]
        return full_file_names

    @cached_property
    def processed_file_names(self):
        full_file_names = [
            file_name.replace("train", "processed_train", 1)
            if self.train else
            file_name.replace("test", "processed_test", 1)
            for file_name in self.unprocessed_file_names
        ]
        return full_file_names

    def download(self):
        os.makedirs(self.root, exist_ok = True)
        if (not os.path.isfile(f"{self.root}/modelnet40_manually_aligned.tar")):
            os.system(f"wget https://lmb.informatik.uni-freiburg.de/resources/datasets/ORION/modelnet40_manually_aligned.tar -P {self.root}")
        if (not os.path.isfile(f"{self.root}/modelnet40_manually_aligned.tar")):
            RuntimeError(".tar not found. Something went wrong!")

    def untar(self):
        if (not os.path.isfile(f"{self.root}/modelnet40_manually_aligned.tar")):
            RuntimeError(".tar not found. Use download = True option.")
        os.system(f"tar -xvzf {self.root}/modelnet40_manually_aligned.tar -C {self.root}")

    def clean(self):
        def get_path(f1, train = self.train):
            f1_c, f1_name = f1.rsplit("_", maxsplit = 1)
            return f"{self.root}/{f1_c}/train/{f1}.off", f"{self.root}/{f1_c}/test/{f1}.off"

        def compare_and_remove(f1, f2):
            if (os.path.isfile(f1) and os.path.isfile(f2)):
                if (filecmp.cmp(f1, f2, shallow=True)):
                    os.remove(f2)
                    print(f"Removed duplicate {f1} = {f2}")

        if (not os.path.isfile(f"main.zip")):
            os.system("wget https://github.com/oqton/M40-cleaning/archive/refs/heads/main.zip")
        if (not os.path.isdir("M40-cleaning-main")):
            os.system("unzip main.zip")

        #Remove duplicates
        duplicates_vfinal = pd.read_csv("M40-cleaning-main/duplicates_vfinal.csv", index_col = 0)
        for f1 in duplicates_vfinal['obj_id']:
            for f2 in duplicates_vfinal['obj_id']:
                if (f1 == f2):
                    continue
                f1_path_train, f1_path_test = get_path(f1)
                f2_path_train, f2_path_test = get_path(f2)
                compare_and_remove(f1_path_test, f1_path_train)
                compare_and_remove(f1_path_test, f2_path_test)
                compare_and_remove(f1_path_train, f2_path_train)
                compare_and_remove(f1_path_train, f2_path_test)

        relabel_vfinal = pd.read_csv("M40-cleaning-main/relabel_vfinal.csv")
        for obj, new_label in zip(relabel_vfinal['obj_id'], relabel_vfinal['new_label']):
            obj_path_train, obj_path_test = get_path(obj)
            if (new_label == 'discard' or True): #Remove all for now...
                if (os.path.isfile(obj_path_train)):
                    os.remove(obj_path_train)
                    print(f"Removed: {obj}")
                if (os.path.isfile(obj_path_test)):
                    os.remove(obj_path_test)
                    print(f"Removed: {obj}")

    def voxelize(self, points, vox_count: int = 1):
        vox_array = torch.zeros((vox_count, vox_count, vox_count), dtype=torch.float32)
        x_buckets = np.digitize(
            points[:, 0],
            np.linspace(
                min(points[:, 0]),
                max(points[:, 0]),
                num=vox_count,
                endpoint=False
            )
        ) - 1
        y_buckets = np.digitize(
            points[:, 1],
            np.linspace(
                min(points[:, 1]),
                max(points[:, 1]),
                num=vox_count,
                endpoint=False
            )
        ) - 1
        z_buckets = np.digitize(
            points[:, 2],
            np.linspace(
                min(points[:, 2]),
                max(points[:, 2]),
                num=vox_count,
                endpoint=False
            )
        ) - 1
        for a, b, c in zip(x_buckets, y_buckets, z_buckets):
            vox_array[a, b, c] += 1
        #Normalization
        #vox_array = (vox_array - torch.min(vox_array)) / (torch.max(vox_array) - torch.min(vox_array))
        #Standardization
        vox_array = (vox_array - torch.mean(vox_array)) / torch.std(vox_array)
        return vox_array

    def process(self, unprocessed_file_name, processed_file_name):
        try:
            data = read_off(unprocessed_file_name)
            base_transform = Compose([
                SamplePoints(num = 2**14, include_normals = False),
            ])
            data_transformed = base_transform(data)
            data_voxelized = self.voxelize(
                points = data_transformed.pos,
                vox_count = 32
            )
            os.makedirs(
                os.path.dirname(processed_file_name),
                exist_ok = True
            )
            p = pathlib.Path(processed_file_name)
            torch.save(
                [
                    data_voxelized,
                    self.classes_lut[p.parts[1]]
                ],
                processed_file_name
            )
        except Exception as e:
            print(f"Failed processing: {unprocessed_file_name}")
            print(f"Excepction: {e}")
            exit(1)

    def __len__(self):
        return len(self.processed_file_names)

    def __getitem__(self, index):
        if (not os.path.isfile(self.processed_file_names[index])):
            self.process(
                self.unprocessed_file_names[index],
                self.processed_file_names[index]
            )
        data, label = torch.load(
            self.processed_file_names[index],
            weights_only = True
        )
        return data.to(device='cuda'), label

    def _get_label(self, index):
        data, label = self.__getitem__(index)
        return label[0]

    def get_weights(self):
        if (not os.path.isfile(f"weights_{self.train}.pt")):
            #TODO: count files in folder...
            pool = p_umap(
                self._get_label,
                range(0, self.__len__())
            )
            weights_dict = dict(Counter(pool))
            torch.save(weights_dict, f"weights_{self.train}.pt")
        weights_dict = torch.load(f"weights_{self.train}.pt")
        weights_tensor = torch.tensor(
            [*weights_dict.values()],
            device = 'cuda'
        )
        weights_tensor = 1 / (weights_tensor / torch.max(weights_tensor))
        return weights_tensor

if __name__ == "__main__":
    classes = [f.name for f in pathlib.Path("dataset").iterdir() if f.is_dir()]
    if (1):
        dataset = ModelNet40_aligned(
            root = "dataset",
            classes = classes,
            train = True,
            transform = None,
            target_transform = None,
            download = True
        )
    else:
        dataset = ModelNet40_aligned(
            root = "dataset",
            classes = classes,
            train = False,
            transform = None,
            target_transform = None,
            download = True
        )
    dataset.clean()
    dataset.get_weights()