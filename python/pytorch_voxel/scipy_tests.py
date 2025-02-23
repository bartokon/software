from os.path import exists
import numpy as np
from dataset_voxel import ModelNet40_aligned
from sklearn.ensemble import RandomForestClassifier
import torch
import pathlib

def prepare_numpy_array(dataset):
    dl = torch.utils.data.DataLoader(
        train_dataset,
        batch_size = len(dataset)
    )
    train_loader_array_x = next(iter(dl))[0].cpu().numpy().reshape((len(dataset), -1))
    train_loader_array_y = next(iter(dl))[1][1].cpu().numpy().reshape((len(dataset), -1))
    return train_loader_array_x, train_loader_array_y

if __name__=="__main__":
    classes = [f.name for f in pathlib.Path("dataset").iterdir() if f.is_dir()]
    classes = ['airplane', 'bed']
    train_dataset = ModelNet40_aligned(
        root = "dataset",
        classes = classes
    )
    reg = RandomForestClassifier(
        random_state = 0,
        verbose = True
    )

    x, y = prepare_numpy_array(train_dataset)
    reg.fit(x, y)
    test_dataset = ModelNet40_aligned(
        root = "dataset",
        classes = classes,
        train = False,
    )
    x, y = prepare_numpy_array(test_dataset)
    print(reg.score(x, y))
    print(reg.predict(x))

