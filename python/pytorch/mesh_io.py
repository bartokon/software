import os.path

import pyrender
import trimesh
import numpy
from torch_geometric.transforms import SamplePoints

def if_exists(path: str = ""):
    if (not os.path.exists(path)):
        print(f"File: {path}, does not exist!")
        exit(-1)
    return path

def get_model(path: str = ""):
    model = trimesh.load(if_exists(path), file_type='off')
    return model

def get_mesh(path: str = ""):
    model = trimesh.load(if_exists(path), file_type='off')
    mesh = pyrender.Mesh.from_trimesh(model)
    return mesh

def get_vertices(path: str = ""):
    model = trimesh.load(if_exists(path), file_type='off')
    return model.vertices

if __name__=="__main__":
    path = "../dataset/"
    #scene = pyrender.Scene(ambient_light=numpy.array([1.0, 1.0, 1.0, 1.0]))
    #for i in range(1, 2):
        #mesh = get_mesh(f"{path}airplane/train/airplane_{i:04d}.off")
        #scene.add(mesh)
    #pyrender.Viewer(scene)

    for i in range(1, 2):
        #mesh = get_model(f"{path}airplane/train/airplane_{i:04d}.off")
        #graph = mesh.vertex_adjacency_graph
        #print(graph.neighbors(0))
        mesh = get_mesh(f"{path}airplane/train/airplane_{i:04d}.off")
        mesh.SamplePoints(256)