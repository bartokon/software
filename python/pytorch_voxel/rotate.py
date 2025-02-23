import math
import random
from typing import Tuple, Union

import torch

from torch_geometric.data import Data
from torch_geometric.data.datapipes import functional_transform
from torch_geometric.transforms import BaseTransform, LinearTransformation

@functional_transform('rotate')
class Rotate(BaseTransform):

    def __init__(
        self,
        degrees: float,
        axis: int = 0,
    ) -> None:
        self.degrees = math.radians(degrees)
        self.axis = axis

    def forward(self, data: Data) -> Data:
        assert data.pos is not None

        degree = self.degrees
        sin, cos = math.sin(degree), math.cos(degree)

        if data.pos.size(-1) == 2:
            matrix = [[cos, sin], [-sin, cos]]
        else:
            if self.axis == 0:
                matrix = [[1, 0, 0], [0, cos, sin], [0, -sin, cos]]
            elif self.axis == 1:
                matrix = [[cos, 0, -sin], [0, 1, 0], [sin, 0, cos]]
            else:
                matrix = [[cos, sin, 0], [-sin, cos, 0], [0, 0, 1]]

        return LinearTransformation(torch.tensor(matrix))(data)

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.degrees}, '
                f'axis={self.axis})')

def rotate(data, degree, axis = 0, device = 'cuda'):
    sin, cos = math.sin(degree), math.cos(degree)
    #data = data.transpose(0, 1)
    if axis == 0:
        matrix = [[1, 0, 0], [0, cos, sin], [0, -sin, cos]]
    elif axis == 1:
        matrix = [[cos, 0, -sin], [0, 1, 0], [sin, 0, cos]]
    else:
        matrix = [[cos, sin, 0], [-sin, cos, 0], [0, 0, 1]]
    matrix = torch.tensor(matrix, device=device)
    #return torch.matmul(data, matrix).transpose(1, 0)
    return torch.matmul(data, matrix)

def rotatete(data, degree, axis = 0, device = 'cuda'):
    sin, cos = math.sin(degree), math.cos(degree)
    data = data.transpose(0, 1)
    if axis == 0:
        matrix = [[1, 0, 0], [0, cos, sin], [0, -sin, cos]]
    elif axis == 1:
        matrix = [[cos, 0, -sin], [0, 1, 0], [sin, 0, cos]]
    else:
        matrix = [[cos, sin, 0], [-sin, cos, 0], [0, 0, 1]]
    matrix = torch.tensor(matrix, device=device)
    return torch.matmul(data, matrix).transpose(1, 0)