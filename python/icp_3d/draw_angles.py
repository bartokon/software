import numpy as np
from scipy.spatial import KDTree
import matplotlib.pyplot as plt
from copy import deepcopy

import point_to_plane_utils_3d as utils
from point_information import point_information
import point_angles

def extract_points(pc, radius = 1, lim=0.1, min_points = 0):
    QUERY_POINTS = 256
    tree = KDTree(pc, balanced_tree=True)
    point_list = []

    for p in pc:
        distance, index = tree.query(p, k=QUERY_POINTS, distance_upper_bound=radius, workers=-1)
        index = index[distance != float('inf')]
        distance = distance[distance != float('inf')]

        if (len(index) <= min_points):
            continue

        point = point_information(distance, index, pc[index])
        avg = (np.var(point.get_neighbours(), axis=0))
        if ((avg[0] > lim) or (avg[1] > lim) or (avg[2] > lim)):
            point_list.append(point)

    if (len(point_list) == 0):
        print("LIST IS EMPTY")
        #exit(-1)

    return point_list

def sift_with_histogram(pc, radius = 5, lim = 0, min_points = 5):
    sift = extract_points(pc, radius = radius, lim = lim, min_points = min_points)
    point_degs_0 = []
    point_degs_1 = []
    point_degs_2 = []
    point_distance = []
    for s, r in zip(sift, range(0, len(sift))):
        a, b, t, d = point_angles.distance_histogram_v3(s)
        point_degs_0.append(a)
        point_degs_1.append(b)
        point_degs_2.append(t)
        point_distance.append(d)
    return sift, point_degs_0, point_degs_1, point_degs_2, point_distance

if __name__ == '__main__':

    pc_fixed = (np.random.rand(1000, 3) - 0.5) * 2
    #pc_fixed = utils.read_to_points_xyz("bunny_part1.xyz")[::1]
    #pc_fixed = utils.read_to_points_xyz("save_txt.txt")[::100]
    #pc_fixed = np.array([[np.sin(i), i, 0] for i in np.arange(start=-np.pi, stop=np.pi, step=0.005)])

    s0 = extract_points(pc_fixed, radius=0.5, lim=0.065, min_points=3)

    fig = plt.figure(figsize=plt.figaspect(2.))
    ax = fig.add_subplot(1, 1, 1, projection="3d")
    print(len(s0))
    for p in s0:
        p.fig = fig
        p.ax = ax
        p.add_coordinates()
        p.draw_circle()
        p.draw_point()
        p.draw_neighbours()
        p.draw_arrows()
        p.draw_plane()
        p.draw_normal()
        break

    #s0[0].ax.scatter(pc_fixed[:, 0], pc_fixed[:, 1], pc_fixed[:, 2], alpha=0.25)
    #x, y, z = s0[0].point_coordinates()
    #s0[0].ax.set_xlim(-10 + x, 10 + x)
    #s0[0].ax.set_ylim(-10 + y, 10 + y)
    #s0[0].ax.set_zlim(-10 + z, 10 + z)

    plt.show()
    exit()
