import numpy as np
import point_to_plane_utils_3d as utils

import point_information

def all(p0, p1):
    def p(i):
        i[0] *= i[0]
        i[1] *= i[1]
        i[2] *= i[2]
        return np.sum(i)

    def angle_between_planes(a, b):
        cos_fi_top = np.abs(a[0]*b[0] + a[1]*b[1] + a[2]*b[2])
        cos_fi_bot = np.sqrt(p(a)) * np.sqrt(p(b))
        if (cos_fi_bot == 0):
            cos_fi_bot = 1e-6
        cos_fi = cos_fi_top / cos_fi_bot
        if (cos_fi > 1):
            cos_fi = 1
        if (cos_fi < -1):
            cos_fi = -1
        fi = np.arccos(cos_fi)
        return fi * 180 / np.pi

    def angle_between_vectors(a, b):
        cos_fi_top = np.dot(a, b)
        cos_fi_bot = np.sqrt(p(a)) * np.sqrt(p(b))
        if (cos_fi_bot == 0):
            cos_fi_bot = 1e-6
        cos_fi = cos_fi_top / cos_fi_bot
        if (cos_fi > 1):
            print(cos_fi)
            cos_fi = 1
        if (cos_fi < -1):
            print(cos_fi)
            cos_fi = -1
        fi = np.arccos(cos_fi)
        return fi * 180 / np.pi

    a0, b0, c0, d0 = utils.three_points_to_plane(p0)
    a1, b1, c1, d1 = utils.three_points_to_plane(p1)

    n0 = [a0, b0, c0]
    n1 = [a1, b1, c1]

    #alpha = angle_between_planes(n0, n1)
    #theta = angle_between_vectors(n1, np.cross((p0[0] - p1[0]), n0))
    #theta = angle_between_vectors(n0, np.cross((p0[0] - p1[0]), n1))
    #d = np.sqrt(np.sum(np.abs(p0[0]-p1[0])))
    d = utils.distance_3d(p1[0], p0[0])
    #beta = 0

    temporary = (p1[0] - p0[0]) / utils.distance_3d(p1[0], p0[0])
    u = np.array(n0)
    v = np.cross(u, temporary)
    w = np.cross(u, v)
    alpha = np.dot(v, n1)
    beta = np.dot(u, temporary)
    theta = np.arctan2(np.dot(w, n1), np.dot(u, n1))
    #print(f"{u=}\n{v=}\n{w=}\n")
    #print(f"{alpha=}\n{beta=}\n{theta=}")
    #exit()
    return alpha, beta, theta, d

#HISTOGRAM OF ONE BALL
def distance_histogram_v3(point: point_information):
    point_alphas = []
    point_betas = []
    point_thetas = []
    point_distance = []
    p0 = [point.pc_indexes[0], point.pc_indexes[1], point.pc_indexes[2]]
    for i in range(0, len(point.pc_indexes) - 3):
        if (point.pc_indexes[i + 1] == None).any(): break
        if (point.pc_indexes[i + 2] == None).any(): break
        if (point.pc_indexes[i + 3] == None).any(): break
        p1 = [point.pc_indexes[i + 1], point.pc_indexes[i + 2], point.pc_indexes[i + 3]]
        a, b, t, d = all(p0, p1)
        point_alphas.append(a)
        point_thetas.append(t)
        point_distance.append(d)

    return point_alphas, point_betas, point_thetas, point_distance