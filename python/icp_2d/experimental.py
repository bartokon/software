import numpy as np
from scipy.spatial import KDTree
import point_to_point_utils_2d as utils
import time
from copy import deepcopy

def calculate_angle(p0, p1, p2):
    v01 = p0 - p1
    v02 = p0 - p2
    vdot = np.dot(v01, v02)
    v01_mag = np.sqrt(v01[0]**2 + v01[1]**2)
    v02_mag = np.sqrt(v02[0]**2 + v02[1]**2)
    radians = np.arccos(vdot/(v01_mag*v02_mag))
    angle = utils.rad_to_deg(radians) #Does not matter if radians or angle
    return angle

#TODO: Add histogram of neighbours?
#TODO: Add p2p icp in last stage?
#TODO: Group histogram and take these with max difference (most unique)?
def histogram_2d(point_cloud, neighbors = 15):
    histogram = []
    tree = KDTree(point_cloud)
    for i in range(len(point_cloud)):
        center = point_cloud[i]
        distance, indexes = tree.query(center, k=neighbors, workers=-1)
        neighbor = point_cloud[indexes]
        angles = []
        for c in range(0, len(neighbor)):
            center = neighbor[c]
            for x in range(0, len(neighbor)):
                for y in range(x, len(neighbor)):
                    if (c == x or c == y or x == y): continue
                    angles.append(calculate_angle(center, neighbor[x], neighbor[y]))
        histogram.append(angles)
    return histogram

def calc_dx(h, b):
    h_det = utils.det_3x3(h)
    #ERROR: IF H_DET = 0 matrix is not inversable!!!!!!! error ~0
    if (h_det == 0):
        h_det = 1e-10

    h_00 = np.array([[h[1,1], h[1,2]], [h[2,1], h[2,2]]])
    h_01 = np.array([[h[1,0], h[1,2]], [h[2,0], h[2,2]]])
    h_02 = np.array([[h[1,0], h[1,1]], [h[2,0], h[2,1]]])
    h_10 = np.array([[h[0,1], h[0,2]], [h[2,1], h[2,2]]])
    h_11 = np.array([[h[0,0], h[0,2]], [h[2,0], h[2,2]]])
    h_12 = np.array([[h[0,0], h[0,1]], [h[2,0], h[2,1]]])
    h_20 = np.array([[h[0,1], h[0,2]], [h[1,1], h[1,2]]])
    h_21 = np.array([[h[0,0], h[0,2]], [h[1,0], h[1,2]]])
    h_22 = np.array([[h[0,0], h[0,1]], [h[1,0], h[1,1]]])
    h_adj = np.array([
        [utils.det_2x2(h_00), -utils.det_2x2(h_10), utils.det_2x2(h_20)],
        [-utils.det_2x2(h_01), utils.det_2x2(h_11), -utils.det_2x2(h_21)],
        [utils.det_2x2(h_02), -utils.det_2x2(h_12), utils.det_2x2(h_22)]
    ])

    h_inverted = h_adj / (h_det)
    dx_0 = -(h_inverted[0,0]*b[0] + h_inverted[0,1]*b[1] + h_inverted[0,2]*b[2])
    dx_1 = -(h_inverted[1,0]*b[0] + h_inverted[1,1]*b[1] + h_inverted[1,2]*b[2])
    dx_2 = -(h_inverted[2,0]*b[0] + h_inverted[2,1]*b[1] + h_inverted[2,2]*b[2])
    return dx_0, dx_1, dx_2

def icp_hist(pc_0, pc_1, iters=1000, neighbors=3):
        start_time = time.time()
        pc_fixed = pc_0
        pc_moved = pc_1

        fi = 0.
        t = np.array([0., 0.])

        pc_fixed_histogram = histogram_2d(pc_0, neighbors)
        pc_moved_histogram = histogram_2d(pc_1, neighbors)

        histogram_tree = KDTree(pc_fixed_histogram)
        corresponding_histogram_distance, corresponding_histogram_idx = histogram_tree.query(pc_moved_histogram, k=1, workers=-1)
        pc_matched = pc_fixed[corresponding_histogram_idx]

        error_prev = 1e8
        for _ in range(iters):
            h = np.zeros((3, 3))
            b = np.zeros((3))
            for p0, p1 in zip(pc_matched, pc_moved):
                h_, b_ = utils.unified(p0, p1, fi, t)
                h += h_
                b += b_
            dx_0, dx_1, dx_2 = calc_dx(h, b)
            t[0] += dx_0
            t[1] += dx_1
            fi   += dx_2
            pc_corrected = utils.rotate_pc_2d(pc_moved, fi)
            pc_corrected = utils.translate_pc_2d(pc_corrected, t)
            error = utils.sum_distance_2d(pc_matched, pc_corrected)
            if error < 1e-8:
                break
            if error_prev <= error:
                print(f"Error: {error} prev: {error_prev}")
                break
            error_prev = error

        stop_time = time.time()
        print("Opt Least Squares")
        print(f"Done in: {stop_time-start_time}s")
        print(f"NEW T: {t}")
        print(f"NEW FI: {utils.rad_to_deg(fi)}")
        print(f"error: {error}")

        return t, utils.rad_to_deg(fi)