import numpy as np
import point_to_point_utils_2d as utils
import time
from scipy.spatial import KDTree
from copy import deepcopy
import experimental

def is_point_in_circle(center, radius, point):
    a = center[0]
    b = center[1]
    x = point[0]
    y = point[1]
    distance = ((x - a)**2 + (y - b)**2)**0.5
    if distance <= radius:
        return True
    else:
        return False

def near(a, b, c):
    d = np.abs(np.array(a) - np.array(b))
    if ((d < c).all()):
        return True
    else:
        return False

def cubic(x):
    return x**3

def parable(x):
    return x**2

def sin(x):
    return np.sin(x)

def cos(x):
    return np.cos(x)

if __name__ == '__main__':
    pc_len = 500
    #pc_fixed = (np.random.rand(pc_len, 2) - 0.5) * 100
    #pc_fixed = np.array([[x, parable(x)*sin(x)*cos(x)] for x in np.linspace(-10, 10, num=pc_len)])
    pc_fixed = np.array([[x, -sin(x)*parable(x)] for x in np.linspace(-5, 5, num=pc_len)])

    fi = utils.deg_to_rad(40)
    t = np.array([10, -5])
    pc_moved = pc_fixed + np.random.normal(0, 0.0, size=(pc_len, 2)) #Add noise
    pc_moved_noise = deepcopy(pc_moved)
    minimum_error = utils.sum_distance_2d(pc_fixed, pc_moved)

    pc_moved = utils.rotate_pc_2d(pc_moved, fi)
    pc_moved = utils.translate_pc_2d(pc_moved, t)

    iters = 1000
    if 0:
        start_time = time.time()
        fi = 0.
        t = np.array([0., 0.])
        for _ in range(iters):

            h = np.zeros((3, 3))
            for p in pc_moved:
                h += utils.h(p, fi)

            b = np.zeros(3)
            for p0, p1 in zip(pc_fixed, pc_moved):
                b += utils.b(p0, p1, fi, t)

            dx = -(np.linalg.inv(h) @ b)
            t[0] += dx[0]
            t[1] += dx[1]
            fi += dx[2]

            pc_corrected = utils.rotate_pc_2d(pc_moved, fi)
            pc_corrected = utils.translate_pc_2d(pc_corrected, t)
            error = utils.sum_distance_2d(pc_fixed, pc_corrected)
            if error < 1e-8:
                break
        stop_time = time.time()
        print("Least Squares")
        print(f"Done in: {stop_time-start_time}s")
        print(f"NEW T: {t}")
        print(f"NEW FI: {fi}")
        print(f"error: {error}")
        fi = utils.rad_to_deg(fi)

    if 0:
        start_time = time.time()
        fi = 0.
        t = np.array([0., 0.])
        for _ in range(iters):
            h = np.zeros((3, 3))
            b = np.zeros((3))

            for p0, p1 in zip(pc_fixed, pc_moved):
                h_, b_ = utils.unified(p0, p1, fi, t)
                h += h_
                b += b_

            h_det = utils.det_3x3(h)
            h_00 = np.array([[h[1,1], h[1,2]], [h[2,1], h[2,2]]])
            h_01 = np.array([[h[1,0], h[1,2]], [h[2,0], h[2,2]]])
            h_02 = np.array([[h[1,0], h[1,1]], [h[2,0], h[2,1]]])
            h_10 = np.array([[h[0,1], h[0,2]], [h[2,1], h[2,2]]])
            h_11 = np.array([[h[0,0], h[0,2]], [h[2,0], h[2,2]]])
            h_12 = np.array([[h[0,0], h[0,1]], [h[2,0], h[2,1]]])
            h_20 = np.array([[h[0,1], h[0,2]], [h[1,1], h[1,2]]])
            h_21 = np.array([[h[0,0], h[0,2]], [h[1,0], h[1,2]]])
            h_22 = np.array([[h[0,0], h[0,1]], [h[1,0], h[1,1]]])
            #h_cofactor = np.array([
                #[det_2x2(h_00), -det_2x2(h_01), det_2x2(h_02)],
                #[-det_2x2(h_10), det_2x2(h_11), -det_2x2(h_12)],
                #[det_2x2(h_20), -det_2x2(h_21), det_2x2(h_22)]
            #])
            h_adj = np.array([
                [utils.det_2x2(h_00), -utils.det_2x2(h_10), utils.det_2x2(h_20)],
                [-utils.det_2x2(h_01), utils.det_2x2(h_11), -utils.det_2x2(h_21)],
                [utils.det_2x2(h_02), -utils.det_2x2(h_12), utils.det_2x2(h_22)]
            ])
            h_inverted = h_adj / h_det
            dx_0 = -(h_inverted[0,0]*b[0] + h_inverted[0,1]*b[1] + h_inverted[0,2]*b[2])
            dx_1 = -(h_inverted[1,0]*b[0] + h_inverted[1,1]*b[1] + h_inverted[1,2]*b[2])
            dx_2 = -(h_inverted[2,0]*b[0] + h_inverted[2,1]*b[1] + h_inverted[2,2]*b[2])
            t[0] += dx_0
            t[1] += dx_1
            fi += dx_2

            pc_corrected = utils.rotate_pc_2d(pc_moved, fi)
            pc_corrected = utils.translate_pc_2d(pc_corrected, t)
            error = utils.sum_distance_2d(pc_fixed, pc_corrected)
            #print(f"NEW T: {t}")
            #print(f"NEW FI: {fi}")
            #print(f"error: {error}")
            if error < 1e-8:
                break
        stop_time = time.time()
        print("Opt Least Squares")
        print(f"Done in: {stop_time-start_time}s")
        print(f"NEW T: {t}")
        print(f"NEW FI: {fi}")
        print(f"error: {error}")
        fi = utils.rad_to_deg(fi)

    if 1:
        #shuffle pc_moved and pc_fixed
        np.random.shuffle(pc_fixed)
        np.random.shuffle(pc_moved)
        #Does not work if data is symmetrical, angles are not enough?
        t, fi = experimental.icp_hist(pc_fixed, pc_moved, iters, 15)

    pc_corrected = utils.rotate_pc_2d(pc_moved, utils.deg_to_rad(fi))
    pc_corrected = utils.translate_pc_2d(pc_corrected, t)
    print(f"Minimum error: {minimum_error}")
    utils.plot_2d_point_clouds(
        (pc_fixed, "fixed", "s", 100),
        (pc_moved, "moved", "+", 100),
        (pc_corrected, "corrected", "x", 100),
        (pc_moved_noise, "noise", ".", 100)
    )