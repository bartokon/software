
import numpy as np
import point_to_point_utils_2d as utils
import time

if __name__ == '__main__':
    pc_fixed = (np.random.rand(1000, 2) - 0.5) * 10

    fi = 20.
    t = np.array([4, -5])

    pc_moved = utils.rotate_pc_2d(pc_fixed, fi)
    pc_moved = utils.translate_pc_2d(pc_moved, t)
    iters = 10000
    start_time = time.time()
    if 1:
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
            #print(f"NEW T: {t}")
            #print(f"NEW FI: {fi}")
            #print(f"error: {error}")
            if error < 1e-8:
                break
        stop_time = time.time()
        print("Least Squares")
        print(f"Done in: {stop_time-start_time}s")
        print(f"NEW T: {t}")
        print(f"NEW FI: {fi}")
        print(f"error: {error}")

    start_time = time.time()
    if 1:
        fi = 0.
        t = np.array([0., 0.])
        for _ in range(iters):
            h = np.zeros((3, 3))
            b = np.zeros((3))

            for p0, p1 in zip(pc_fixed, pc_moved):
                h_, b_ = utils.unified(p0, p1, fi, t)
                h += h_
                b += b_

            dx = -(np.linalg.inv(h) @ b)

            t[0] += dx[0]
            t[1] += dx[1]
            fi += dx[2]

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

    pc_corrected = utils.rotate_pc_2d(pc_moved, fi)
    pc_corrected = utils.translate_pc_2d(pc_corrected, t)
    utils.plot_2d_point_clouds((pc_fixed, "fixed", "*", 500), (pc_moved, "moved", "x", 100), (pc_corrected, "corrected", "o", 100))