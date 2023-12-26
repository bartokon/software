import numpy as np
import point_to_point_utils_3d as utils
import time

if __name__ == '__main__':
    pc_fixed = (np.random.rand(1000, 3) - 0.5) * 10

    fi = utils.deg_to_rad(np.array([10., -5., 2.])) # ROT X ROT Y ROT Z
    t = np.array([50., -10., 20.])

    pc_moved = utils.rotate_pc_3d(pc_fixed, fi)
    pc_moved = utils.translate_pc_3d(pc_moved, t)
    fi = np.array([0., 0., 0.])
    t = np.array([0., 0., 0.])
    error_prev = 1e8
    for _ in range(100):

        h = np.zeros((6, 6))
        for p in pc_moved:
            h += utils.h(p, fi)

        b = np.zeros(6)
        for p0, p1 in zip(pc_fixed, pc_moved):
            b += utils.b(p0, p1, fi, t)

        dx = -(np.linalg.inv(h) @ b)

        t[0] += dx[0]
        t[1] += dx[1]
        t[2] += dx[2]
        fi[0] += dx[3]
        fi[1] += dx[4]
        fi[2] += dx[5]

        pc_corrected = utils.rotate_pc_3d(pc_moved, fi)
        pc_corrected = utils.translate_pc_3d(pc_corrected, t)
        error = utils.sum_distance_3d(pc_fixed, pc_corrected)
        print(f"NEW T: {t}")
        print(f"NEW FI: rad: {fi}, deg: {utils.rad_to_deg(fi)}")
        print(f"error: {error}")
        if error < 1e-8:
            print(f"Iter: {_}")
            break
        if (error > error_prev):
            print(f"Stall")
            break
        error_prev = error

    utils.plot_3d_point_clouds(
        (pc_fixed, "fixed"),
        (pc_moved, "moved"),
        (pc_corrected, "corrected")
    )