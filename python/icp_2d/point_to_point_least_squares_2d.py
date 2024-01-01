import numpy as np
import point_to_point_utils_2d as utils
import time

if __name__ == '__main__':
    pc_fixed = (np.random.rand(1000, 2) - 0.5) * 10

    fi = 21.
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

    pc_corrected = utils.rotate_pc_2d(pc_moved, fi)
    pc_corrected = utils.translate_pc_2d(pc_corrected, t)
    utils.plot_2d_point_clouds((pc_fixed, "fixed", "*", 500), (pc_moved, "moved", "x", 100), (pc_corrected, "corrected", "o", 100))