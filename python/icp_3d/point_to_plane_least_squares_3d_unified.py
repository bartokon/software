import numpy as np
import point_to_plane_utils_3d as utils

def det_nxn(h, n):
    sum = 0
    if (n == 2):
        sum = h[0][0]*h[1][1] - h[0][1]*h[1][0]
    else:
        for i in range(0, n):
            sum += np.power(-1, i) * h[i][0] * det_nxn(d(h, i, 0), n-1)
    return sum

def d(matrix, row, col):
    t = np.delete(matrix, row, 0)
    return np.delete(t, col, 1)

def cofactor_matrix_6x6(h):
    cofactor_h = np.zeros(shape=(6,6), dtype=np.float64)
    for i in range(0, 6):
        for j in range(0, 6):
            cofactor_h[i][j] = np.power(-1, i+j) * det_nxn(d(h, i, j), 5)
    return cofactor_h

def inverse_6x6(h):
    #h = np.random.randint(0, 1e3, size=(6,6))
    #h_det = det_nxn(h, 6) #DET 6x6 is max level... (Error is sus high...)
    #print (abs(np.linalg.det(h) - h_det))
    #exit()

    h_det = det_nxn(h, 6)
    h_cofactor = cofactor_matrix_6x6(h)
    h_adjugate = h_cofactor.T
    h_inv = h_adjugate / h_det
    return h_inv

if __name__ == '__main__':
    pc_fixed = (np.random.rand(1000, 3) - 0.5) * 10
    #pc_fixed = utils.read_to_points_xyz("bunny_part1.xyz")
    #pc_fixed = utils.read_to_points_xyz("save_txt.txt")

    pc_fixed_normals = utils.get_point_cloud_normals(pc_fixed)
    fi = utils.deg_to_rad(np.array([10., -5., 2.])) # ROT X ROT Y ROT Z
    t = np.array([45., -10., 20.])

    pc_moved = utils.rotate_pc_3d(pc_fixed, fi)
    pc_moved = utils.translate_pc_3d(pc_moved, t)

    fi = np.array([0., 0., 0.])
    t = np.array([0., 0., 0.])
    error_prev = 1e8

    for _ in range(100):
        #ERROR CALC PHASE
        pc_corrected = utils.rotate_pc_3d(pc_moved, fi)
        pc_corrected = utils.translate_pc_3d(pc_corrected, t)
        error = utils.sum_distance_3d(pc_fixed, pc_corrected)
        print(f"NEW T: {t}")
        print(f"NEW FI: rad: {fi}, deg: {utils.rad_to_deg(fi)}")
        print(f"error: {error}")
        if error < 1e-6:
            print(f"Iter: {_}")
            break
        if (error > error_prev):
            print(f"Stall")
            break
        error_prev = error

        #ICP CALC
        h_s = np.zeros((6, 6))
        b_s = np.zeros(6)
        for p0, p1, n0 in zip(pc_fixed, pc_moved, pc_fixed_normals):
            h, b = utils.hb(p0, p1, fi, t, n0[0:3])
            h_s += h
            b_s += b

        dx = -(inverse_6x6(h_s) @ b_s)
        t[0]  += dx[0]
        t[1]  += dx[1]
        t[2]  += dx[2]
        fi[0] += dx[3]
        fi[1] += dx[4]
        fi[2] += dx[5]

    utils.plot_3d_point_clouds(
        (pc_fixed, "fixed"),
        (pc_moved, "moved"),
        (pc_corrected, "corrected")
    )