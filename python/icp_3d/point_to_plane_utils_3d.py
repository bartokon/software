import numpy as np
import matplotlib.pyplot as plt
import random
from scipy.spatial import KDTree

def plot_3d_point_clouds(*args):
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    #ax.set_xlim(-10, 10)
    #ax.set_ylim(-10, 10)
    #ax.set_zlim(-10, 10)
    for pc, label in args:
        ax.scatter(xs=pc[:, 0], ys=pc[:, 1], zs=pc[:, 2], label=label, s=1)
    plt.legend()
    plt.show()

#Ecluidean distance
def distance_3d(p0, p1):
    return np.sqrt((p0[0]-p1[0])**2 + (p0[1]-p1[1])**2 + (p0[2]-p1[2])**2)

#OBJECTIVE
#Sum of Ecluidean distances
def sum_distance_3d(pc0, pc1):
    sum = 0
    for p0, p1 in zip(pc0, pc1):
        sum += distance_3d(p0, p1)
    return sum

#Matrix multiply each point by R
def rotate_pc_3d(pc, fi):
    R = euler_angles_to_rotation_matrix(fi)
    return np.array([R @ P for P in pc])

def translate_pc_3d(pc, T):
    return np.array([p + T for p in pc])

#error vector
def error_vector(p0, p1, fi, T, n0):
    R = euler_angles_to_rotation_matrix(fi)
    return (R @ p1 + T - p0)*n0

def h(p1, fi, n0):
    j = jackobian_error_vector(p1, fi, n0)
    return j.T @ j

def b(p0, p1, fi, T, n0):
    j = jackobian_error_vector(p1, fi, n0)
    e = error_vector(p0, p1, fi, T, n0)
    return j.T @ e

def hb(p0, p1, rad, T, n0):
    h, b = jackobian_error_vector_unified(p0, p1, rad, T, n0)
    return h, b

#ROLL X ROLL Y ROLL Z
def euler_angles_to_rotation_matrix(deg: np.array) -> np.array:
    """Get Euler angles from rotation matrix."""
    a = deg[0]
    b = deg[1]
    c = deg[2]

    R = np.array(
        [
            [
                np.cos(b) * np.cos(c),
                np.sin(a) * np.sin(b) * np.cos(c) - np.cos(a) * np.sin(c),
                np.cos(a) * np.sin(b) * np.cos(c) + np.sin(a) * np.sin(c),
            ],
            [
                np.cos(b) * np.sin(c),
                np.sin(a) * np.sin(b) * np.sin(c) + np.cos(a) * np.cos(c),
                np.cos(a) * np.sin(b) * np.sin(c) - np.sin(a) * np.cos(c),
            ],
            [
                -np.sin(b),
                np.sin(a) * np.cos(b),
                np.cos(a) * np.cos(b),
            ],
        ]
    )
    return R

#TODO: check for more common ops
def jackobian_error_vector_unified(p0, p1, rad, T, n0):
    a = rad[0]
    b = rad[1]
    c = rad[2]

    #MULTIPLY EXTRACT
    cos_b = np.cos(b)
    sin_b = np.sin(b)

    sin_a_sin_b = np.sin(a) * np.sin(b)
    sin_a_sin_c = np.sin(a) * np.sin(c)
    sin_a_cos_b = np.sin(a) * np.cos(b)
    sin_a_cos_c = np.sin(a) * np.cos(c)

    sin_b_sin_c = np.sin(b) * np.sin(c)
    sin_c_cos_b = np.sin(c) * np.cos(b)

    cos_a_sin_b = np.cos(a) * np.sin(b)
    cos_a_sin_c = np.cos(a) * np.sin(c)
    cos_a_cos_b = np.cos(a) * np.cos(b)
    cos_a_cos_c = np.cos(a) * np.cos(c)
    cos_b_cos_c = np.cos(b) * np.cos(c)
    cos_c_sin_b = np.cos(c) * np.sin(b)
    cos_b_sin_c = np.cos(b) * np.sin(c)

    sin_a_sin_b_cos_c = sin_a_sin_b * np.cos(c)
    cos_a_sin_b_sin_c = cos_a_sin_b * np.sin(c)
    cos_a_sin_b_cos_c = cos_a_sin_b * np.cos(c)
    sin_a_sin_b_sin_c = sin_a_sin_b * np.sin(c)
    sin_a_cos_b_cos_c = sin_a_cos_b * np.cos(c)
    cos_a_cos_b_cos_c = cos_a_cos_b * np.cos(c)
    cos_a_cos_b_sin_c = cos_a_cos_b * np.sin(c)
    sin_a_cos_b_sin_c = sin_a_cos_b * np.sin(c)

    #Equations
    r01_a =  cos_a_sin_b_cos_c + sin_a_sin_c
    r02_a = -sin_a_sin_b_cos_c + cos_a_sin_c
    r11_a =  cos_a_sin_b_sin_c - sin_a_cos_c
    r12_a = -sin_a_sin_b_sin_c - cos_a_cos_c
    r21_a =  cos_a_cos_b
    r22_a = -sin_a_cos_b

    r00_b = -cos_c_sin_b
    r01_b =  sin_a_cos_b_cos_c
    r02_b =  cos_a_cos_b_cos_c
    r10_b = -sin_b_sin_c
    r11_b =  sin_a_cos_b_sin_c
    r12_b =  cos_a_cos_b_sin_c
    r20_b = -cos_b
    r21_b = -sin_a_sin_b
    r22_b =  cos_a_sin_b

    r00_c = -sin_c_cos_b
    r01_c = -sin_a_sin_b_sin_c - cos_a_cos_c
    r02_c = -cos_a_sin_b_sin_c + sin_a_cos_c
    r10_c =  cos_b_cos_c
    r11_c =  sin_a_sin_b_cos_c - cos_a_sin_c
    r12_c =  cos_a_sin_b_cos_c + sin_a_sin_c

    df_da_0 = r01_a * p1[1] + r02_a * p1[2]
    df_da_1 = r11_a * p1[1] + r12_a * p1[2]
    df_da_2 = r21_a * p1[1] + r22_a * p1[2]

    df_db_0 = r00_b * p1[0] + r01_b * p1[1] + r02_b * p1[2]
    df_db_1 = r10_b * p1[0] + r11_b * p1[1] + r12_b * p1[2]
    df_db_2 = r20_b * p1[0] + r21_b * p1[1] + r22_b * p1[2]

    df_dc_0 = r00_c * p1[0] + r01_c * p1[1] + r02_c * p1[2]
    df_dc_1 = r10_c * p1[0] + r11_c * p1[1] + r12_c * p1[2]

    df_da_0_n0_0 = df_da_0 * n0[0]
    df_da_1_n0_1 = df_da_1 * n0[1]
    df_da_2_n0_2 = df_da_2 * n0[2]
    df_db_0_n0_0 = df_db_0 * n0[0]
    df_db_1_n0_1 = df_db_1 * n0[1]
    df_db_2_n0_2 = df_db_2 * n0[2]
    df_dc_0_n0_0 = df_dc_0 * n0[0]
    df_dc_1_n0_1 = df_dc_1 * n0[1]

    j = np.array((
        np.array([n0[0], 0, 0, df_da_0_n0_0, df_db_0_n0_0, df_dc_0_n0_0]),
        np.array([0, n0[1], 0, df_da_1_n0_1, df_db_1_n0_1, df_dc_1_n0_1]),
        np.array([0, 0, n0[2], df_da_2_n0_2, df_db_2_n0_2,               0])
    ), dtype=np.float64)

    j_t = np.array((
        np.array([          n0[0],               0,               0]),
        np.array([              0,           n0[1],               0]),
        np.array([              0,               0,           n0[2]]),
        np.array([df_da_0_n0_0, df_da_1_n0_1, df_da_2_n0_2]),
        np.array([df_db_0_n0_0, df_db_1_n0_1, df_db_2_n0_2]),
        np.array([df_dc_0_n0_0, df_dc_1_n0_1,               0]),
    ), dtype=np.float64)

    hm_00 = j_t[0][0]*j[0][0]
    hm_30 = j_t[3][0]*j[0][0]
    hm_40 = j_t[4][0]*j[0][0]
    hm_50 = j_t[5][0]*j[0][0]

    hm_11 = j_t[1][1]*j[1][1]
    hm_31 = j_t[3][1]*j[1][1]
    hm_41 = j_t[4][1]*j[1][1]
    hm_51 = j_t[5][1]*j[1][1]

    hm_22 = j_t[2][2]*j[2][2]
    hm_32 = j_t[3][2]*j[2][2]
    hm_42 = j_t[4][2]*j[2][2]

    hm_03 = j_t[0][0]*j[0][3]
    hm_13 = j_t[1][1]*j[1][3]
    hm_23 = j_t[2][2]*j[2][3]
    hm_33 = j_t[3][0]*j[0][3] + j_t[3][1]*j[1][3] + j_t[3][2]*j[2][3]
    hm_43 = j_t[4][0]*j[0][3] + j_t[4][1]*j[1][3] + j_t[4][2]*j[2][3]
    hm_53 = j_t[5][0]*j[0][3] + j_t[5][1]*j[1][3]

    hm_04 = j_t[0][0]*j[0][4]
    hm_14 = j_t[1][1]*j[1][4]
    hm_24 = j_t[2][2]*j[2][4]
    hm_34 = j_t[3][0]*j[0][4] + j_t[3][1]*j[1][4] + j_t[3][2]*j[2][4]
    hm_44 = j_t[4][0]*j[0][4] + j_t[4][1]*j[1][4] + j_t[4][2]*j[2][4]
    hm_54 = j_t[5][0]*j[0][4] + j_t[5][1]*j[1][4]

    hm_05 = j_t[0][0]*j[0][5]
    hm_15 = j_t[1][1]*j[1][5]
    hm_35 = j_t[3][0]*j[0][5] + j_t[3][1]*j[1][5]
    hm_45 = j_t[4][0]*j[0][5] + j_t[4][1]*j[1][5]
    hm_55 = j_t[5][0]*j[0][5] + j_t[5][1]*j[1][5]

    h = np.array((
        np.array([hm_00,     0,     0, hm_03, hm_04, hm_05]),
        np.array([    0, hm_11,     0, hm_13, hm_14, hm_15]),
        np.array([    0,     0, hm_22, hm_23, hm_24,     0]),
        np.array([hm_30, hm_31, hm_32, hm_33, hm_34, hm_35]),
        np.array([hm_40, hm_41, hm_42, hm_43, hm_44, hm_45]),
        np.array([hm_50, hm_51,     0, hm_53, hm_54, hm_55]),
    ))

    R = np.array(
        [
            [
                cos_b_cos_c,
                sin_a_sin_b_cos_c - cos_a_sin_c,
                cos_a_sin_b_cos_c + sin_a_sin_c,
            ],
            [
                cos_b_sin_c,
                sin_a_sin_b_sin_c + cos_a_cos_c,
                cos_a_sin_b_sin_c - sin_a_cos_c,
            ],
            [
                -sin_b,
                sin_a_cos_b,
                cos_a_cos_b,
            ],
        ]
    )

    rp1_0 = (R[0][0] * p1[0] + R[0][1] * p1[1] + R[0][2] * p1[2] + T[0] - p0[0]) * n0[0]
    rp1_1 = (R[1][0] * p1[0] + R[1][1] * p1[1] + R[1][2] * p1[2] + T[1] - p0[1]) * n0[1]
    rp1_2 = (R[2][0] * p1[0] + R[2][1] * p1[1] + R[2][2] * p1[2] + T[2] - p0[2]) * n0[2]
    rp = np.array([rp1_0, rp1_1, rp1_2])

    b_0 = j_t[0][0] * rp[0]
    b_1 = j_t[1][1] * rp[1]
    b_2 = j_t[2][2] * rp[2]
    b_3 = j_t[3][0] * rp[0] + j_t[3][1] * rp[1] + j_t[3][2] * rp[2]
    b_4 = j_t[4][0] * rp[0] + j_t[4][1] * rp[1] + j_t[4][2] * rp[2]
    b_5 = j_t[5][0] * rp[0] + j_t[5][1] * rp[1]
    b = np.array([b_0, b_1, b_2, b_3, b_4, b_5])

    return h, b

def jackobian_error_vector(p1, deg, n0):
    a = deg[0]
    b = deg[1]
    c = deg[2]
    x = p1[0]
    y = p1[1]
    z = p1[2]

    #r11 = np.cos(c) * np.cos(b)
    #r12 = np.sin(a) * np.sin(b) * np.cos(c) - np.cos(a) * np.sin(c)
    #r13 = np.cos(a) * np.sin(b) * np.cos(c) + np.sin(a) * np.sin(c)
    #r21 = np.cos(b) * np.sin(c)
    #r22 = np.sin(a) * np.sin(b) * np.sin(c) + np.cos(a) * np.cos(c)
    #r23 = np.cos(a) * np.sin(b) * np.sin(c) - np.sin(a) * np.cos(c)
    #r31 = -np.sin(b)
    #r32 = np.sin(a) * np.cos(b)
    #r33 = np.cos(a) * np.cos(b)

    r00_a =  0
    r01_a =  np.cos(a) * np.sin(b) * np.cos(c) + np.sin(a) * np.sin(c)
    r02_a = -np.sin(a) * np.sin(b) * np.cos(c) + np.cos(a) * np.sin(c)
    r10_a =  0
    r11_a =  np.cos(a) * np.sin(b) * np.sin(c) - np.sin(a) * np.cos(c)
    r12_a = -np.sin(a) * np.sin(b) * np.sin(c) - np.cos(a) * np.cos(c)
    r20_a =  0
    r21_a =  np.cos(a) * np.cos(b)
    r22_a = -np.sin(a) * np.cos(b)

    r00_b = -np.cos(c) * np.sin(b)
    r01_b =  np.sin(a) * np.cos(b) * np.cos(c)
    r02_b =  np.cos(a) * np.cos(b) * np.cos(c)
    r10_b = -np.sin(b) * np.sin(c)
    r11_b =  np.sin(a) * np.cos(b) * np.sin(c)
    r12_b =  np.cos(a) * np.cos(b) * np.sin(c)
    r20_b = -np.cos(b)
    r21_b = -np.sin(a) * np.sin(b)
    r22_b =  np.cos(a) * np.sin(b)

    r00_c = -np.sin(c) * np.cos(b)
    r01_c = -np.sin(a) * np.sin(b) * np.sin(c) - np.cos(a) * np.cos(c)
    r02_c = -np.cos(a) * np.sin(b) * np.sin(c) + np.sin(a) * np.cos(c)
    r10_c =  np.cos(b) * np.cos(c)
    r11_c =  np.sin(a) * np.sin(b) * np.cos(c) - np.cos(a) * np.sin(c)
    r12_c =  np.cos(a) * np.sin(b) * np.cos(c) + np.sin(a) * np.sin(c)
    r20_c = 0
    r21_c = 0
    r22_c = 0

    df_da_0 = r00_a * x + r01_a * y + r02_a * z
    df_da_1 = r10_a * x + r11_a * y + r12_a * z
    df_da_2 = r20_a * x + r21_a * y + r22_a * z

    df_db_0 = r00_b * x + r01_b * y + r02_b * z
    df_db_1 = r10_b * x + r11_b * y + r12_b * z
    df_db_2 = r20_b * x + r21_b * y + r22_b * z

    df_dc_0 = r00_c * x + r01_c * y + r02_c * z
    df_dc_1 = r10_c * x + r11_c * y + r12_c * z
    df_dc_2 = r20_c * x + r21_c * y + r22_c * z

    j = np.array((
        np.array([1, 0, 0, df_da_0, df_db_0, df_dc_0]) * n0[0],
        np.array([0, 1, 0, df_da_1, df_db_1, df_dc_1]) * n0[1],
        np.array([0, 0, 1, df_da_2, df_db_2, df_dc_2]) * n0[2]
    ))

    return j

def rad_to_deg(fi):
    return fi * 180 / np.pi

def deg_to_rad(fi):
    return fi * np.pi / 180

def extract_points(points: [], indexes: []):
    container = []
    for index in indexes:
        container.append(points[index])
    return np.array(container)

def three_points_to_plane(points):
    p1 = points[0]
    p2 = points[1]
    p3 = points[2]

    v1 = p3 - p1
    v2 = p2 - p1
    #a = <ai,aj,ak> #b<bi,bj,bk)
    #axb = | i  j  k| =
    #      |ai aj ak|
    #      |bi bj bk|
    #|aj ak|  + |ai ak|  + |ai aj|
    #|bj bk|i + |bi bk|j + |bi bj|k =
    #cross axb =(ajbk  - akbj)i + (aibk - akbi)j + (aibj - ajbi)k
    cp = np.cross(v1, v2)
    a, b, c = cp
    #dot a.b = (a1b1) + (a2b2) + (a3b3)
    d = np.dot(cp, p3)
    #equation: ax + by + cz = d
    return np.array([a, b, c, d])

def get_point_cloud_normals(point_cloud: np.array):
    kd_tree = KDTree(point_cloud)
    normals = []
    for point in point_cloud:
        _, idx = kd_tree.query(point, k = 3, p=2, workers = -1)
        neighbours = extract_points(point_cloud, idx)
        three = np.row_stack([neighbours[0], neighbours[1], neighbours[2]])
        plane = three_points_to_plane(three)
        normal = [plane[0], plane[1], plane[2], plane[3]]
        normals.append(normal)
    normals = np.array(normals)
    return normals

def read_to_points_bin(path: str):
    arr = np.fromfile(file=path, dtype=np.float32, sep="")
    number_of_points = len(arr)
    print(f"Number of points: {number_of_points}")
    assert(len(arr) % 3 == 0)
    arr = arr.reshape(int(number_of_points / 3), 3)
    return arr

def read_to_points_xyz(path: str):
    arr = []
    with open(path, "r") as file:
        for line in file:
            x, y, z = line.strip().split()
            arr.append([x, y, z])
    number_of_points = len(arr)
    print(f"Number of points: {number_of_points}")
    return np.array(arr, dtype=np.float32)

def read_to_points_xyz__(path: str):
    arr = []
    with open(path, "r") as file:
        for line in file:
            x, y, z, _, _, _ = line.strip().split()
            arr.append([x, y, z])
    number_of_points = len(arr)
    print(f"Number of points: {number_of_points}")
    return np.array(arr, dtype=np.float32)

def center_mass(point_cloud):
    return np.mean(point_cloud, axis=0)

def point_to_plane_distance(point, plane):
    A = plane[0]
    B = plane[1]
    C = plane[2]
    D = -plane[3]
    X = point[0]
    Y = point[1]
    Z = point[2]
    distance = np.abs(A*X+B*Y+C*Z+D)/np.sqrt(A*A+B*B+C*C)
    return distance

def plane_ransac(point_cloud_fixed_corresponding_points, point_cloud_moved, distance_threshold=100, iters=None):
    def point_equal(p0, p1):
        return (p0[0] == p1[0] or p0[1] == p1[2] or p1[1] == p1[2])
    p = 0.99 #probablity 99%
    e = 0.20 #Outlier ratio 20%
    if (iters == None):
        iters = int(np.round(np.log(1-p)/np.log(1-(e)**3)))
    print(f"Ransac iters: {iters}")
    all_pts = point_cloud_fixed_corresponding_points
    best_inliners_id = []
    best_plane = []
    for _ in range(iters):
        index_samples = random.sample(range(len(all_pts)), 3)
        while (
            point_equal(all_pts[index_samples[0]], all_pts[index_samples[1]]) or
            point_equal(all_pts[index_samples[0]], all_pts[index_samples[2]]) or
            point_equal(all_pts[index_samples[1]], all_pts[index_samples[2]])
        ):
            index_samples = random.sample(range(len(all_pts)), 3)

        samples = all_pts[index_samples]
        plane = three_points_to_plane(samples)
        inliners = []
        outliners = []
        for i in range(len(all_pts)):
            p = all_pts[i]
            p2p_distance = point_to_plane_distance(p, plane)
            #print(p2p_distance)
            if (p2p_distance < distance_threshold):
                inliners.append(i)
            else:
                outliners.append(i)
        if (len(inliners) > len(best_inliners_id)):
            best_inliners_id = inliners
            best_plane = plane
    print(f"Outliners: {len(outliners)}, inliners: {len(inliners)}")
    return all_pts[best_inliners_id], point_cloud_moved[best_inliners_id], np.array(best_plane)
