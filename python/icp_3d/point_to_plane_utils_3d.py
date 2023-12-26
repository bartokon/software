import numpy as np
import matplotlib.pyplot as plt
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

def distance_3d(p0, p1):
    return np.sqrt((p0[0]-p1[0])**2 + (p0[1]-p1[1])**2 + (p0[2]-p1[2])**2)

def sum_distance_3d(pc0, pc1):
    sum = 0
    for p0, p1 in zip(pc0, pc1):
        sum += distance_3d(p0, p1)
    return sum

def rotate_pc_3d(pc, fi):
    R = euler_angles_to_rotation_matrix(fi)
    return np.array([R @ P for P in pc])

def translate_pc_3d(pc, T):
    return np.array([p + T for p in pc])

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
    cp = np.cross(v1, v2)
    a, b, c = cp
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
