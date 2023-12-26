import numpy as np
import matplotlib.pyplot as plt

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
        ax.scatter(xs=pc[:, 0], ys=pc[:, 1], zs=pc[:, 2], label=label)
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

def error_vector(p0, p1, fi, T):
    R = euler_angles_to_rotation_matrix(fi)
    return R @ p1 + T - p0

def h(p1, fi):
    j = jackobian_error_vector(p1, fi)
    return j.T @ j

def b(p0, p1, fi, T):
    j = jackobian_error_vector(p1, fi)
    e = error_vector(p0, p1, fi, T)
    return j.T @ e

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
    #MATRIX
    #[a,b,c]
    #[d,e,f]
    #[g,h,i]
    #R = np.array(
        #[
            #[
                #"a",
                #"b",
                #"c",
            #],
            #[
                #"d",
                #"e",
                #"f",
            #],
            #[
                #"g",
                #"h",
                #"i"
            #],
        #]
    #)
    return R

def jackobian_error_vector(p1, deg):
    a = deg[0]
    b = deg[1]
    c = deg[2]
    x = p1[0]
    y = p1[1]
    z = p1[2]

    #Original rotation matrix
    #r11 = np.cos(c) * np.cos(b)
    #r12 = np.sin(a) * np.sin(b) * np.cos(c) - np.cos(a) * np.sin(c)
    #r13 = np.cos(a) * np.sin(b) * np.cos(c) + np.sin(a) * np.sin(c)
    #r21 = np.cos(b) * np.sin(c)
    #r22 = np.sin(a) * np.sin(b) * np.sin(c) + np.cos(a) * np.cos(c)
    #r23 = np.cos(a) * np.sin(b) * np.sin(c) - np.sin(a) * np.cos(c)
    #r31 = -np.sin(b)
    #r32 = np.sin(a) * np.cos(b)
    #r33 = np.cos(a) * np.cos(b)

    #DERIVATIVES
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
        [1, 0, 0, df_da_0, df_db_0, df_dc_0],
        [0, 1, 0, df_da_1, df_db_1, df_dc_1],
        [0, 0, 1, df_da_2, df_db_2, df_dc_2]
    ))

    return j

def rad_to_deg(fi):
    return fi * 180 / np.pi

def deg_to_rad(fi):
    return fi * np.pi / 180

