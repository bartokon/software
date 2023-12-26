import numpy as np
import matplotlib.pyplot as plt

def plot_2d_point_clouds(*args):
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)
    plt.axhline(y=0, color='black', linestyle='-')
    plt.axvline(x=0, color='black', linestyle='-')
    for pc, label, marker, s in args:
        ax.scatter(x=pc[:, 0], y=pc[:, 1], label=label, marker=marker, s=s)
    plt.legend()
    plt.show()

def create_r(fi):
    R = np.array((
        [np.cos(fi), -np.sin(fi)],
        [np.sin(fi), np.cos(fi)]
    ))
    return R

def distance_2d(p0, p1):
    return np.sqrt((p0[0]-p1[0])**2 + (p0[1]-p1[1])**2)

def sum_distance_2d(pc0, pc1):
    sum = 0
    for p0, p1 in zip(pc0, pc1):
        sum += distance_2d(p0, p1)
    return sum

def rotate_pc_2d(pc, fi):
    R = create_r(fi)
    return np.array([R @ P for P in pc])

def translate_pc_2d(pc, T):
    return np.array([p + T for p in pc])

def error_vector(p0, p1, fi, T):
    R = create_r(fi)
    return R @ p1 + T - p0

def jackobian_error_vector(p1, fi):
    j = np.array([
        [1, 0, -np.sin(fi) * p1[0] - np.cos(fi) * p1[1]],
        [0, 1, np.cos(fi) * p1[0] - np.sin(fi) * p1[1]]
    ])
    return j

def h(p1, fi):
    j = jackobian_error_vector(p1, fi)
    return j.T @ j

def b(p0, p1, fi, T):
    j = jackobian_error_vector(p1, fi)
    e = error_vector(p0, p1, fi, T)
    return j.T @ e

def unified(p0, p1, fi, T):
    def u(p0, p1, fi, T):
        sin_fi = np.sin(fi)
        cos_fi = np.cos(fi)
        R = np.array((
            [cos_fi, -sin_fi],
            [sin_fi, cos_fi]
        ))
        sin_fi_p10 = sin_fi * p1[0]
        cos_fi_p11 = cos_fi * p1[1]
        cos_fi_p10 = cos_fi * p1[0]
        sin_fi_p11 = sin_fi * p1[1]
        n_sin_fi_p10_n_fi_p11 = -sin_fi_p10 - cos_fi_p11
        cos_fi_p10_n_sin_fi_p11 = cos_fi_p10 - sin_fi_p11
        j = np.array([
            [1, 0, n_sin_fi_p10_n_fi_p11],
            [0, 1, cos_fi_p10_n_sin_fi_p11]
        ])
        e = R @ p1 + T - p0
        return e, j, j.T

    e, j, j_t = u(p0, p1, fi, T)
    h = j_t @ j
    b = j_t @ e
    return h, b
