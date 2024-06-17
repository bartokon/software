import numpy as np
import point_to_plane_utils_3d as utils
from scipy.ndimage import gaussian_filter
from scipy.spatial import KDTree
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import proj3d
from matplotlib.patches import FancyArrowPatch
from copy import deepcopy

class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        super().__init__((0,0), (0,0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def do_3d_projection(self, renderer=None):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
        self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))
        return np.min(zs)

#DRAW SQUARE
#SUBDIV SQUARE to N SQUARES
class point_information:
    def __init__(self, distance, index, pc_indexes):
        self.distance = distance
        self.index = index
        self.pc_indexes = pc_indexes
        self.average_point = np.average(pc_indexes, axis=0)
        self.fig = None
        self.ax = None

    def check(self):
        if (self.ax == None):
            self.ax = fig.add_subplot(1, 1, 1, projection="3d")

    def point_coordinates(self):
        return self.pc_indexes[0]

    def point_normal(self):
        a, b, c, d = utils.three_points_to_plane([self.pc_indexes[0], self.pc_indexes[1], self.pc_indexes[2]])
        return a, b, c, d

    def draw_plane(self):
        self.check()
        a, b, c, d = self.point_normal()
        x0, y0, z0 = self.point_coordinates()
        extent = max(self.distance)

        #When y is 0 cant find the plane... c is 0
        if (c != 0):
            xx, yy = np.meshgrid(np.linspace(x0-extent, x0+extent, 10, endpoint=True), np.linspace(y0-extent, y0+extent, 10, endpoint=True))
            z = (-a * xx - b * yy + d) / c
        elif (b != 0):
            xx, z = np.meshgrid(np.linspace(x0-extent, x0+extent, 10, endpoint=True), np.linspace(z0-extent, z0+extent, 10, endpoint=True))
            yy = (-a * xx - c * z + d) / b
        elif (a != 0):
            z, yy = np.meshgrid(np.linspace(x0-extent, x0+extent, 10, endpoint=True), np.linspace(y0-extent, y0+extent, 10, endpoint=True))
            xx = (-c * z - b * yy + d) / a
        else:
            print("No plane equation?")
            exit()

        self.ax.plot_surface(xx, yy, z, alpha=0.2)


    def draw_circle(self):
        self.check()

        #https://stackoverflow.com/questions/11140163/plotting-a-3d-cube-a-sphere-and-a-vector
        u, v = np.mgrid[0:2*np.pi:10j, 0:np.pi:10j]
        x = np.cos(u)*np.sin(v)
        y = np.sin(u)*np.sin(v)
        z = np.cos(v)
        radius = max(self.distance)
        x = radius * x
        y = radius * y
        z = radius * z
        self.ax.plot_wireframe(x + self.pc_indexes[0][0], y + self.pc_indexes[0][1], z + self.pc_indexes[0][2], color='r', linewidth=0.1)

    def draw_point(self):
        self.check()
        x, y, z = self.point_coordinates()
        self.ax.scatter(x, y, z)

    def get_neighbours(self):
        return np.array(self.pc_indexes)

    def draw_neighbours(self):
        self.check()
        a = []
        b = []
        c = []
        for i in range(1, len(self.pc_indexes)):
            x, y, z = self.pc_indexes[i]
            a.append(x)
            b.append(y)
            c.append(z)
        self.ax.scatter(a, b, c)

    def add_coordinates(self):
        self.check()
        x, y, z = self.point_coordinates()
        self.ax.text(x, y, z, '(%s,%s,%s)' % (str(round(x, 2)), str(round(y, 2)), str(round(z, 2))), color='k')

    def draw_arrows(self):
        self.check()
        x, y, z = self.point_coordinates()
        for i in range(1, len(self.pc_indexes)):
            a,b,c = self.pc_indexes[i]
            arw = Arrow3D([x, a], [y, b], [z,c], arrowstyle="->", color="purple", lw = 0.1, mutation_scale=25)
            self.ax.add_artist(arw)

    #???? is it any good?
    def draw_normal(self):
        self.check()
        x, y, z = self.point_coordinates()
        a, b, c, d = self.point_normal()
        magnitude = np.sqrt(a**2 + b**2 + c**2)
        a, b, c = a / magnitude, b / magnitude, c / magnitude
        dot = np.dot([x, y, z], [a, b, c])
        #print(dot)
        if (dot > 0):
            arw = Arrow3D([x, x + a], [y, y + b], [z, z + c], arrowstyle="->", color="purple", lw = 0.5, mutation_scale=25)
        else:
            arw = Arrow3D([x, x - a], [y, y - b], [z, z - c], arrowstyle="->", color="purple", lw = 0.5, mutation_scale=25)
        self.ax.add_artist(arw)

def SIFT_3D(pc, radius = 1, lim=0.1, min_points = 0):
    QUERY_POINTS = 256
    tree = KDTree(pc, balanced_tree=True)
    point_list = []

    for p in pc:
        distance, index = tree.query(p, k=QUERY_POINTS, distance_upper_bound=radius, workers=-1)
        index = index[distance != float('inf')]
        distance = distance[distance != float('inf')]

        if (len(index) <= min_points):
            continue

        point = point_information(distance, index, pc[index])
        avg = (np.var(point.get_neighbours(), axis=0))
        if ((avg[0] > lim) or (avg[1] > lim) or (avg[2] > lim)):
            point_list.append(point)

    if (len(point_list) == 0):
        print("LIST IS EMPTY")
        #exit(-1)

    return point_list

if __name__ == '__main__':

    pc_fixed = (np.random.rand(1000, 3) - 0.5) * 2
    #pc_fixed = utils.read_to_points_xyz("bunny_part1.xyz")[::1]
    #pc_fixed = utils.read_to_points_xyz("save_txt.txt")[::100]

    #pc_fixed = np.array([[np.sin(i), i, 0] for i in np.arange(start=-np.pi, stop=np.pi, step=0.005)])

    s0 = SIFT_3D(pc_fixed, radius=0.5, lim=0.065, min_points=3)

    fig = plt.figure(figsize=plt.figaspect(2.))
    ax = fig.add_subplot(1, 1, 1, projection="3d")
    print(len(s0))
    for p in s0:
        p.fig = fig
        p.ax = ax
        p.add_coordinates()
        p.draw_circle()
        p.draw_point()
        p.draw_neighbours()
        p.draw_arrows()
        p.draw_plane()
        p.draw_normal()
        break

    s0[0].ax.scatter(pc_fixed[:, 0], pc_fixed[:, 1], pc_fixed[:, 2], alpha=0.25)
    #x, y, z = s0[0].point_coordinates()
    #s0[0].ax.set_xlim(-10 + x, 10 + x)
    #s0[0].ax.set_ylim(-10 + y, 10 + y)
    #s0[0].ax.set_zlim(-10 + z, 10 + z)

    plt.show()
    exit()
