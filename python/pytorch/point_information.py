import numpy as np

from arrow_3d import Arrow3D
import point_to_plane_utils_3d as utils

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
            self.ax = self.fig.add_subplot(1, 1, 1, projection="3d")

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
            print(x, y, z)
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

    def draw_normal(self):
        self.check()
        x, y, z = self.point_coordinates()
        a, b, c, d = self.point_normal()
        magnitude = np.sqrt(a**2 + b**2 + c**2)
        a, b, c = a / magnitude, b / magnitude, c / magnitude
        dot = np.dot([x, y, z], [a, b, c])
        if (dot > 0):
            arw = Arrow3D([x, x + a], [y, y + b], [z, z + c], arrowstyle="->", color="purple", lw = 0.5, mutation_scale=25)
        else:
            arw = Arrow3D([x, x - a], [y, y - b], [z, z - c], arrowstyle="->", color="purple", lw = 0.5, mutation_scale=25)
        self.ax.add_artist(arw)