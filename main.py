import numpy as np
from scipy.interpolate import splprep, splev, BSpline
from geomdl import BSpline, utilities
from geomdl.fitting import interpolate_curve, interpolate_surface
from geomdl.construct import construct_surface
from geomdl.visualization import VisMPL
from copy import deepcopy


class Vertex:
    def __init__(self, loc):
        # location of vertex, array of shape (3,)
        self.loc = np.array(loc)


class Edge:
    all_edges = []

    def __init__(self, start, end, n):
        # starting point and ending point, array of shape (3,) OR Vertex object
        # n is the number of vertex points, int
        if type(start) is Vertex:
            self.start = start.loc
        else:
            self.start = np.array(start)

        if type(start) is Vertex:
            self.end = end.loc
        else:
            self.end = np.array(end)
        self.n = n
        self.start_tangent = None
        self.end_tangent = None
        self.points = None
        self.all_edges.append(self)

    def translate(self, v):
        self.points += v
        tmp = np.array(self.curve.ctrlpts) + v
        self.curve.ctrlpts = tmp.tolist()

    def plot(self, ax):
        ax.plot3D(self.points[:, 0], self.points[:, 1], self.points[:, 2])
        points = self.points

    def plot_all(self, ax):
        points = None
        for k in self.all_edges:
            k.plot(ax)
            if points is not None:
                points = np.append(points, k.points, axis=0)
            else:
                points = k.points
        xlim = (np.min(points[:, 0]), np.max(points[:, 0]))
        ax.set_xlim(xlim)
        ylim = (np.min(points[:, 1]), np.max(points[:, 1]))
        ax.set_ylim(ylim)
        zlim = (np.min(points[:, 2]), np.max(points[:, 2]))
        ax.set_zlim(zlim)
        ax.set_box_aspect([xlim[1] - xlim[0],
                           ylim[1] - ylim[0],
                           zlim[1] - zlim[0]])


class StraightEdge(Edge):
    def __init__(self, start, end, n=5):
        super(StraightEdge, self).__init__(start, end, n)
        self.start_tangent = self.end - self.start
        self.start_tangent = normalize(self.start_tangent)

        self.end_tangent = self.start_tangent

        # self.points = np.linspace(self.start, self.end, n)
        self.ccp = np.array([self.start, self.end])

        self.curve = BSpline.Curve()
        self.curve.degree = 1
        self.curve.ctrlpts = self.ccp.tolist()
        self.curve.knotvector = [0, 0, 1, 1]

        self.curve.delta = 1 / n
        self.points = np.array(self.curve.evalpts)


class ArcEdge(Edge):
    def __init__(self, start, end, center, n=10, direction=True):
        super(ArcEdge, self).__init__(start, end, n)
        # center of arc is array of shape (3,)
        self.center = np.array(center)
        self.angle = np.arccos(np.dot(normalize(self.start - self.center),
                                      normalize(self.end - self.center))
                               )

        self.normal = np.cross(self.start - self.center, self.end - self.center)
        self.normal = normalize(self.normal)
        # direction is boolean, True to draw the arc from start to end around center via the shorter route
        # if 2 routes have same length, direction=True to draw the arc via the route counter-clockwise about normal
        self.start_tangent = np.cross(self.normal, self.start - self.center)
        self.start_tangent = normalize(self.start_tangent)
        self.end_tangent = np.cross(self.normal, self.end - self.center)
        self.end_tangent = normalize(self.end_tangent)

        if np.dot(self.start - self.center,
                  self.end - self.center):  # the 2 vectors are not parallel, i.e. shorter route exist
            chord = self.end - self.start
            if (direction and np.dot(self.start_tangent, chord) < 0) or \
                    (not direction and np.dot(self.start_tangent, chord) > 0):
                self.start_tangent, self.end_tangent = -self.start_tangent, -self.end_tangent
                self.angle -= 2 * np.pi

        else:  # the 2 vectors are not parallel, i.e. 2 routes same length
            if not direction:
                self.start_tangent, self.end_tangent = -self.start_tangent, -self.end_tangent
                self.angle -= 2 * np.pi

        self.points = np.array([R(self.normal, theta) @ (self.start - self.center) + self.center for theta in
                                np.linspace(0, self.angle, n)])
        self.curve = interpolate_curve(self.points.tolist(), 1)
        self.curve.delta = 1 / n


class BSplineEdge(Edge):
    def __init__(self, ccp, knotvector=None, n=20, k=3):
        super(BSplineEdge, self).__init__(ccp[0], ccp[-1], n)
        # ccp is collection of control points, array of shape (int, 3)
        self.ccp = np.array(ccp)
        n_cp = self.ccp.shape[0]
        self.k = k

        if knotvector is None:
            self.curve = interpolate_curve(self.ccp.tolist(),
                                           self.k)
            self.knotvector = self.curve.knotvector
        else:
            self.curve = BSpline.Curve()
            self.curve.degree = self.k
            self.curve.ctrlpts = self.ccp.tolist()
            self.curve.knotvector = knotvector

            self.knotvector = knotvector

        self.curve.delta = 1 / n
        # Set evaluation delta
        self.points = np.array(self.curve.evalpts)
        pass


class CombinedEdge(BSplineEdge):
    def __init__(self, edges, k=3, ccp_as_point=True):
        ccp = None
        # knot_vectors = []
        new_n = 0
        for edge in edges:
            if issubclass(type(edge), Edge):
                new_n += edge.n
            elif type(edge) == list:
                new_n += len(edge)
        num_edges = len(edges)

        while edges:
            if issubclass(type(edges[0]), Edge):
                edge0 = edges[0].points.tolist()
                self.all_edges.remove(edges[0])
            elif type(edges[0]) == list:
                edge0 = edges[0]

            if ccp is None:
                if issubclass(type(edges[1]), Edge):
                    edge1 = edges[1].points.tolist()
                elif type(edges[1]) == list:
                    edge1 = edges[1]

                if edge0[-1] == edge1[0] or \
                        edge0[-1] == edge1[-1]:
                    ccp = edge0
                    # knot_vectors.append(edges[0].curve.knotvector)
                elif edge0[0] == edge1[0] or \
                        edge0[0] == edge1[-1]:
                    ccp = edge0[::-1]
                    # knot_vectors.append(edges[0].curve.knotvector[::-1])
                else:
                    raise ValueError("Not continuous curve")

            elif edge0[0] == ccp[-1]:
                ccp.extend(edge0[1:])
                # knot_vectors.append(edges[0].curve.knotvector)
            elif edge0[-1] == ccp[-1]:
                ccp.extend(edge0[-2::-1])
                # knot_vectors.append(edges[0].curve.knotvector[::-1])
            else:
                raise ValueError("Not continuous curve")
            edges.pop(0)

        super(CombinedEdge, self).__init__(ccp=ccp, n=new_n, k=k)

        if ccp_as_point is True:
            self.points = np.array(ccp)


class Surface:
    all_surfaces = []

    def __init__(self, Q0, Q1, P0, P1, nu=None, nv=None, k=3, symmetry=True, ccp_as_point=False):
        if nu is None:
            if issubclass(type(Q0), Edge):
                nu = len(Q0.points)
            elif type(Q0) == list:
                nu = len(Q0)
        if nv is None:
            if issubclass(type(P0), Edge):
                nv = len(P0.points)
            elif type(Q0) == list:
                nv = len(P0)

        self.cu = np.linspace(0, 1, nu)
        self.cv = np.linspace(0, 1, nv)
        self.k = k

        # Q0 to list of points, shape (-1, 3).
        if issubclass(type(Q0), Edge):
            if ccp_as_point:
                Q0 = Q0.points.tolist()
            else:
                Q0 = Q0.curve.evaluate_list(self.cu.tolist())
        elif type(Q0) == list and len(Q0) != nu:
            raise ValueError("Q0 have element number different from nu")

        # P0 to list of points, shape (-1, 3).
        if issubclass(type(P0), Edge):
            if ccp_as_point:
                P0 = P0.points.tolist()
            else:
                P0 = P0.curve.evaluate_list(self.cv.tolist())
        elif type(P0) == list and len(P0) != nv:
            raise ValueError("P0 have element number different from nv")

        # reverse P0 if P0 does not start at start point of Q0
        if P0[0] == Q0[0]:
            pass
        elif P0[-1] == Q0[0]:
            P0 = P0[::-1]
        elif P0[0] == Q0[-1]:
            Q0 = Q0[::-1]
        elif P0[-1] == Q0[-1]:
            Q0 = Q0[::-1]
            P0 = P0[::-1]
        else:
            raise ValueError("Wrong P0, does not connect to start point of Q0")

        # Q1 to list of points, shape (-1, 3).
        if issubclass(type(Q1), Edge):
            if ccp_as_point:
                Q1 = Q1.points.tolist()
            else:
                Q1 = Q1.curve.evaluate_list(self.cu.tolist())
        elif type(Q1) == list and len(Q1) != nu:
            raise ValueError("Q1 have element number different from nu")
        # reverse Q1 if Q1 does not start at end point of P0
        if Q1[0] == P0[-1]:
            pass
        elif Q1[-1] == P0[-1]:
            Q1 = Q1[::-1]
        else:
            raise ValueError("Wrong Q1, does not connect to end point of P0")

        # P1 to list of points, shape (-1, 3).
        if issubclass(type(P1), Edge):
            if ccp_as_point:
                P1 = P1.points.tolist()
            else:
                P1 = P1.curve.evaluate_list(self.cv.tolist())
        elif type(P1) == list and len(P1) != nv:
            raise ValueError("P1 have element number different from nv")
        # reverse P1 if P1 does not start at end point of Q0
        if P1[0] == Q0[-1]:
            pass
        elif P1[-1] == Q0[-1]:
            P1 = P1[::-1]
        else:
            raise ValueError("Wrong P1, does not connect to end point of Q0,\n"
                             f"P1 end points are {P1[0]} and {P1[1]}.\n"
                             f"Q0 end point are {Q0[-1]}.")

        # check if P1 and Q1 connect
        # if P1[-1] != Q1[-1]:
        #     raise ValueError("Not continuous edge loop")

        Q0 = np.array(Q0)
        Q1 = np.array(Q1)
        P0 = np.array(P0)
        P1 = np.array(P1)

        S1 = (1 - self.cu).reshape((-1, 1)) @ P0.T.reshape((3, 1, -1)) + \
             (self.cu).reshape((-1, 1)) @ P1.T.reshape((3, 1, -1))
        S1 = np.transpose(S1, (1, 2, 0))

        # S2 = (1 - self.cv) * Q0.T + self.cv * Q1.T
        # S2 = np.reshape(S2.T, (1, -1, 3))

        S2 = np.reshape(Q0.T, (3, -1, 1)) @ (1 - self.cv).reshape((1, -1)) + \
             np.reshape(Q1.T, (3, -1, 1)) @ (self.cv).reshape((1, -1))
        S2 = np.transpose(S2, (1, 2, 0))

        S3 = np.repeat(
            np.expand_dims(
                np.outer(1 - self.cu, 1 - self.cv), axis=2
            )
            , 3, axis=2) * Q0[0] \
             + np.repeat(
            np.expand_dims(
                np.outer(self.cu, 1 - self.cv), axis=2
            )
            , 3, axis=2) * Q0[-1] \
             + np.repeat(
            np.expand_dims(
                np.outer(1 - self.cu, self.cv), axis=2
            )
            , 3, axis=2) * Q1[0] \
             + np.repeat(
            np.expand_dims(
                np.outer(self.cu, self.cv), axis=2
            )
            , 3, axis=2) * Q1[-1]

        self.points = S1 + S2 - S3

        self.surface = interpolate_surface(self.points.reshape(-1, 3).tolist(),
                                           size_u=nu, size_v=nv, degree_u=k, degree_v=k)

        self.all_surfaces.append(self)
        if symmetry:
            symmetry_surface = deepcopy(self)
            symmetry_surface.points[:, :, 1] *= -1
            symmetry_surface.all_surfaces.append(symmetry_surface)

    def translate(self, v):
        self.points += v
        tmp = np.array(self.surface.ctrlpts) + v
        self.curve.ctrlpts = tmp.tolist()

    def plot(self, ax):
        ax.plot_surface(self.points[:, :, 0], self.points[:, :, 1], self.points[:, :, 2])

    def plot_all(self, ax):
        points = None
        for k in self.all_surfaces:
            k.plot(ax)
            current_points = k.points.reshape((-1, 3))
            if points is not None:
                points = np.append(points, current_points, axis=0)
            else:
                points = current_points
        xlim = (np.min(points[:, 0]), np.max(points[:, 0]))
        ax.set_xlim(xlim)
        ylim = (np.min(points[:, 1]), np.max(points[:, 1]))
        ax.set_ylim(ylim)
        zlim = (np.min(points[:, 2]), np.max(points[:, 2]))
        ax.set_zlim(zlim)
        ax.set_box_aspect([xlim[1] - xlim[0],
                           ylim[1] - ylim[0],
                           zlim[1] - zlim[0]])


def R(k, theta):
    kx, ky, kz = k[0], k[1], k[2]
    s, c = np.sin(theta), np.cos(theta)
    v = 1 - c
    return np.array(
        [
            [kx ** 2 * v + c, kx * ky * v - kz * s, kx * kz * v + ky * s],
            [kx * ky * v + kz * s, ky ** 2 * v + c, ky * kz * v - kx * s],
            [kx * kz * v - ky * s, ky * kz * v + kx * s, kz ** 2 * v + c]
        ]
    )


def normalize(v):
    return v / dist(v)


def dist(v):
    return np.sqrt(np.sum(v ** 2))


def point_in_rect(pt_x, pt_y, quad):
    min_x = quad[:, :, 0].min()
    max_x = quad[:, :, 0].max()
    min_y = quad[:, :, 1].min()
    max_y = quad[:, :, 1].max()

    x_grid = np.tile(pt_x, (pt_y.size, 1)).T
    y_grid = np.tile(pt_y, (pt_x.size, 1))

    bool = np.all([x_grid >= min_x,
                   x_grid < max_x,
                   y_grid >= min_y,
                   y_grid < max_y],
                  axis=0)
    return bool


def find_uv_in_quad(x, y, quad):
    x00 = quad[0, 0, 0]
    x01 = quad[0, 1, 0]
    x10 = quad[1, 0, 0]
    x11 = quad[1, 1, 0]
    y00 = quad[0, 0, 1]
    y01 = quad[0, 1, 1]
    y10 = quad[1, 0, 1]
    y11 = quad[1, 1, 1]

    kxu, kxv, kxuv, cx = x10 - x00, x01 - x00, x11 - x01 - (x10 - x00), x - x00
    kyu, kyv, kyuv, cy = y10 - y00, y01 - y00, y11 - y10 - (y01 - y00), y - y00

    # u = a + b * v
    if not check_between(kyuv, up=0, down=0):
        if not check_between(kyu * kxuv - kxu * kyuv, up=0, down=0):
            a = (cy * kxuv - cx * kyuv) / (kyu * kxuv - kxu * kyuv)
            b = -(kyv * kxuv - kxv * kyuv) / (kyu * kxuv - kxu * kyuv)

            v = quad_eq(b * kyuv, kyv + a * kyuv + b * kyu, a * kyu - cy)
        else:
            v = (cy * kxuv - cx * kyuv) / (kyv * kxuv - kxv * kyuv)
            u = (cy - v * kyv) / (kyu + v * kyuv)
            if check_between(u) and check_between(v):
                return u, v
            return None, None
    else:
        a = cy / kyu
        b = -kyv / kyu

        v = quad_eq(b * kxuv, kxv + a * kxuv + b * kxu, a * kxu - cx)

    if v is None:
        return None, None

    for v0 in v:
        u = a + b * v0
        if check_between(u):
            return u, v0
    return None, None


def check_between(a, up=1, down=0, eps=1e-10):
    if down - eps <= a <= up + eps:
        return True
    return False


def quad_eq(a, b, c):
    if a == 0:  # linear
        if b != 0:
            return [-c / b]
        return None
    mean = -b / 2 / a
    delta = b ** 2 - 4 * a * c

    if delta < 0:
        return None
    delta = np.sqrt(delta) / 2 / a

    roots = []

    root = mean - delta
    if 0 <= root <= 1:
        roots.append(root)
    root = mean + delta
    if 0 <= root <= 1:
        roots.append(root)

    if roots:
        return roots
    return None


# Just for testing
if __name__ == '__main__':
    from matplotlib import pyplot as plt

    straight = StraightEdge([0, 0, 0], [-1, 2, 1])
    arc = ArcEdge([-1, 2, 1], [2, -1, .5], [.2, .3, .4])
    ccp = np.array([
        [-1.0, 2.0, 1.0],
        [0.9, 0.6, 0.],
        [1.2, 1.2, 0.9],
        [1.8, 1.5, 1.5],
        [0.4, 0.3, 1.6],
        [1.6, 0.5, 1.3]
    ])
    bline = BSplineEdge(ccp, n=40, k=3)

    fig = plt.figure()
    ax = plt.axes(projection='3d')

    # straight.plot(ax)
    # arc.plot(ax)
    bline.plot(ax)

    ax.scatter(ccp[:, 0], ccp[:, 1], ccp[:, 2])

    plt.show()
    pass
