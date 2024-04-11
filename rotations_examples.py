import numpy as np
from mpv_abs import Mpv
from manifold_classes import SkewSymmetric, Rotation
from lane_risenfeld import modified_lr
from manifolds_plot import RotationsPlot
from manifold_hermite_via_parallel_transport import manifold_hermite


def geodesic(n: int = 4):
    p = Rotation(np.identity(3))
    v = np.array([[0, 1, 0], [-1, 0, 1], [0, -1, 0]])
    v = np.array([[0, 1, 1], [-1, 0, 1], [-1, -1, 0]])
    v = SkewSymmetric(v)
    v.normalize(deep=True)
    initial = Mpv(p, v)
    data = []
    for t in np.linspace(0, 10, n):
        q = initial.exp(t)
        data.append(initial.parallel_transport(q))
    return data


def curve(n: int = 4, unit_tangents: bool = True):
    data = []
    for theta in np.linspace(0, 2*np.pi, n)[:-1]:
        x = 0.5
        theta1 = theta
        theta2 = 0

        a = np.sin(theta1)*(1-x)**0.5
        b = np.cos(theta1)*(1-x)**0.5
        c = np.sin(theta2)*x**0.5
        d = np.cos(theta2)*x**0.5

        R = np.array([[1-2*(c**2+d**2), 2*(b*c-a*d), 2*(b*d+a*c)], [2*(b*c+a*d), 1-2*(b**2+d**2), 2*(c*d-a*b)],
                      [2*(b*d-a*c), 2*(c*d+a*b), 1-2*(b**2+c**2)]])
        R_dot = np.array([[0, -2*b*d-2*a*c, -2*a*d+2*b*c], [2*b*d-2*a*c, 4*a*b, -2*(b**2-a**2)],
                          [-2*a*d-2*b*c, 2*(b**2-a**2), 4*a*b]])
        V = np.matmul(np.linalg.inv(R), R_dot)
        data.append(Mpv(Rotation(R), SkewSymmetric(V), unit_tangent=unit_tangents))
    return data


# masks
A = (1/8)*np.array([[[48/25, -29/25], [29/50, 13/20]],
                    [[152/25, -31/25], [29/50, 277/100]],
                    [[152/25, 31/25], [-29/50, 277/100]],
                    [[48/25, 29/25], [-29/50, 13/20]]])
B = np.array([[[0, 0], [0, 0]],
              [[0.5, -1/8], [0.75, -1/8]],
              [[1, 0], [0, 0.5]],
              [[0.5, 1/8], [-0.75, -1/8]]])

data = curve(n=4)
true = curve(n=100)

f = modified_lr(data, 4, 1, periodic=True)

m = manifold_hermite(B, [-2, 1], curve(n=4, unit_tangents=False), 4)

plotter = RotationsPlot(approximations=[m], curve=true, data=data, colors=['red'],
                        plot_initial_vectors=True, plot_vectors=False)

# method = "multiple figures"
method = "triangles"
# method = "axes"
plotter.plot(method=method)  # choose one of the following methods: "triangles", "axes", "multiple figures"


