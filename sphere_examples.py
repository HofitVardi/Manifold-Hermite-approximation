import numpy as np
import pandas as pd
import copy
from mpv_abs import Mpv
from manifold_classes import EuclideanPoint, SpherePoint
from lane_risenfeld import modified_lr
from manifolds_plot import SpherePlot
from manifold_hermite_via_parallel_transport import manifold_hermite


def sinusoid(n: int = 9, epsilon: float = 0, unit_tangents: bool = True):
    data = []
    for t in np.linspace(epsilon, 2*np.pi+epsilon, n+1):
        alpha = 0.2
        m = 10
        x = 1 + alpha**2 * np.cos(m*t)**2
        point = [np.cos(t) / x**0.5,
                 np.sin(t) / x**0.5,
                 alpha * np.cos(m*t) / x**0.5]
        y = 0.5 * m * alpha**2 * np.sin(2*m*t)
        vector = [(-np.sin(t) * x + y * np.cos(t)) / x**1.5,
                  (np.cos(t) * x + y * np.sin(t)) / x**1.5,
                  (-m * alpha * np.sin(m*t) * x + y * alpha * np.cos(m*t)) / x**1.5]

        data.append(Mpv(SpherePoint(point), EuclideanPoint(vector), unit_tangent=unit_tangents))

    return data


def spiral(a: float = -10, b: float = 10, n: int = 8, unit_tangents: bool = True):
    data = []
    for t in np.linspace(a, b, n+1):
        alpha = 0.4
        point = [np.cos(t) / (1 + alpha ** 2 * t ** 2) ** 0.5, np.sin(t) / (1 + alpha ** 2 * t ** 2) ** 0.5,
                 alpha * t / (1 + alpha ** 2 * t ** 2) ** 0.5]
        vector = [(-np.sin(t) * (1 + alpha ** 2 * t ** 2) - t * alpha ** 2 * np.cos(t)) / (
                1 + alpha ** 2 * t ** 2) ** 1.5,
                  (np.cos(t) * (1 + alpha ** 2 * t ** 2) - t * alpha ** 2 * np.sin(t)) / (
                          1 + alpha ** 2 * t ** 2) ** 1.5,
                  (alpha * (1 + alpha ** 2 * t ** 2) - t ** 2 * alpha ** 3) / (1 + alpha ** 2 * t ** 2) ** 1.5]

        data.append(Mpv(SpherePoint(point), EuclideanPoint(vector), unit_tangent=unit_tangents))

    return data


def geodesic():
    p = SpherePoint([0, 0, 1])
    v = EuclideanPoint(np.array([0, 1, 0]))
    initial = Mpv(p, v)
    data = []
    n = 4
    for t in np.linspace(0, 7, n):
        q = initial.exp(t)
        data.append(initial.parallel_transport(q))
    return data


def compare_sinusoid():
    for n in [5, 10, 20, 40, 50]:
        epsilon = np.pi/20
        data = sinusoid(n=n, epsilon=epsilon, unit_tangents=True)

        f = modified_lr(data, 4, 1)

        B = np.array([[[0,0],[0,0]],[[0.5,-1/8],[0.75,-1/8]],[[1,0],[0,0.5]],[[0.5,1/8],[-0.75,-1/8]]])

        # data = sinusoid(n=n, epsilon=epsilon, unit_tangents=False)
        data = pd.Series(data).apply(lambda mpv: Mpv(mpv.point, mpv.vector, unit_tangent=False))
        data = list(data)

        m = manifold_hermite(B, [-2, 1], data[:-1], 4, periodic=True)

        plotter = SpherePlot(approximations=[sinusoid(n=1000), m, f],
                             data=data,
                             colors=['gray', 'red', 'black', 'gray'],
                             plot_vectors=False, plot_initial_vectors=True)

        plotter.plot()


def compare_spiral():
    for n in [9, 20, 40, 50]:
        data = spiral(a=-10-10*(n+60)/n-10, b=10+10*(n+60)/n+10, n=n+60, unit_tangents=True)
        f = modified_lr(data, 5, 1, periodic=True)
        k = int((len(f)-300)/2)
        f = f[k:-k]

        B = np.array(
            [[[0, 0], [0, 0]], [[0.5, -1 / 8], [0.75, -1 / 8]], [[1, 0], [0, 0.5]], [[0.5, 1 / 8], [-0.75, -1 / 8]]])

        data = spiral(n=n, unit_tangents=False)
        data = pd.Series(data).apply(lambda mpv: Mpv(mpv.point, mpv.vector, unit_tangent=False))
        data = list(data)
        new_data = []
        mpv = copy.copy(data[0])
        mpv.vector.point = np.zeros(mpv.vector.point.shape)
        for i in range(16):
            new_data.append(mpv)
        new_data = new_data + data

        mpv = copy.copy(data[-1])
        mpv.vector.point = np.zeros(mpv.vector.point.shape)
        for i in range(16):
            new_data.append(mpv)
        m = manifold_hermite(B, [-2, 1], new_data, 4, periodic=False)

        plotter = SpherePlot(approximations=[spiral(n=1000), f],
                             data=data,
                             colors=['gray', 'red', 'black', 'gray'],
                             plot_vectors=False, plot_initial_vectors=True)

        plotter.plot()


# compare_spiral()


def default_triangle(r: float = 0.1):
    a = np.array([r, 0, (1 - r ** 2) ** 0.5])
    b = np.array([r * np.cos(2 * np.pi / 3), r * np.sin(2 * np.pi / 3), (1 - r ** 2) ** 0.5])
    c = np.array([r * np.cos(4 * np.pi / 3), r * np.sin(4 * np.pi / 3), (1 - r ** 2) ** 0.5])
    vertices = np.array([a, b, c])
    return vertices


def unitary(theta):
    x = 0
    theta1 = 0
    theta2 = theta

    a = np.sin(theta1) * (1 - x) ** 0.5
    b = np.cos(theta1) * (1 - x) ** 0.5
    c = np.sin(theta2) * x ** 0.5
    d = np.cos(theta2) * x ** 0.5

    u = np.array([[1 - 2 * (c ** 2 + d ** 2), 2 * (b * c - a * d), 2 * (b * d + a * c)],
               [2 * (b * c + a * d), 1 - 2 * (b ** 2 + d ** 2), 2 * (c * d - a * b)],
               [2 * (b * d - a * c), 2 * (c * d + a * b), 1 - 2 * (b ** 2 + c ** 2)]])

    return u


t = default_triangle(r=0.5)
t_ = default_triangle(r=0.7)
t_ = np.matmul(unitary(np.pi/3), t_.T).T
a = Mpv(SpherePoint(t[0]), EuclideanPoint(np.array([0, 1, 0])), unit_tangent=True)
# a_ = Mpv(SpherePoint(t_[0]), EuclideanPoint(np.array([-1, 0, t_[0, 0]/t_[0, 2]])), unit_tangent=True)
b = Mpv(SpherePoint(t[1]), EuclideanPoint(np.array([t[1, 2]/t[1, 0], 0, -1])), unit_tangent=True)
# b_ = Mpv(SpherePoint(t_[1]), EuclideanPoint(np.array([1, 0, -t_[1, 0]/t_[1, 2]])), unit_tangent=True)
c = Mpv(SpherePoint(t[2]), EuclideanPoint(np.array([t[2, 2]/t[2, 0]/2, t[2, 2]/t[2, 1]/2, -1])), unit_tangent=True)
# b = Mpv(SpherePoint(t_[0]), EuclideanPoint(), unit_tangent=True)
# a_ = a.bezier_av(b)
# b_ = b.bezier_av(c)
# c_ = c.bezier_av(a)
data = [a, b, c]
# data_ = [a_, b_]
# data = data + data_
f1 = modified_lr(data, 8, 1, periodic=True)
f3 = modified_lr(data, 8, 3, periodic=True)
# f5 = modified_lr(data, 8, 5, periodic=True)
f10 = modified_lr(data, 8, 10, periodic=True)
plotter = SpherePlot(approximations=[f1, f3, f10],
                     data=data,
                     colors=['black', 'red', 'tab:green', 'gray'],
                     plot_vectors=False, plot_initial_vectors=True)
plotter.plot()
