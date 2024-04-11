import numpy as np
from numpy import array, sin, cos, matmul, real
from numpy.linalg import svd
from scipy.linalg import fractional_matrix_power as power
from mpv_abs import Mpv
from manifold_classes import PosDefMatrices, Symmetric
from manifolds_plot import PosDefPlot
from lane_risenfeld import modified_lr
from manifold_hermite_via_parallel_transport import manifold_hermite


def curve(a: float = 0, b: float = 1, n: int = 11, unit_tangents: bool = False):
    data = []
    A = array([[0, 1, 1], [1, 0, 1], [1, 1, 0]])
    u, s, v = svd(A)
    for t in np.linspace(a, b, n):
        s_t = array([[s[0]*(sin(t)**2+1), 0, 0], [0, s[1], 0], [0, 0, s[2]*(t+1)]])
        u_t = power(v, t)
        v_t = u_t.T
        p = matmul(u_t, matmul(s_t, v_t))
        p = real(p)

        S_dot = array([[s[0]*2*cos(t), 0, 0], [0, 0, 0], [0, 0, s[2]]])
        U_dot = t*power(v_t, t-1)
        V_dot = U_dot.T

        v_t = matmul(U_dot, matmul(s_t, v_t)) + matmul(u_t, matmul(S_dot, v_t)) + matmul(u_t, matmul(s_t, V_dot))
        v_t = Symmetric(real(v_t), project=True, basis=p)
        data.append(Mpv(PosDefMatrices(p), v_t, unit_tangent=unit_tangents))
    return data


def unitary(theta):
    x = 0.5
    theta1 = theta
    theta2 = 0

    a = sin(theta1) * (1 - x) ** 0.5
    b = cos(theta1) * (1 - x) ** 0.5
    c = sin(theta2) * x ** 0.5
    d = cos(theta2) * x ** 0.5

    u = array([[1 - 2 * (c ** 2 + d ** 2), 2 * (b * c - a * d), 2 * (b * d + a * c)],
               [2 * (b * c + a * d), 1 - 2 * (b ** 2 + d ** 2), 2 * (c * d - a * b)],
               [2 * (b * d - a * c), 2 * (c * d + a * b), 1 - 2 * (b ** 2 + c ** 2)]])

    u_dot = array([[0, -2 * b * d - 2 * a * c, -2 * a * d + 2 * b * c],
                   [2 * b * d - 2 * a * c, 4 * a * b, -2 * (b ** 2 - a ** 2)],
                   [-2 * a * d - 2 * b * c, 2 * (b ** 2 - a ** 2), 4 * a * b]])
    return u, u_dot


def spd(theta):
    u, u_dot = unitary(theta)
    s = np.identity(3)
    s[0, 0] = theta
    s_dot = np.identity(3)
    p = matmul(u, matmul(s, u.T))
    v = matmul(u_dot, matmul(s, u.T)) + matmul(u, matmul(s_dot, u.T)) + matmul(u, matmul(s, u_dot.T))
    return PosDefMatrices(p), Symmetric(v)


def symmetric(theta):
    u = unitary(theta)[0]
    s = np.identity(3)
    s[0, 0] = 1
    s[1, 1] = -1
    s[2, 2] = 1
    p = matmul(u, matmul(s, u.T))
    return Symmetric(p)


def geodesic_data():
    m0, v0 = spd(np.pi/8)
    m1, v1 = spd(np.pi/3)

    v0 = m0.log(other=m1, keep_length=True)
    v1 = m0.parallel_transport(vector=v0, other=m1)

    return [Mpv(m0, v0), Mpv(m1, v1)]


def paper_example():
    data = geodesic_data()
    data[0].vector = Symmetric(-np.identity(3))
    data[1] = data[0].parallel_transport(data[1].point)
    return data


def arbitrary_vectors():
    data = geodesic_data()
    data[0].vector = Symmetric(np.random.uniform(-1, 1, (3, 3)), project=True)
    data[1].vector = Symmetric(np.random.uniform(-1, 1, (3, 3)), project=True)
    return data


# data = geodesic_data()
data = paper_example()
# data = arbitrary_vectors()
# data = curve(a=1, b=5, n=2, unit_tangents=True)

for n in [5, 30]:
    bez = []
    for t in np.linspace(0, 1, n):
        bez.append(data[0].bezier_av(data[1], omega=t))

    f = modified_lr(data, 3, 1, periodic=False)

    plotter = PosDefPlot([bez, data])
    plotter.plot()
