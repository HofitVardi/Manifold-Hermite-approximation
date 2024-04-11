from abc import ABC
from typing import Union
from mpv_abs import ManifoldPoint, Vector
import numpy as np
from numpy import array
from numpy.linalg import norm, inv
from numpy import dot, cos, sin, arccos, matmul
from scipy.linalg import expm, logm, sqrtm
from pymanopt.tools.multi import multilog, multiprod, multisym, multitransp
import copy


class EuclideanPoint(Vector):
    def __init__(self, point: array):
        if isinstance(point, list):
            point = array(point)
        if len(point.shape) != 1:
            raise ValueError("point must be a one dimensional array")
        # super.__init__(point)
        self.point = point
        self.manifold = "Euclidean"
        self.dim = len(point)

    def __copy__(self):
        return EuclideanPoint(self.point)

    def distance(self, other):
        return norm(self.point - other.point)

    def inner_product(self, other):
        return dot(self.point, other.point)

    def is_tangent(self, vector: Vector):
        if isinstance(vector, EuclideanPoint):
            return self.dim == vector.dim
        return False

    def exp(self, vector, alpha: float = 1):
        if not self.is_tangent(vector):
            raise ValueError("vector must be in the tangent space of the given point")
        return EuclideanPoint(self.point + alpha * vector.point)

    def log(self, other, keep_length: bool = False):
        if not isinstance(other, EuclideanPoint) or self.dim != other.dim:
            raise TypeError("objects must be of the same type")
        vector = EuclideanPoint(other.point - self.point)
        if not keep_length:
            vector.normalize(deep=True)
        return vector

    def mean(self, other, weight: float = 0.5):
        if not isinstance(other, EuclideanPoint) or self.dim != other.dim:
            raise TypeError("objects must be of the same type")
        return EuclideanPoint(weight * self.point + (1 - weight) * other.point)

    def parallel_transport(self, vector, other=None):
        return vector


class SpherePoint(ManifoldPoint):
    def __init__(self, point: array, project: bool = False, tol: float = 10**(-8)):
        if isinstance(point, list):
            point = array(point)
        if len(point.shape) != 1:
            raise ValueError("point must be a one dimensional array")
        # super.__init__(point)
        self.point = point
        self.manifold = "sphere"
        self.dim = len(point) - 1
        if project or abs(norm(point) - 1) < tol:
            self.project()
        else:
            raise ValueError("point must lie on the unit sphere")

    def __copy__(self):
        return SpherePoint(self.point)

    def project(self):
        length = norm(self.point)
        if length == 0:
            raise ValueError("point must be different than zero")
        self.point = self.point/length

    def distance(self, other):
        d = dot(self.point, other.point)
        return arccos(min(1, max(-1, d)))

    def is_tangent(self, vector: EuclideanPoint, tol: float = 10**(-10)):
        if isinstance(vector, EuclideanPoint) and vector.dim == self.dim + 1:
            return abs(dot(self.point, vector.point)) < tol
        return False

    def exp(self, vector, alpha: float = 1, tol: float = 10**(-8)):
        if not self.is_tangent(vector):
            raise ValueError("vector must be in the tangent space of the given point")
        vector = alpha * vector.point
        length = norm(vector)
        if length < tol:
            return self
        return SpherePoint(cos(length) * self.point + sin(length) * vector/length)

    def log(self, other, keep_length: bool = False, tol: float = 10**(-8)) -> array:
        if not isinstance(other, SpherePoint) or self.dim != other.dim:
            raise TypeError("objects must be of the same type")
        if norm(other.point - self.point) < tol:
            return EuclideanPoint(np.zeros(self.dim + 1))
        c = dot(self.point, other.point)
        c = max(c, -1)
        c = min(c, 1)
        v = other.point - c * self.point
        vector = EuclideanPoint(arccos(c) * v / norm(v))
        if not keep_length:
            vector.normalize(deep=True)
        return vector

    def parallel_transport(self, vector, other, tol: float = 10**(-8)):
        if not self.is_tangent(vector):
            raise ValueError(f"vector must be tangent to {self.point.manifold} at the given point")
        if norm(self.point - other.point) < tol:
            return vector
        u = self.log(other).point
        w = np.cross(u, self.point)  # the perpendicular vector to p and u
        A = np.array([[u[0], w[0]], [u[1], w[1]], [u[2], w[2]]])
        coef = np.matmul(np.linalg.pinv(A), vector.point)
        vector = coef[0] * (- other.log(self).point) + coef[1] * w
        return EuclideanPoint(vector)


class SkewSymmetric(Vector):
    def __init__(self, point: array, tol: float = 10**(-8)):
        if isinstance(point, list):
            point = array(point)
        if len(point.shape) != 2:
            raise ValueError("point must be two dimensional array")
        if (point + np.transpose(point) > tol).any():
            raise ValueError("point must be a skew symmetric matrix")
        # super.__init__(point)
        self.point = point
        self.manifold = "skew symmetric matrices"
        n = point.shape[0]
        self.dim = n*(n-1)/2

    def __copy__(self):
        return SkewSymmetric(self.point)

    def distance(self, other):
        raise NotImplementedError

    def inner_product(self, other):
        product = matmul(np.transpose(self.point), other.point)
        return np.trace(product)

    def is_tangent(self, vector: Vector):
        if isinstance(vector, SkewSymmetric):
            return self.dim == vector.dim
        return False

    def exp(self, vector, alpha: float = 1):
        raise NotImplementedError

    def log(self, other, keep_length: bool = False):
        raise NotImplementedError

    def parallel_transport(self, vector, other=None):
        return vector


class Rotation(ManifoldPoint):
    def __init__(self, point: array, tol: float = 10**(-10)):
        if isinstance(point, list):
            point = array(point)
        if len(point.shape) != 2:
            raise ValueError("point must be two dimensional array")
        if abs(np.linalg.det(point) - 1) > tol:
            raise ValueError("point must be a rotation matrix")
        # super.__init__(point)
        self.point = point
        self.manifold = "rotations"
        n = point.shape[0]
        self.dim = n*(n-1)/2

    def __copy__(self):
        return Rotation(self.point)

    # todo: verify
    def distance(self, other):
        return self.log(other, keep_length=True).norm()

    def is_tangent(self, vector: Vector):
        if isinstance(vector, SkewSymmetric):
            return self.dim == vector.dim
        return False

    def exp(self, vector, alpha: float = 1):
        if not self.is_tangent(vector):
            raise ValueError("vector must be in the tangent space of the given point")
        return Rotation(matmul(self.point, expm(alpha * vector.point)))

    def log(self, other, keep_length: bool = False):
        if not isinstance(other, Rotation) or self.dim != other.dim:
            raise TypeError("objects must be of the same type")
        inverse = inv(self.point)
        vector = logm(matmul(inverse, other.point))
        vector = SkewSymmetric(vector)
        if not keep_length:
            vector.normalize(deep=True)
        return vector

    def parallel_transport(self, vector, other):
        return vector


class Symmetric(Vector):
    def __init__(self, point: array, tol: float = 10**(-10), project: bool = False,
                 basis: Union[array, None] = None):
        if isinstance(point, list):
            point = array(point)
        if len(point.shape) != 2:
            raise ValueError("point must be two dimensional array")
        if (point - np.transpose(point) > tol).any() and not project:
            raise ValueError("point must be a symmetric matrix")
        # super.__init__(point)
        self.point = self.project(point)
        self.manifold = "symmetric matrices"
        n = point.shape[0]
        self.shape = (n, n)
        self.dim = n*(n+1)/2
        if basis is None:
            basis = np.identity(n)
        self.basis = basis

    def __copy__(self):
        return Symmetric(self.point)

    @staticmethod
    def project(point):
        if np.isnan(point).sum() > 0:
            raise ValueError("point must not contain nan values")
        return 0.5 * (point + point.T)

    def inner_product(self, other):
        xinvu = np.linalg.solve(self.basis, self.point)
        if self.point is other.point:
            xinvv = xinvu
        else:
            xinvv = np.linalg.solve(self.basis, other.point)
        return np.trace(matmul(xinvu, xinvv))

    def angular_distance(self, other, tol: float = 10**(-14)):
        product = self.inner_product(other)
        n = self.norm() * other.norm()
        if n < tol:
            raise ValueError("vectors are too small")
        return arccos(min(1, max(-1, product/n)))

    # todo: verify
    def is_tangent(self, vector: Vector):
        if isinstance(vector, Symmetric):
            return self.dim == vector.dim
        return False

    def exp(self, vector, alpha: float = 1):
        raise NotImplementedError

    def log(self, other, keep_length: bool = False):
        raise NotImplementedError

    def parallel_transport(self, vector, other):
        vector = copy.copy(vector)
        vector.basis = other.point
        return vector


class PosDefMatrices(ManifoldPoint):
    def __init__(self, point: array, tol: float = 10**(-10), project: bool = False):
        # todo: add positivity condition
        if isinstance(point, list):
            point = array(point)
        if len(point.shape) != 2:
            raise ValueError("point must be two dimensional array")
        if (point - np.transpose(point) > tol).any():
            raise ValueError("point must be a symmetric matrix")
        # super.__init__(point)
        self.point = point
        self.manifold = "symmetric matrices"
        n = point.shape[0]
        self.shape = (n, n)
        self.dim = n*(n+1)/2
        self.manifold = "positive definite matrices"

    def __copy__(self):
        return PosDefMatrices(self.point)

    # todo: verify
    def is_tangent(self, vector: Vector):
        if isinstance(vector, Symmetric):
            return self.dim == vector.dim
        return False

    def exp(self, vector: Symmetric, alpha: float = 1):
        if not self.is_tangent(vector):
            raise ValueError("vector must be in the tangent space of the given point")
        x_inv_u = np.linalg.solve(self.point, alpha * vector.point)
        ex = np.real(expm(x_inv_u))
        sym = Symmetric(matmul(self.point, ex), project=True)
        try:
            PosDefMatrices(sym.point)
        except:
            print()
        return PosDefMatrices(sym.point)

    # todo: verify direction
    def log(self, other: ManifoldPoint, keep_length: bool = False):
        if not isinstance(other, PosDefMatrices) or self.dim != other.dim:
            raise TypeError("objects must be of the same type")
        x_inv_y = np.linalg.solve(self.point, other.point)
        lo = np.real(logm(x_inv_y))
        sym = Symmetric(matmul(self.point, lo), project=True, basis=self.point)
        if not keep_length:
            sym.normalize(deep=True)
        return sym

    def parallel_transport(self, vector: Symmetric, other: ManifoldPoint):
        E = matmul(other.point, np.linalg.inv(self.point))
        E = np.real(sqrtm(E))
        vector = matmul(E, matmul(vector.point, E.T))
        vector = Symmetric(vector, project=True, basis=other.point)
        return vector
