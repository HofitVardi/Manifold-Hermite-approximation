from typing import Union
from abc import ABC, abstractmethod
import copy
import pandas as pd
from numpy import array, cos, arccos, sqrt


class ManifoldPoint(ABC):
    def __init__(self, point: array):
        self.point = point
        self.manifold = None
        self.dim = None
        self.unit_tangents = False

    @abstractmethod
    def __copy__(self):
        raise NotImplementedError

    @abstractmethod
    def is_tangent(self, vector):
        raise NotImplementedError

    def distance(self, other):
        return self.log(other, keep_length=True).norm()

    @abstractmethod
    def exp(self, vector: array):
        raise NotImplementedError

    @abstractmethod
    def log(self, other, keep_length: bool = False):
        raise NotImplementedError

    def mean(self, other, weight: float = 0.5):
        if not isinstance(other, type(self)) or self.dim != other.dim:
            raise TypeError("objects must be of the same type")
        return self.exp(self.log(other=other, keep_length=True), alpha=weight)

    @abstractmethod
    def parallel_transport(self, vector, other):
        raise NotImplementedError


class Vector(ManifoldPoint, ABC):
    def __init__(self, vector: array):
        super.__init__(vector)

    @abstractmethod
    def inner_product(self, other):
        raise NotImplementedError

    def angular_distance(self, other, tol: float = 10**(-14)):
        product = self.inner_product(other)
        n = self.norm() * other.norm()
        if n < tol:
            raise ValueError("vectors are too small")
        return arccos(min(1, max(-1, product/n)))

    def norm(self):
        return sqrt(self.inner_product(self))

    def normalize(self, deep=False, tol: float = 10 * (-12)):
        if self.norm() < tol:
            raise ValueError("vector's norm is too small")
        normalized = self.point / self.norm()
        if deep:
            self.point = normalized
        return normalized


class Mpv:
    def __init__(self, point: Union[ManifoldPoint, None] = None, vector: Union[Vector, None] = None,
                 unit_tangent: bool = True):
        self.point = point
        self.unit_tangent = unit_tangent
        if not self.point.is_tangent(vector):
            raise ValueError(f"vector must be tangent to {self.point.manifold} at the given point")
        if unit_tangent:
            vector.normalize(deep=True)
        self.vector = vector
        self.manifold = point.manifold

    def __copy__(self):
        return Mpv(copy.copy(self.point), copy.copy(self.vector), self.unit_tangent)

    def points_distance(self, other):
        return self.point.distance(other.point)

    def vectors_distance(self, other):
        transported_vector = other.point.parallel_transport(other.vector, self.point)
        return self.vector.angular_distance(transported_vector)

    def exp(self, alpha: float = 1):
        if self.point is None or self.vector is None:
            raise TypeError("point and vector must initialized")
        return self.point.exp(self.vector, alpha)

    def parallel_transport(self, other):
        if isinstance(other, Mpv):
            point = other.point
        else:
            point = other
        vector = self.point.parallel_transport(copy.copy(self.vector), point)
        return Mpv(point=copy.copy(point), vector=copy.copy(vector), unit_tangent=self.unit_tangent)

    def bezier_av(self, other, omega: float = 0.5):
        difference_vector = self.point.log(other.point)
        new_mpv = Mpv(copy.copy(self.point), difference_vector)
        theta_0 = self.vectors_distance(new_mpv)
        theta_1 = other.vectors_distance(new_mpv)
        alpha = self.points_distance(other) / (3 * cos((theta_0 + theta_1) / 4) ** 2)
        p0 = copy.copy(self.point)
        p1 = self.exp(alpha)
        p2 = other.exp(-alpha)
        p3 = copy.copy(other.point)
        points = [p0, p1, p2, p3]
        while len(points) > 1:
            new_points = []
            for i in range(len(points) - 1):
                new_points.append(points[i].mean(other=points[i + 1], weight=omega))
            points = new_points
            if len(points) == 2:
                direction = points[1]
        point = points[0]
        vector = point.log(other=direction)
        return Mpv(point, vector, unit_tangent=self.unit_tangent)
