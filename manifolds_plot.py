import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
from mpv_abs import Mpv
from manifold_classes import SpherePoint, EuclideanPoint
from typing import Union


class ManifoldPlot(ABC):
    def __init__(self, approximations: list, curve: pd.Series = [], data: pd.Series = [], colors: list = [],
                 plot_initial_vectors: bool = True, plot_vectors: bool = True):
        self.manifold = None
        self.plot_initial_vectors = plot_initial_vectors
        self.plot_vectors = plot_vectors
        self.curve = Approximation(curve, kind="true")
        self.data = Approximation(data, kind="data")
        self.approximations = approximations
        self.set_approximations()
        self.colors = colors[:len(self.approximations) + 1]

    def set_approximations(self):
        try:
            self.approximations = pd.Series(self.approximations)\
                .apply(lambda approximation: Approximation(approximation))
        except:
            raise TypeError("approximations must be a series of series of Mpv objects")

    def plot(self, title: str = "", save: bool = False):
        raise NotImplementedError


class Approximation:
    def __init__(self, f: pd.Series, kind: str = ""):
        self.series = f
        self.kind = kind
        if not self.check_series():
            raise TypeError("f must be a series of Mpv objects")

    def check_series(self):
        if isinstance(self.series, list):
            self.series = pd.Series(self.series)
        if not isinstance(self.series, pd.Series):
            return 0
        return self.series.apply(lambda mpv: isinstance(mpv, Mpv)).all()

    def get_points(self):
        return self.series.apply(lambda x: x.point.point)

    def get_vectors(self):
        return self.series.apply(lambda x: x.vector.point)


class Euclidean2DPlot(ManifoldPlot):
    def __init__(self, approximations: list, curve: pd.Series = [], data: pd.Series = [], colors: list = [],
                 plot_initial_vectors: bool = True, plot_vectors: bool = True):
        super().__init__(approximations, curve, data, colors, plot_initial_vectors, plot_vectors)
        self.manifold = "Euclidean"

    def plot(self, title: str = "", save: bool = False):

        # plot series
        self.approximations['data'] = self.data
        colors = self.colors + ['black'] * (len(self.approximations) + 1 - len(self.colors))

        plot_vectors = self.plot_vectors
        for approximation, color in zip(self.approximations, colors):
            points = approximation.get_points()
            vectors = approximation.get_vectors()
            if approximation.kind == "data":
                plot_vectors = self.plot_initial_vectors

            if approximation.kind != "data":
                p = points.apply(lambda x: pd.Series(x))
                # plt.plot(p.iloc[:, 0], p.iloc[:, 1], color=color)
                plt.scatter(p.iloc[:, 0], p.iloc[:, 1], color=color, marker=".")
            else:
                for i in range(len(points)):
                    p = points.iloc[i]
                    plt.scatter(p[0], p[1], marker="o", color=color)

            if plot_vectors:
                for i in range(len(points)):
                    p = points.iloc[i]
                    v = vectors.iloc[i]
                    plt.quiver(p[0], p[1], v[0], v[1], color=color, scale=6, width=0.002)

        if save:
            plt.savefig(title + ".pdf")
        plt.title(title)
        plt.axis('equal')
        # plt.xlim(-0.5, 2)
        plt.axis('off')
        plt.show()
        return


class Euclidean3DPlot(ManifoldPlot):
    def __init__(self, approximations: list, curve: pd.Series = [], data: pd.Series = [], colors: list = [],
                 plot_initial_vectors: bool = True, plot_vectors: bool = True):
        super().__init__(approximations, curve, data, colors, plot_initial_vectors, plot_vectors)
        self.manifold = "Euclidean"

    def plot(self, title: str = "", save: bool = False):

        ax = plt.axes(projection='3d')

        # plot series
        self.approximations['data'] = self.data
        colors = self.colors + ['black'] * (len(self.approximations) + 1 - len(self.colors))

        plot_vectors = self.plot_vectors
        for approximation, color in zip(self.approximations, colors):
            points = approximation.get_points()
            vectors = approximation.get_vectors()
            if approximation.kind == "data":
                plot_vectors = self.plot_initial_vectors

            transparency = points.apply(lambda p: (p[0] ** 2 + p[1] ** 2) ** 0.5).max()

            for i in range(len(points)):
                p = points.iloc[i]
                t = 0.5 * ((p[0] ** 2 + p[1] ** 2) ** 0.5) / transparency
                if self.plot_vectors and approximation.kind == "data":
                    t = "purple"
                ax.scatter(p[0], p[1], p[2], marker="o", color=str(t))
                if plot_vectors:
                    v = vectors.iloc[i]
                    ax.quiver(p[0], p[1], p[2], v[0], v[1], v[2], color=str(t))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        if save:
            plt.savefig(title + ".pdf")
        ax.set_title(title)
        plt.show()
        return


class SpherePlot(ManifoldPlot):
    def __init__(self, approximations: list, curve: pd.Series = [], data: pd.Series = [],  colors: list = [],
                 plot_initial_vectors: bool = True, plot_vectors: bool = True):
        super().__init__(approximations, curve, data, colors, plot_initial_vectors, plot_vectors)
        self.manifold = "sphere"

    def plot(self, title: str = "", save: bool = False):
        # plot sphere
        ax = plt.axes(projection='3d')
        u = np.linspace(0, 4 * np.pi / 2, 100)
        v = np.linspace(0, 2 * np.pi / 2, 100)
        a = np.outer(np.cos(u), np.sin(v))
        b = np.outer(np.sin(u), np.sin(v))
        c = np.outer(np.ones(100), np.cos(v))
        ax.plot_surface(a, b, c, color="moccasin", alpha=0.3)

        # plot series
        colors = self.colors + ['gray'] * (len(self.approximations) + 2 - len(self.colors))
        self.approximations['true'] = self.curve
        self.approximations['data'] = self.data
        plot_vectors = self.plot_vectors
        for approximation, color in zip(self.approximations, colors):
            points = approximation.get_points()
            vectors = approximation.get_vectors()
            if approximation.kind == "data":
                plot_vectors = self.plot_initial_vectors

            if len(points) > 0:
                if approximation.kind != "data":
                    p = points.apply(lambda x: pd.Series(x))
                    ax.plot(p.iloc[:, 0], p.iloc[:, 1], p.iloc[:, 2], color=color)
                    # ax.scatter(p.iloc[:, 0], p.iloc[:, 1], p.iloc[:, 2], color=color)
                else:
                    for i in range(len(points)):
                        p = points.iloc[i]
                        ax.scatter(p[0], p[1], p[2], marker="o", color=color)
                # transparency = points.apply(lambda p: (p[0] ** 2 + p[1] ** 2) ** 0.5).max()

                for i in range(len(points)):
                    p = points.iloc[i]
                    # t = 0.5 * ((p[0] ** 2 + p[1] ** 2) ** 0.5) / transparency
                    t = 'black'
                    if self.plot_vectors and approximation.kind == "data":
                        t = "purple"
                    # ax.scatter(p[0], p[1], p[2], marker="o", color=str(t))
                    if plot_vectors:
                        v = vectors.iloc[i]
                        v = v/np.linalg.norm(v)/3
                        ax.quiver(p[0], p[1], p[2], v[0], v[1], v[2], color=color)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        if save:
            plt.savefig(title + ".pdf")
        ax.set_title(title)
        ax.set_axis_off()
        plt.show()
        return


class RotationsPlot(ManifoldPlot):
    def __init__(self, approximations: list, curve: pd.Series = [], data: pd.Series = [],  colors: list = [],
                 plot_initial_vectors: bool = True, plot_vectors: bool = True):
        super().__init__(approximations=approximations, curve=curve, data=data, colors=colors,
                         plot_initial_vectors=plot_initial_vectors, plot_vectors=plot_vectors)
        self.manifold = "rotations"

    def plot(self, method: str = "triangles", title: str = "", save: bool = False):
        if method == "triangles":
            self.triangles_plot(title=title, save=save)
        if method == "axes":
            self.axes_plot(title=title, save=save)
        if method == "multiple figures":
            if self.plot_vectors:
                self.multiple_figures(title=title, save=save)
            else:
                self.triangles_plot(title=title, save=save)
        return

    def triangles_plot(self, title: str = "", save: bool = False):
        # plot sphere
        ax = plt.axes(projection='3d')
        a, b, c = self.sphere_coordinates()
        ax.plot_surface(a, b, c, color="palegoldenrod", alpha=0.3)

        self.approximations['true'] = self.curve
        self.approximations['data'] = self.data
        colors = self.colors + ['gray'] * (len(self.approximations) - len(self.colors))
        plot_vectors = self.plot_vectors

        vertices = self.default_triangle()
        triangle = self.triangle(vertices)

        # plot approximations
        for approximation, color in zip(self.approximations, colors):
            if approximation.kind == "data":
                plot_vectors = self.plot_initial_vectors

            points = approximation.get_points()
            vectors = approximation.get_vectors()
            vectors = pd.Series(np.arange(0, len(points))).apply(lambda i: np.matmul(points.iloc[i], vectors.iloc[i]))

            points_actions = points.apply(lambda p: np.matmul(p, vertices.T).T)
            vectors_actions = vectors.apply(lambda v: np.matmul(v, vertices.T).T)

            if approximation.kind != "true":
                for i in range(len(points)):
                    # plot_triangles
                    t = np.matmul(points.iloc[i], triangle.T).T
                    ax.plot(t[:, 0], t[:, 1], t[:, 2], linewidth=0.75, color=color)

                    if plot_vectors:
                        p = points_actions.iloc[i]
                        v = vectors_actions.iloc[i]
                        length = 0.5
                        for k in range(3):
                            if not SpherePoint(p[k]).is_tangent(EuclideanPoint(v[k])):
                                raise ValueError("vectors action is not tangent to the sphere")

                        for k in range(3):
                            ax.quiver(p[k, 0], p[k, 1], p[k, 2],
                                      v[k, 0], v[k, 1], v[k, 2],
                                      length=length, color=color, linewidths=0.7)
            else:
                for k in range(3):
                    p = points_actions.apply(lambda x: pd.Series(x[k]))
                    ax.plot(p.iloc[:, 0], p.iloc[:, 1], p.iloc[:, 2], linewidth=0.75, color=color)

        ax.set_axis_off()
        plt.show()
        return

    def multiple_figures(self, title: str = "", save: bool = False):
        # plot sphere
        fig, ax = plt.subplots(2, 1, subplot_kw=dict(projection="3d"))
        a, b, c = self.sphere_coordinates()
        ax[0].plot_surface(a, b, c, color="palegoldenrod", alpha=0.3)

        self.approximations['data'] = self.data
        colors = self.colors + ['black'] * (len(self.approximations) + 1 - len(self.colors))

        vertices = self.default_triangle()
        triangle = self.triangle(vertices)

        # plot approximations
        for approximation, color in zip(self.approximations, colors):
            points = approximation.get_points()
            vectors = approximation.get_vectors()
            vectors = pd.Series(np.arange(0, len(points))).apply(lambda i: np.matmul(points.iloc[i], vectors.iloc[i]))

            for i in range(len(points)):
                # plot_triangles
                t = np.matmul(points.iloc[i], triangle.T).T
                v = np.matmul(vectors.iloc[i], triangle.T).T
                ax[0].plot(t[:, 0], t[:, 1], t[:, 2], linewidth=0.5, color=color, alpha=0.5)
                ax[1].plot(v[:, 0], v[:, 1], v[:, 2], linewidth=0.5, color=color, alpha=0.5)

        ax[1].set_axis_off()
        plt.show()
        return

    def axes_plot(self, title: str = "", save: bool = False):
        return

    @staticmethod
    def sphere_coordinates():
        u = np.linspace(0, 4 * np.pi / 2, 100)
        v = np.linspace(0, 2 * np.pi / 2, 100)
        a = np.outer(np.cos(u), np.sin(v))
        b = np.outer(np.sin(u), np.sin(v))
        c = np.outer(np.ones(100), np.cos(v))
        return a, b, c

    def triangle(self, vertices: Union[np.array, None] = None, n: int = 20):
        if vertices is None:
            vertices = self.default_triangle()
        vertices = pd.DataFrame(vertices).apply(lambda vertex: SpherePoint(vertex, project=True), axis=1)
        triangle = []
        for i in range(3):
            for t in np.linspace(0, 1, n):
                triangle.append(vertices[i].mean(vertices[np.mod(i+1, 3)], t).point)
        return np.array(triangle)

    @staticmethod
    def default_triangle(r: float = 0.1):
        a = np.array([r, 0, (1 - r ** 2) ** 0.5])
        b = np.array([r * np.cos(2 * np.pi / 3), r * np.sin(2 * np.pi / 3), (1 - r ** 2) ** 0.5])
        c = np.array([r * np.cos(4 * np.pi / 3), r * np.sin(4 * np.pi / 3), (1 - r ** 2) ** 0.5])
        vertices = np.array([a, b, c])
        return vertices


class PosDefPlot(ManifoldPlot):
    def __init__(self, approximations: list, curve: pd.Series = [], data: pd.Series = [],  colors: list = []):
        super().__init__(approximations=approximations, curve=curve, data=data, colors=colors)
        self.manifold = "positive definite matrices"

    def plot(self, title: str = "", save: bool = False):

        self.approximations['true'] = self.curve
        self.approximations['data'] = self.data
        colors = self.colors + ['gray'] * (len(self.approximations) - len(self.colors))

        # plot approximations
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.set_axis_off()
        for approximation, color in zip(self.approximations, colors):
            points = approximation.get_points()
            n = len(points) - 1
            for i, point in enumerate(points):
                self.ellipsoid(point, ax, 5*np.array([i/n, i/n, 0]), color)
        plt.show()
        return

    def ellipsoid(self, point, ax, center, color):
        u, s, v = np.linalg.svd(point)

        u_ = np.linspace(0.0, 2.0 * np.pi, 60)
        v_ = np.linspace(0.0, np.pi, 60)
        x = s[0] * np.outer(np.cos(u_), np.sin(v_))
        y = s[1] * np.outer(np.sin(u_), np.sin(v_))
        z = s[2] * np.outer(np.ones_like(u_), np.cos(v_))

        for i in range(len(x)):
            for j in range(len(x)):
                [x[i, j], y[i, j], z[i, j]] = np.dot([x[i, j], y[i, j], z[i, j]], v) + center

        ax.plot_surface(x, y, z, rstride=3, cstride=3, linewidth=0.1, alpha=1, shade=True, color=color)
        return
