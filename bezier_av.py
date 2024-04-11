from mpv_abs import Mpv
from numpy import cos


def bezier_av(x: Mpv, y: Mpv, omega: float):
    theta = x.vectors_distance(y)
    alpha = x.points_distance(y)/(3 * cos(theta/4)**2)
    p0 = x.point
    p1 = x.exp(alpha)
    p2 = y.exp(-alpha)
    p3 = y.point
    points = [p0, p1, p2, p3]
    while len(points) > 1:
        new_points = []
        for i in range(len(points) - 1):
            new_points.append(points[i].mean(other=points[i+1], weight=omega))
        points = new_points
        if len(points) == 2:
            vector = points[1].log(other=points[0])/2
    point = points[0]
    return Mpv(point, vector)
