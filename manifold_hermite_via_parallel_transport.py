import copy
import numpy as np
import pandas as pd
from scipy.linalg import expm,logm
from numpy import matmul
from numpy.linalg import inv
from mpv_abs import Mpv


# calculate geodesic mean
def sphere_mean(p, q, t):
    if np.linalg.norm(p-q) < 10**-8:
        return p
    c = p[0]*q[0]+p[1]*q[1]+p[2]*q[2]
    alpha = np.arccos(c)
    t = t*alpha
    q = q-c*p
    q = q/np.linalg.norm(q)
    return np.cos(t)*p+np.sin(t)*q


# calculate the tangent vector of the geodesic from p to q at t
def tangent(p, q, t):
    if np.linalg.norm(p-q) < 10**-8:
        return np.zeros(len(p))
    c = p[0]*q[0]+p[1]*q[1]+p[2]*q[2]
    alpha = np.arccos(c)
    t = t*alpha
    q = q-c*p
    q = q/np.linalg.norm(q)
    return -alpha*np.sin(t)*p+alpha*np.cos(t)*q


def mean(p, q, manifold):
    if (manifold == "surface"):
        return (sphere_mean(p, q, 0.5))
    else:
        return exp(p, 0.5*log(q, p, manifold), manifold)


# manifold takes values "surface" or "matrix"
def exp(p, v, manifold):  # end point of the geodesic from p at direction and speed v,||v|| resp.
    if manifold == "surface":
        a = np.linalg.norm(v)
        if a < 10**-8:
            return p
        return np.cos(a)*p+np.sin(a)*v/a
    elif manifold == "matrix":
        return matmul(p, expm(matmul(inv(p), v)))
    else:
        raise "invalid manifold type"
    return


# returns the tangent vector of the geodesic between p and m at m
def log(p, m, manifold):
    if manifold == "surface":
        if np.linalg.norm(p-m) < 10**-8:
            return np.zeros(len(p))  # the limit of alpha*v/||v|| is zero when alpha goes to zero
        c = p[0]*m[0]+p[1]*m[1]+p[2]*m[2]
        alpha = np.arccos(c)
        v = p-c*m 
        v = v/np.linalg.norm(v)
        return alpha*v
    elif manifold == "matrix":
        return matmul(m, logm(matmul(inv(m), p)))
    else:
        raise "invalid manifold type"
    return


# parallel takes values -1,0,1. defalt is 0.
def parallel(p, m, v, manifold="surface", parallel_transport=0):
    if manifold == "surface":
        if np.linalg.norm(p-m) < 10**-8:
            return v
        u = tangent(p, m, 0)
        w = np.cross(u, p)  # the perpendicular vector to p and u
        A = np.array([[u[0], w[0]], [u[1], w[1]], [u[2], w[2]]])
        coef = np.matmul(np.linalg.pinv(A), v)
        return coef[0]*tangent(p, m, 1)+coef[1]*w

    elif manifold == "matrix":
        if parallel_transport == 0:
            u = exp(p,0.5*log(m, p, manifold), manifold)  # the geodesic mid point between p and m ##########need to check if it doesnt coinside with exp(p,0.5v)
            return matmul(u, matmul(p.T, matmul(v, matmul(inv(p), u))))
        elif parallel_transport > 0:
            return matmul(m, matmul(inv(p), v))
        else:
            return matmul(v, matmul(inv(p), m))
    else:
        raise "invalid manifold type"
    return


# A is the mask, f is the data assumed to be point-vector which concate to create one row-vector
# f.shape=(n,2,dim) n is the number of data points (point-vector) and dim is the dimention of the space
# base_point can be "mid" or "left". defalt is "mid"
# parllel_transport can be -1,0,1. defalt is 0
# mask_bounds =[a,b] assumed a is even, b is odd; insure by providing zero matrices if necessary.
# A.shape=(b-a+1,2,2)


def manifold_hermite(mask, mask_bounds, f, num_iterations=1, periodic: bool = True):
    a = int(0.5*mask_bounds[0])
    b = int(0.5*mask_bounds[1])
    for k in range(num_iterations):
        g = [] # temp data
        n = len(f)
        for i in range(n):
            m = copy.copy(f[i].point)
            even_point = copy.copy(f[i].vector)
            even_vector = copy.copy(f[i].vector)
            odd_point = copy.copy(f[i].vector)
            odd_vector = copy.copy(f[i].vector)

            even_point.point = np.zeros(even_point.point.shape)
            even_vector.point = np.zeros(even_point.point.shape)
            odd_point.point = np.zeros(even_point.point.shape)
            odd_vector.point = np.zeros(even_point.point.shape)

            for j in np.arange(a, b+1):
                if periodic:
                    mpv = copy.copy(f[np.mod(i-j, n)])
                else:
                    if i-j < 0:
                        mpv = copy.copy(f[0])
                        mpv.vector.point = np.zeros(mpv.vector.point.shape)
                    elif i-j >= n:
                        mpv = copy.copy(f[-1])
                        mpv.vector.point = np.zeros(mpv.vector.point.shape)
                    else:
                        mpv = copy.copy(f[i-j])
                left_v = m.log(mpv.point, keep_length=True).point
                right_v = mpv.parallel_transport(m).vector.point
                even_point.point = even_point.point + mask[2*j-2*a][0,0]*left_v + mask[2*j-2*a][0,1]*right_v
                even_vector.point = even_vector.point + mask[2*j-2*a][1,0]*left_v + mask[2*j-2*a][1,1]*right_v
                odd_point.point = odd_point.point + mask[2*j+1-2*a][0,0]*left_v + mask[2*j+1-2*a][0,1]*right_v
                odd_vector.point = odd_vector.point + mask[2*j+1-2*a][1,0]*left_v + mask[2*j+1-2*a][1,1]*right_v
            even_point = m.exp(even_point)
            even_vector = m.parallel_transport(even_vector, even_point)

            odd_point = m.exp(odd_point)
            odd_vector = m.parallel_transport(odd_vector, odd_point)

            g.append(Mpv(even_point, even_vector, unit_tangent=False))
            g.append(Mpv(odd_point, odd_vector, unit_tangent=False))
        f = g
    return g
