from numpy import array
from typing import Union


def av_scheme(f: Union[list, array], num_iterations: int = 1, periodic: bool = False):
    prev = f
    new = []
    for j in range(num_iterations):
        n = len(prev)
        for i in range(n-1):
            new.append(prev[i])
            new.append(prev[i].bezier_av(prev[i+1]))
        new.append(prev[n-1])
        if periodic:
            new.append(prev[n-1].bezier_av(prev[0]))
        prev = new
        new = []
    return prev


def smoothing(f: Union[list, array], num_iterations: int, periodic: bool = True):
    if num_iterations == 0:
        return f
    n = len(f)
    new = []
    for i in range(n-1):
        new.append(f[i].bezier_av(f[i+1]))
    if periodic:
        new.append(f[n-1].bezier_av(f[0]))
    return smoothing(new, num_iterations-1, periodic)


# Lane-Risenfeld with m-1 smoothing steps
def modified_lr(f: Union[list, array], iterations: int, m: int = 1, periodic: Union[bool, None] = None):
    if periodic is None:
        if m > 1:
            periodic = True
        else:
            periodic = False
    for j in range(iterations):
        f = av_scheme(f, 1, periodic)  # refinement step
        f = smoothing(f, m-1, periodic)  # smoothing
    return f
