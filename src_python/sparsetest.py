import math
import itertools
import random
import numpy as np
from timeit import timeit


def join_pairs(points, r):
    for p, q in itertools.combinations(points, 2):
        if np.linalg.norm(np.array(p) - np.array(q)) < r:
            points.remove(q)
            return True
    return False


def sparse_subset(points, r):
    """Return a maximal list of elements of points such that no pairs of
    points in the result have distance less than r.

    """
    points = points[:]

    removing = True
    tmpts = points
    while removing:
        for p, q in itertools.combinations(tmpts, 2):
            if np.linalg.norm(np.array(p) - np.array(q)) < r:
                tmpts.remove(q)
                break
        removing = False if len(tmpts) == len(points) else True

    return tmpts


POINTS = np.random.random(10).tolist()
sp = sparse_subset1(POINTS, 0.4)
print(POINTS)
print(sp)
