import os, re, itertools, math
import numpy as np
import random

from parameters_and_constants import *

from sympy.physics.quantum.cg import CG
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import mpmath

sbas = np.array([
    a[1]
    for a in [[1, [0, 1, 0, 1, 0, 1, 0, 1]], [2, [0, 1, 0, 1, 0, 1, 0, 1]],
              [3, [0, 1, 0, 1, 0, 1, 0, 1]], [4, [0, 1, 0, 1, 0, 1, 0, 1]],
              [5, [0, 1, 0, 1, 0, 1, 0, 1]], [6, [0, 1, 0, 1, 0, 1, 0, 1]],
              [7, [0, 1, 0, 1, 0, 1, 0, 1]], [8, [0, 1, 0, 1, 0, 1, 0, 1]],
              [9, [1, 0, 1, 0, 1, 0, 1, 0]], [10, [1, 0, 1, 0, 1, 0, 1, 0]],
              [11, [1, 0, 1, 0, 1, 0, 1, 0]], [12, [1, 0, 1, 0, 1, 0, 1, 0]],
              [13, [1, 0, 1, 0, 1, 0, 1, 0]], [14, [1, 0, 1, 0, 1, 0, 1, 0]],
              [15, [1, 0, 1, 0, 1, 0, 1, 0]], [16, [1, 0, 1, 0, 1, 0, 1, 0]]]
])

with open('test.out', 'w') as f:
    np.savetxt(f, sbas, delimiter=' ', fmt='%d')