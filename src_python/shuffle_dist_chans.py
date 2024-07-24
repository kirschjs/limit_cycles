import os, re, itertools, math
import numpy as np
import random

from parameters_and_constants import *

from sympy.physics.quantum.cg import CG
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import mpmath

os.chdir('/home/kirscher/kette_repo/limit_cycles/systems/4_B2-05_B3-13/4.00')

# 1) read the sequentially-ordered file

inen = [line for line in open('INEN')]
for ll in range(len(inen)):
    if ((inen[ll][-3:-1] == '-1') & (len(inen[ll]) == 13)):
        anzDist = int((len(inen) - ll) / 4)
        lineoffirstDist = ll
        break

# remove dist-chan marker
inen[lineoffirstDist] = inen[lineoffirstDist].strip()[:-3] + '\n'

# 2) redistribute the ordered sequence of channels

nbrCh = int((len(inen) - lineoffirstDist) / 4)
rndorder = np.arange(nbrCh)
np.random.shuffle(rndorder)

outstr = ''.join(inen[:lineoffirstDist])

nc = 0
for nch in rndorder:
    if nc == 0:
        inen[lineoffirstDist +
             4 * rndorder[nch]] = inen[lineoffirstDist +
                                       4 * rndorder[nch]].strip() + ' -1\n'

    outstr += ''.join(
        inen[lineoffirstDist + 4 * rndorder[nch]:lineoffirstDist +
             4 * rndorder[nch] + 4])

    nc += 1

print(outstr)

exit()
