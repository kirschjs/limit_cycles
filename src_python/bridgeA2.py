import subprocess
import os, fnmatch, copy, struct
import numpy as np
import sympy as sy
# CG(j1, m1, j2, m2, j3, m3)
from sympy.physics.quantum.cg import CG
from scipy.linalg import eigh

from three_particle_functions import *
from PSI_parallel_M import *
from rrgm_functions import *
from genetic_width_growth import *

import multiprocessing
from multiprocessing.pool import ThreadPool

home = os.getenv("HOME")

pathbase = home + '/kette_repo/limit_cycles'
suffix = '2'
sysdir = pathbase + '/systems/' + suffix

BINBDGpath = pathbase + '/src_nucl/'

# numerical stability
minCond = 10**-12
denseEVinterval = [-2, 2]

# genetic parameters
anzNewBV = 4
muta_initial = 0.03
anzGen = 31
civ_size = 30

os.chdir(sysdir)

nnpot = 'nn_pot'

la = 4.00
cloW = -470.0613865
cloB = -35.1029135

prep_pot_file_2N(lam=la, wiC=cloW, baC=cloB, ps2=nnpot)

# convention: bound-state-expanding BVs: (1-8), i.e., 8 states per rw set => nzf0*8
channel = 'np3s'

J0 = 1
deutDim = 10

costr = ''
zop = 14
for nn in range(1, zop):
    cf = 1.0 if (1 <= nn <= 28) else 0.0
    costr += '%12.7f' % cf if (nn % 7 != 0) else '%12.7f\n' % cf

# 1) prepare an initial set of bases ----------------------------------------------------------------------------------
civs = []
while len(civs) < civ_size:
    basCond = -1
    gsREF = 42.0
    while ((basCond < minCond) | (gsREF > 0)):
        seedMat = span_initial_basis2(channel=channel,
                                      coefstr=costr,
                                      Jstreu=float(J0),
                                      funcPath=sysdir,
                                      ini_grid_bounds=[0.00001, 6.5],
                                      ini_dims=deutDim,
                                      binPath=BINBDGpath,
                                      mindist=0.025)

        seedMat = np.core.records.fromfile('MATOUTB', formats='f8', offset=4)
        dim = int(np.sqrt(len(seedMat) * 0.5))

        # read Norm and Hamilton matrices
        normat = np.reshape(
            np.array(seedMat[:dim**2]).astype(float), (dim, dim))
        hammat = np.reshape(
            np.array(seedMat[dim**2:]).astype(float), (dim, dim))
        # diagonalize normalized norm (using "eigh(ermitian)" to speed-up the computation)
        # returns e-values in ascending order
        try:
            ewN, evN = eigh(normat)
            ewH, evH = eigh(hammat, normat)
        except:
            basCond = -42.0
            continue

        qualREF, gsREF, basCond = basQ(ewN, ewH, minCond)

        gsvREF = evH[:, 0]
        condREF = basCond

    print('civilization %d) Egs = %4.4e   cond = %4.4e' %
          (1 + len(civs), gsREF, basCond))

    # 2) rate each basis-vector block according to its contribution to the ground-state energy -------------------

    relw = sum([
        np.array(ln.split()).astype(float).tolist() for ln in open('relw.dat')
    ], [])

    initialCiv = [channel, relw, qualREF, gsREF, basCond]

    civs.append(initialCiv)

civs.sort(key=lambda tup: tup[2])

print('\n')

for civ in civs:
    print(civ[2:])

for nGen in range(anzGen):
    qualCUT, gsCUT, basCondCUT = civs[-anzNewBV][2:]

    # 3) select a subset of basis vectors which are to be replaced -----------------------------------------------

    weights = polynomial_sum_weight(civ_size, order=1)[1::]
    # 4) select a subset of basis vectors from which replacements are generated ----------------------------------
    children = []

    while len(children) < anzNewBV:

        parent_pair = np.random.choice(range(civ_size),
                                       size=2,
                                       replace=False,
                                       p=weights)

        mother = civs[parent_pair[0]]
        father = civs[parent_pair[1]]

        daughterson = [
            intertwining(mother[1][n],
                         father[1][n],
                         mutation_rate=muta_initial)
            for n in range(len(mother[1]))
        ]

        rw1 = np.array(daughterson)[:, 0]  #.sort()
        rw1.sort()
        rw2 = np.array(daughterson)[:, 1]  #.sort()
        rw2.sort()

        sbas = []
        bv = two_body_channels[channel]

        sbas += [[bv, [x for x in range(1, 1 + len(rw1))]]]

        daughter = [channel, rw1[::-1], 0, 0, 0]
        son = [channel, rw2[::-1], 0, 0, 0]
        twins = [daughter, son]

        for twin in twins:
            ma = blunt_ev2(cfgs=[channel],
                           widi=[twin[1]],
                           basis=sbas,
                           nzopt=zop,
                           costring=costr,
                           binpath=BINBDGpath,
                           potNN=nnpot,
                           jay=J0,
                           funcPath=sysdir)

            try:
                dim = int(np.sqrt(len(ma) * 0.5))
                # read Norm and Hamilton matrices
                normat = np.reshape(
                    np.array(ma[:dim**2]).astype(float), (dim, dim))
                hammat = np.reshape(
                    np.array(ma[dim**2:]).astype(float), (dim, dim))
                # diagonalize normalized norm (using "eigh(ermitian)" to speed-up the computation)
                ewN, evN = eigh(normat)
                ewH, evH = eigh(hammat, normat)
            except:
                print('unstable child!')
                qualTWIN = -42.0

            qualTWIN, gsTWIN, basCondTWIN = basQ(ewN, ewH, minCond)
            twin[2:] = qualTWIN, gsTWIN, basCondTWIN

            if ((qualTWIN > qualCUT) & (basCondTWIN > minCond)):
                children.append(twin)
                if len(children) == anzNewBV:
                    break

    print('\n qualCUT = %4.4e\n qualGSE = %4.4e' % (qualCUT, gsCUT))
    #    for civ in civs:
    #        print(civ[2:])
    #    print('\n')
    #    for child in children:
    #        print(child[2:])

    civs[-anzNewBV:] = children
    civs.sort(key=lambda tup: tup[3])

    nGen += 1

print('\n')

for civ in civs:
    print(civ[2:])