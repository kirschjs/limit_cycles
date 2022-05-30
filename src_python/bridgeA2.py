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
minidi = 0.03
denseEVinterval = [-2, 2]

# genetic parameters
anzNewBV = 10
muta_initial = 0.02
anzGen = 2
civ_size = 10

os.chdir(sysdir)

nnpot = 'nn_pot'

la = 4.00
cloW = -505.20166016
cloB = -0.0

prep_pot_file_2N(lam=la, wiC=cloW, baC=cloB, ps2=nnpot)

# convention: bound-state-expanding BVs: (1-8), i.e., 8 states per rw set => nzf0*8
channel = 'np3s'

J0 = 1
deutDim = 8

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
                                      ini_grid_bounds=[0.00001, 12.5],
                                      ini_dims=deutDim,
                                      binPath=BINBDGpath,
                                      mindist=minidi)

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
    qualCUT, gsCUT, basCondCUT = civs[-len(civs)][2:]

    # 3) select a subset of basis vectors which are to be replaced -----------------------------------------------

    weights = polynomial_sum_weight(civ_size, order=1)[1::]
    # 4) select a subset of basis vectors from which replacements are generated ----------------------------------
    children = 0

    while children < anzNewBV:

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
                civs.append(twin)
                children += 1
                if children == anzNewBV:
                    break

    print('\n qualCUT = %4.4e   gsCUT = %4.4e' % (qualCUT, gsCUT), end="")
    #    for civ in civs:
    #        print(civ[2:])
    #    print('\n')
    #    for child in children:
    #        print(child[2:])

    civs.sort(key=lambda tup: tup[3])

    civs = civs[:anzNewBV]

    nGen += 1

print('\n\n')

for civ in civs:
    print(civ[1:])

ma = blunt_ev2(cfgs=[channel],
               widi=[civs[0][1]],
               basis=sbas,
               nzopt=zop,
               costring=costr,
               binpath=BINBDGpath,
               potNN=nnpot,
               jay=J0,
               funcPath=sysdir)

dim = int(np.sqrt(len(ma) * 0.5))
# read Norm and Hamilton matrices
normat = np.reshape(np.array(ma[:dim**2]).astype(float), (dim, dim))
hammat = np.reshape(np.array(ma[dim**2:]).astype(float), (dim, dim))
# diagonalize normalized norm (using "eigh(ermitian)" to speed-up the computation)
ewN, evN = eigh(normat)
ewH, evH = eigh(hammat, normat)

path_out = sysdir + '/opt_deut.dat'
if os.path.exists(path_out): os.remove(path_out)
sout = ''
for n in range(len(civs[0][1])):
    sout += '%12.4e %12.4e\n' % (evH[0][n], civs[0][1][n])

with open(path_out, 'w') as oof:
    oof.write(sout)
oof.close()
