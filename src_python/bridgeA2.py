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
from plot_dist import *
from parameters_and_constants import lec_list_c

import multiprocessing
from multiprocessing.pool import ThreadPool

home = os.getenv("HOME")

pathbase = home + '/kette_repo/limit_cycles'
suffix = '2'
sysdir = pathbase + '/systems/' + suffix
subprocess.call('rm -rf %s/civ_*' % sysdir, shell=True)

BINBDGpath = pathbase + '/src_nucl/'

# numerical stability
minCond = 10**-11
minidi = 0.01
denseEVinterval = [-2, 2]

# genetic parameters
anzNewBV = 6
muta_initial = 0.2
anzGen = 42
civ_size = 30
target_pop_size = 30

os.chdir(sysdir)

nnpot = 'nn_pot'

lam = 10.00
la = ('%-4.2f' % lam)[:4]
if la in lec_list_def.keys():
    pass
else:
    print('LECs unavailable for chosen cutoff! Available cutoffs:\n',
          lec_list_def.keys())
    exit()

# B2 = 1 MeV and B3 = 8.48 MeV
cloW = lec_list_def[la][0]
cloB = 0.0
d0 = lec_list_def[la][1]

prep_pot_file_2N(lam=lam, wiC=cloW, baC=0.0, ps2=nnpot)

# convention: bound-state-expanding BVs: (1-8), i.e., 8 states per rw set => nzf0*8
channel = 'nn1s'  # no DSI
#channel = 'np1s'  # DSI

J0 = 0
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
    seedIter = 0
    while ((basCond < minCond) | (gsREF > 1)):
        seedMat = span_initial_basis2(channel=channel,
                                      coefstr=costr,
                                      Jstreu=float(J0),
                                      funcPath=sysdir,
                                      ini_grid_bounds=[0.001, 12.1],
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
            print('unfit seed.')
            basCond = -0.0
            continue

        qualREF, gsREF, basCond = basQ(ewN, ewH, minCond)

        print(qualREF, gsREF, basCond)
        gsvREF = evH[:, 0]
        condREF = basCond
        seedIter += 1
        #print(gsvREF)
        #exit()
        if seedIter > 1000:
            exit()

    print('%d ' % (civ_size - len(civs)), end='')

    # 2) rate each basis-vector block according to its contribution to the ground-state energy -------------------

    relw = sum([
        np.array(ln.split()).astype(float).tolist() for ln in open('relw.dat')
    ], [])

    initialCiv = [channel, relw, qualREF, gsREF, basCond, gsvREF]

    civs.append(initialCiv)

civs = sortprint(civs, pr=True)

for nGen in range(anzGen):

    qualCUT, gsCUT, basCondCUT, coeffCUT = civs[-int(len(civs) / 4)][2:]
    qualREF, gsREF, basCondREF, coeffREF = civs[0][2:]

    # 3) select a subset of basis vectors which are to be replaced -----------------------------------------------

    civ_size = len(civs)
    weights = polynomial_sum_weight(civ_size, order=2)[1::]
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

        daughter = [channel, rw1[::-1], 0, 0, 0, []]
        son = [channel, rw2[::-1], 0, 0, 0, []]
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
                # ('unstable child!')
                qualTWIN = -42.0

            qualTWIN, gsTWIN, basCondTWIN = basQ(ewN, ewH, minCond)
            twin[2:] = qualTWIN, gsTWIN, basCondTWIN, evH[:, 0]

            if ((qualTWIN > qualCUT) & (basCondTWIN > minCond) &
                (gsTWIN < gsCUT)):
                civs.append(twin)
                children += 1
                if children == anzNewBV:
                    break

    civs = sortprint(civs, pr=False)

    if len(civs) > target_pop_size:
        currentdim = len(civs)
        weights = polynomial_sum_weight(currentdim, order=2)[1::]
        individual2remove = np.random.choice(range(currentdim),
                                             size=currentdim - target_pop_size,
                                             replace=False,
                                             p=weights)

        civs = [
            civs[n] for n in range(len(civs))
            if (n in individual2remove) == False
        ]

    civs = sortprint(civs, pr=False)

    nGen += 1

    outfile = 'civ_%d.dat' % nGen
    if civs[0][2] > qualREF:
        print('%d) New optimum.' % nGen)
        write_indiv(civs[0], outfile)
        print('   opt E = %4.4f   opt cond. = %4.4e' %
              (civs[0][3], civs[0][4]),
              end='\n')

print('\n\n')

civs = sortprint(civs, pr=True)
plotwidths(sysdir)

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

os.system('cp OUTPUT bndg_out')
os.system('cp INEN_STR INEN')
subprocess.run([BINBDGpath + 'DR2END_AK.exe'])

print(">>> calculating 2-body phases.")
spole_2(nzen=200,
        e0=0.01,
        d0=0.01,
        eps=0.03,
        bet=1.4,
        nzrw=100,
        frr=0.06,
        rhg=8.0,
        rhf=1.0,
        pw=0)

subprocess.run([BINBDGpath + 'S-POLE_PdP.exe'])
