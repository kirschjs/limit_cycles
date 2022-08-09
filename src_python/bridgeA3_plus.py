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
suffix = '3'
sysdir = pathbase + '/systems/' + suffix

BINBDGpath = pathbase + '/src_nucl/'

parall = -1
anzproc = 6  #int(len(os.sched_getaffinity(0)) / 1)

# numerical stability
nBV = 6
nREL = 4
mindisti = [0.02, 0.02]
width_bnds = [0.001, 3.15, 0.001, 1.25]
minCond = 10**-12

# NN: tnni=10   NN+NNN: tnni=11
tnni = 11

# genetic parameters
anzNewBV = 4
muta_initial = 0.05
anzGen = 50
civ_size = 30
target_pop_size = 20

os.chdir(sysdir)
subprocess.call('rm -rf *.dat', shell=True)

nnpot = 'nn_pot'
nnnpot = 'nnn_pot'

J0 = 1 / 2

lam = 4.00
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
prep_pot_file_3N(lam=la, d10=d0, ps3=nnnpot)

# convention: bound-state-expanding BVs: (1-8), i.e., 8 states per rw set => nzf0*8
channels = [
    #['000', ['he_no1', 'he_no6']],
    ['000', ['t_no1', 't_no6', 't_no1', 't_no6']],
    #['000', ['he_no0']],
]

costr = ''
zop = 31 if tnni == 11 else 14
for nn in range(1, zop):
    cf = 1.0 if ((nn == 1) | (nn == 2) | (nn == 14)) else 0.0
    costr += '%12.7f' % cf if (nn % 7 != 0) else '%12.7f\n' % cf
prepare_einzel3(sysdir, BINBDGpath)

# 1) prepare an initial set of bases ----------------------------------------------------------------------------------
civs = []
while len(civs) < civ_size:
    basCond = -1
    gsREF = 42.0

    while ((basCond < minCond) | (gsREF == 0)):

        seedMat = span_initial_basis3(fragments=channels,
                                      coefstr=costr,
                                      Jstreu=float(J0),
                                      funcPath=sysdir,
                                      mindists=mindisti,
                                      ini_grid_bounds=width_bnds,
                                      ini_dims=[nBV, nREL],
                                      binPath=BINBDGpath,
                                      parall=parall)

        seedMat = np.core.records.fromfile('MATOUTB', formats='f8', offset=4)

        dim = int(np.sqrt(len(seedMat) * 0.5))

        # read Norm and Hamilton matricesch
        normat = np.reshape(
            np.array(seedMat[:dim**2]).astype(float), (dim, dim))
        hammat = np.reshape(
            np.array(seedMat[dim**2:]).astype(float), (dim, dim))
        # diagonalize normalized norm (using "eigh(ermitian)" to speed-up the computation)
        # returns e-values in ascending order
        try:
            ewN, evN = eigh(normat)
            ewH, evH = eigh(hammat, normat)

            basCond = np.min(np.abs(ewN)) / np.max(np.abs(ewN))
            gsREF = ewH[0]
            gsvREF = evH[:, 0]
            condREF = basCond
            subprocess.call('cp -rf INQUA_M_V18 INQUA_M_V18_REF', shell=True)
            subprocess.call('cp -rf INQUA_M_UIX INQUA_M_UIX_REF', shell=True)
            print('E0(seed) = %4.4f MeV' % gsREF, condREF)
        except:
            basCond = 0.0
            gsREF = 42.
            continue

        qualREF, gsREF, basCond = basQ(ewN, ewH, minCond)

        gsvREF = evH[:, 0]
        condREF = basCond

    print('%d ' % (civ_size - len(civs)), end='')

    # 2) rate each basis-vector block according to its contribution to the ground-state energy -------------------

    intw = [
        np.array(ln.split()).astype(float).tolist() for ln in open('intw.dat')
    ]

    relw = [
        np.array(ln.split()).astype(float).tolist() for ln in open('relw.dat')
    ]

    cfgs = [con.split() for con in open('frags.dat')]

    initialCiv = [cfgs, [intw, relw], qualREF, gsREF, basCond, gsvREF, normat]

    civs.append(initialCiv)

civs = sortprint(civs, pr=True)

for nGen in range(anzGen):

    qualCUT, gsCUT, basCondCUT, coeffCUT, normCUT = civs[-int(len(civs) /
                                                              2)][2:]
    qualREF, gsREF, basCondREF, coeffREF, normREF = civs[0][2:]

    # 3) select a subset of basis vectors which are to be replaced -----------------------------------------------

    civ_size = len(civs)
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

        sbas = []
        bv = 1
        for n in range(len(mother[0])):
            for m in range(len(mother[1][0][n])):
                sbas += [[
                    bv,
                    [
                        x
                        for x in range(1 + bv % 2, 1 + len(mother[1][1][n]), 2)
                    ]
                ]]
                bv += 1

        # 1) N-1 widths sets
        wson = []
        wdau = []
        for wset in range(len(mother[1])):
            # 2) basis-dependent nbr. of cfgs
            wdau.append([])
            wson.append([])
            for cfg in range(len(mother[0])):

                daughterson = [
                    intertwining(mother[1][wset][cfg][n],
                                 father[1][wset][cfg][n],
                                 mutation_rate=muta_initial)
                    for n in range(len(mother[1][wset][cfg]))
                ]

                rw1 = np.array(daughterson)[:, 0]  #.sort()
                rw1.sort()
                rw2 = np.array(daughterson)[:, 1]  #.sort()
                rw2.sort()
                wdau[-1].append(list(rw1)[::-1])
                wson[-1].append(list(rw2)[::-1])

        daughter = [mother[0], wdau, 0, 0, 0, [], np.array([])]
        son = [mother[0], wson, 0, 0, 0, [], np.array([])]
        twins = [daughter, son]

        for twin in twins:

            ma = blunt_ev3(twin[0],
                           twin[1][0],
                           twin[1][1],
                           sbas,
                           funcPath=sysdir,
                           nzopt=zop,
                           costring=costr,
                           bin_path=BINBDGpath,
                           mpipath=MPIRUN,
                           potNN='./%s' % nnpot,
                           potNNN='./%s' % nnnpot,
                           parall=-0,
                           anzcores=max(2, min(len(initialCiv[0]), MaxProc)),
                           tnnii=tnni,
                           jay=J0)
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
                qualTWIN, gsTWIN, basCondTWIN = basQ(ewN, ewH, minCond)
                twin[2:] = qualTWIN, gsTWIN, basCondTWIN, evH[:, 0], normat

            except:
                # ('unstable child!')
                qualTWIN, gsTWIN, basCondTWIN = -42, 42, 0

            if ((qualTWIN > qualCUT) & (basCondTWIN > minCond)):
                civs.append(twin)
                children += 1
                if children == anzNewBV:
                    break

    civs = sortprint(civs, pr=False)

    if len(civs) > target_pop_size:
        currentdim = len(civs)
        weights = polynomial_sum_weight(currentdim, order=1)[1::]
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
        write_indiv3(civs[0], outfile)
        print('   opt E = %4.4f   opt cond. = %4.4e' %
              (civs[0][3], civs[0][4]),
              end='\n')

print('\n\n')

civs = sortprint(civs, pr=True)
#plotwidths3(sysdir)

ma = blunt_ev3(civs[0][0],
               civs[0][1][0],
               civs[0][1][1],
               sbas,
               funcPath=sysdir,
               nzopt=zop,
               costring=costr,
               bin_path=BINBDGpath,
               mpipath=MPIRUN,
               potNN='./%s' % nnpot,
               potNNN='./%s' % nnnpot,
               parall=-0,
               anzcores=max(2, min(len(initialCiv[0]), MaxProc)),
               tnnii=tnni,
               jay=J0)

dim = int(np.sqrt(len(ma) * 0.5))
# read Norm and Hamilton matrices
normat = np.reshape(np.array(ma[:dim**2]).astype(float), (dim, dim))
hammat = np.reshape(np.array(ma[dim**2:]).astype(float), (dim, dim))
# diagonalize normalized norm (using "eigh(ermitian)" to speed-up the computation)
ewN, evN = eigh(normat)
ewH, evH = eigh(hammat, normat)