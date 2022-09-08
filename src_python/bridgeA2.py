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
from parameters_and_constants import *

import multiprocessing
from multiprocessing.pool import ThreadPool

subprocess.call('rm -rf %s/civ_*' % sysdir2, shell=True)

# numerical stability
minCond = 10**-11
minidi = 0.1
denseEVinterval = [-2, 2]

# genetic parameters
anzNewBV = 5
muta_initial = 0.08
anzGen = 142
civ_size = 20
target_pop_size = civ_size

os.chdir(sysdir2)

prep_pot_file_2N(lam=lam, wiC=cloW, baC=0.0, ps2=nnpot)
prep_pot_file_3N(lam=la, d10=d0, ps3=nnnpot)

# convention: bound-state-expanding BVs: (1-8), i.e., 8 states per rw set => nzf0*8
channel = 'np1s'  # no DSI
#channel = 'np1s'  # DSI

J0 = 0
deutDim = 8

zop = 14

costr = ''
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
                                      funcPath=sysdir2,
                                      ini_grid_bounds=[0.001, 6.1],
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

        gsvREF = evH[:, 0]
        condREF = basCond
        seedIter += 1
        #print(gsvREF)
        #exit()
        if seedIter > 1000:
            exit()

    print('%d ' % (civ_size - len(civs)), end='')
    print('E0(seed) = %4.4f MeV' % gsREF, condREF)

    # 2) rate each basis-vector block according to its contribution to the ground-state energy -------------------

    relw = sum([
        np.array(ln.split()).astype(float).tolist() for ln in open('relw.dat')
    ], [])

    initialCiv = [channel, relw, qualREF, gsREF, basCond, gsvREF]

    civs.append(initialCiv)

outfile = 'civ_0.dat'
write_indiv(civs[0], outfile)
print('   opt E = %4.4f   opt cond. = %4.4e' % (civs[0][3], civs[0][4]),
      end='\n')

civs = sortprint(civs, pr=True, ordn=2)

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
                           funcPath=sysdir2)

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

    civs = sortprint(civs, pr=False, ordn=2)

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

    civs = sortprint(civs, pr=False, ordn=2)

    nGen += 1

    outfile = 'civ_%d.dat' % nGen
    if civs[0][2] > qualREF:
        print('%d) New optimum.' % nGen)
        write_indiv(civs[0], outfile)
        print('   opt E = %4.4f   opt cond. = %4.4e' %
              (civs[0][3], civs[0][4]),
              end='\n')

print('\n\n')

civs = sortprint(civs, pr=True, ordn=2)
plotwidths(sysdir2)

ma = blunt_ev2(cfgs=[channel],
               widi=[civs[0][1]],
               basis=sbas,
               nzopt=zop,
               costring=costr,
               binpath=BINBDGpath,
               potNN=nnpot,
               jay=J0,
               funcPath=sysdir2)

dim = int(np.sqrt(len(ma) * 0.5))
# read Norm and Hamilton matrices
normat = np.reshape(np.array(ma[:dim**2]).astype(float), (dim, dim))
hammat = np.reshape(np.array(ma[dim**2:]).astype(float), (dim, dim))
# diagonalize normalized norm (using "eigh(ermitian)" to speed-up the computation)
ewN, evN = eigh(normat)
ewH, evH = eigh(hammat, normat)

os.system('cp INQUA_N INQUA_N_%s' % lam)
os.system('cp OUTPUT bndg_out_%s' % lam)
os.system('cp INEN INEN_BDG')
os.system('cp INEN_STR INEN')
subprocess.run([BINBDGpath + 'DR2END_AK.exe'])

print(">>> calculating 2-body phases.")
spole_2(nzen=nzEN,
        e0=E0,
        d0=D0,
        eps=Eps,
        bet=Bet,
        nzrw=100,
        frr=0.06,
        rhg=8.0,
        rhf=1.0,
        pw=0)

subprocess.run([BINBDGpath + 'S-POLE_PdP.exe'])
os.system('cp PHAOUT phaout_%s' % lam)