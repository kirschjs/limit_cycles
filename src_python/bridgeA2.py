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
minCond = 10**-15
denseEVinterval = [-2, 2]

# genetic parameters
anzNewBV = 1
muta_initial = 0.04
anzGen = 26

os.chdir(sysdir)

nnpot = 'nn_pot'

la = 4.00
cloW = -470.0613865
cloB = -35.1029135

prep_pot_file_2N(lam=la, wiC=cloW, baC=cloB, ps2=nnpot)

# convention: bound-state-expanding BVs: (1-8), i.e., 8 states per rw set => nzf0*8
channel = [
    'np3s',
]
J0 = 1

costr = ''
zop = 14
for nn in range(1, zop):
    cf = 1.0 if (1 <= nn <= 28) else 0.0
    costr += '%12.7f' % cf if (nn % 7 != 0) else '%12.7f\n' % cf

# 1) prepare an initial basis --------------------------------------------------------------------------------

basCond = -1
while basCond < minCond:
    seedMat = span_initial_basis2(fragments=channel,
                                  coefstr=costr,
                                  Jstreu=float(J0),
                                  funcPath=sysdir,
                                  ini_grid_bounds=[0.00001, 5.5],
                                  ini_dims=8,
                                  binPath=BINBDGpath)

    seedMat = np.core.records.fromfile('MATOUTB', formats='f8', offset=4)
    dim = int(np.sqrt(len(seedMat) * 0.5))

    # read Norm and Hamilton matrices
    normat = np.reshape(np.array(seedMat[:dim**2]).astype(float), (dim, dim))
    hammat = np.reshape(np.array(seedMat[dim**2:]).astype(float), (dim, dim))
    # diagonalize normalized norm (using "eigh(ermitian)" to speed-up the computation)
    # returns e-values in ascending order
    ewN, evN = eigh(normat)
    basCond = np.min(np.abs(ewN)) / np.max(np.abs(ewN))

    ewH, evH = eigh(hammat, normat)

    gsREF = ewH[0]
    gsvREF = evH[:, 0]
    condREF = basCond
    print("initial basis: E0 = %f  cond = %e" % (gsREF, basCond))
    exit()

print('GENERATION X) Egs = ', ewH[:4])

# 2) rate each basis-vector block according to its contribution to the ground-state energy -------------------

cfgs = [con.split() for con in open('frags.dat')]
origCFGs = copy.deepcopy(cfgs)

intw = [np.array(ln.split()).astype(float).tolist() for ln in open('intw.dat')]
relw = [np.array(ln.split()).astype(float).tolist() for ln in open('relw.dat')]
rws = []
rw0 = 0
for cfg in range(len(intw)):
    rws.append([])
    for bv in range(len(intw[cfg])):
        rws[-1].append(relw[bv + rw0])
    rw0 += len(intw[cfg])

initialCiv = [cfgs, intw, rws, []]

nbv = 0
for cfg in range(len(initialCiv[0])):
    nbvc = 0
    for bv in initialCiv[1][cfg]:
        nbv += 1
        nbvc += 1
        initialCiv[3] += [[
            nbv,
            np.array(range(1, 1 + len(initialCiv[2][cfg][nbvc - 1]))).tolist()
        ]]

initialCivi = condense_basis(initialCiv, MaxBVsPERcfg=10)

for nGen in range(anzGen):

    initialCiv = copy.deepcopy(initialCivi)

    D0 = initialCiv[3]

    anzBV = len(sum(initialCiv[1], []))

    ParaSets = []

    for bvTrail in D0:
        bvID = [int(bvTrail[0]), int(''.join(map(str, bvTrail[1])))]
        cpy = copy.deepcopy(D0)
        cpy.remove(bvTrail)
        ParaSets.append([
            cpy, J0, costr, zop, tnni, bvID, BINBDGpath, minCond,
            denseEVinterval
        ])

    assert len(ParaSets) == anzBV

    bv_list = []

    pool = ThreadPool(max(min(MaxProc, anzBV), 2))
    jobs = []
    for procnbr in range(anzBV):
        recv_end, send_end = multiprocessing.Pipe(False)
        pars = ParaSets[procnbr]
        p = multiprocessing.Process(target=endmat, args=(pars, send_end))
        jobs.append(p)
        bv_list.append(recv_end)
        p.start()
    for proc in jobs:
        proc.join()

    bv_ladder = [x.recv() for x in bv_list]
    # ranking following condition-number (0) or quality (1)
    ordpara = 1
    bv_ladder.sort(key=lambda tup: np.abs(tup[ordpara]))

    #for bv in bv_ladder:
    #    print(bv[:4])

    # 3) select a subset of basis vectors which are to be replaced -----------------------------------------------

    weights = polynomial_sum_weight(anzBV, order=1)[1::]

    replacement_list = np.random.choice(range(anzBV),
                                        size=anzNewBV,
                                        replace=False,
                                        p=weights)

    # 4) select a subset of basis vectors from which replacements are generated ----------------------------------
    for de in replacement_list:

        bvPerCfg = [len(iws) for iws in initialCiv[1]]
        cfgboundaries = np.cumsum(bvPerCfg)
        repl_cfg = 0
        for cf in range(len(cfgboundaries)):
            if cfgboundaries[cf] >= de:
                break
            repl_cfg += 1

        cfgboundaries = np.insert(cfgboundaries, 0, 0)
        de_cfgrel = de - cfgboundaries[repl_cfg] - 1

        bv_ladder_par = [
            bv[3][0] for bv in bv_ladder
            if bv[3][0] in range(1 + cfgboundaries[repl_cfg], 1 +
                                 cfgboundaries[repl_cfg + 1])
        ]

        anzCand = len(bv_ladder_par)
        nor = int(0.5 * anzCand * (anzCand + 1))
        parent_pair = np.random.choice(range(anzCand),
                                       size=2,
                                       replace=False,
                                       p=np.arange(1, 1 + anzCand) / nor)

        parentIWs = initialCiv[1][repl_cfg]

        parentRWs = initialCiv[2][repl_cfg]

        daughtersonI = []
        daughtersonR = []

        motherI = parentIWs[parent_pair[0]]
        fatherI = parentIWs[parent_pair[1]]

        daughtersonI.append(
            intertwining(motherI, fatherI, mutation_rate=muta_initial))

        for nrw in range(len(parentRWs[parent_pair[0]])):
            motherR = parentRWs[parent_pair[0]][nrw]
            if parent_pair[1] >= 0:
                fatherR = parentRWs[parent_pair[1]][nrw]
            else:
                fatherR = motherR * np.random.random()
            daughtersonR.append(
                intertwining(motherR, fatherR, mutation_rate=muta_initial))

        rw1 = list(np.array(daughtersonR)[:, 1])
        rw1.sort()
        rw1 = rw1[::-1]
        rw2 = list(np.array(daughtersonR)[:, 0])
        rw2.sort()
        rw2 = rw2[::-1]

        initialCiv[1][repl_cfg][de_cfgrel] = daughtersonI[0][0]
        initialCiv[2][repl_cfg][de_cfgrel] = rw1

    ma = blunt_ev(initialCiv[0],
                  initialCiv[1],
                  initialCiv[2],
                  initialCiv[3],
                  funcPath=sysdir,
                  nzopt=zop,
                  costring=costr,
                  bin_path=BINBDGpath,
                  mpipath=MPIRUN,
                  potNN='./%s' % nnpot,
                  potNNN='./%s' % nnnpot,
                  parall=-1,
                  anzcores=max(2, min(len(initialCiv[0]), MaxProc)),
                  tnnii=tnni,
                  jay=J0,
                  dia=False)

    try:
        dim = int(np.sqrt(len(ma) * 0.5))
        # read Norm and Hamilton matrices
        normat = np.reshape(np.array(ma[:dim**2]).astype(float), (dim, dim))
        hammat = np.reshape(np.array(ma[dim**2:]).astype(float), (dim, dim))
        # diagonalize normalized norm (using "eigh(ermitian)" to speed-up the computation)
        ewN, evN = eigh(normat)
        ewH, evH = eigh(hammat, normat)
    except:
        print('new BV yields unstable matrix. Retorting to parents.')
        ewH[0] = 10**4

    basCond = np.min(np.abs(ewN)) / np.max(np.abs(ewN))

    if ((gsREF > ewH[0]) & (basCond > condREF)):
        initialCivi = copy.deepcopy(initialCiv)
        gsREF = ewH[0]
        gsvREF = evH[:, 0]
        print(evH[:, 0])
        condREF = basCond
        subprocess.call('cp -rf INQUA_M_V18 INQUA_M_V18_REF', shell=True)
        subprocess.call('cp -rf INQUA_M_UIX INQUA_M_UIX_REF', shell=True)
        print('GENERATION %d) Egs = %f, cond = %e -> update' %
              (nGen, ewH[0], basCond))
    else:
        print('GENERATION %d) Egs = %f, cond = %e -> no update' %
              (nGen, ewH[0], basCond))

print('THE LAST GENERATION) Egs = %f, cond = %e\n GS =' % (gsREF, condREF),
      gsvREF)
