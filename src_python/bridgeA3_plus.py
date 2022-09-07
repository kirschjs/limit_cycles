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
from four_particle_functions import from3to4

# numerical stability
nBV = 6
nREL = 8
mindisti = [0.0001, 0.0001]
width_bnds = [0.0001, 4.15, 0.0001, 2.25]
minCond = 10**-14

# genetic parameters
anzNewBV = 5
muta_initial = 0.15
anzGen = 2
seed_civ_size = 2
target_pop_size = 5

J0 = 1 / 2

# convention: bound-state-expanding BVs: (1-8), i.e., 8 states per rw set => nzf0*8
channels = [
    ['000', ['he_no1', 'he_no6']],
    #['000', ['t_no1', 't_no6']],
    #['000', ['he_no0']],
]

sysdir3 = sysdir3t if channels[0][1][0].split('_')[0] == 't' else sysdir3he
print('>>> working directory: ', sysdir3)

os.chdir(sysdir3)
subprocess.call('rm -rf *.dat', shell=True)

costr = ''
zop = 31 if tnni == 11 else 14
for nn in range(1, zop):
    cf = 1.0 if ((nn == 1) | (nn == 2) | (nn == 14)) else 0.0
    costr += '%12.7f' % cf if (nn % 7 != 0) else '%12.7f\n' % cf

prepare_einzel3(sysdir3, BINBDGpath)

# 1) prepare an initial set of bases ----------------------------------------------------------------------------------
civs = []
while len(civs) < seed_civ_size:

    new_civs, basi = span_population3(anz_civ=int(3 * seed_civ_size),
                                      fragments=channels,
                                      Jstreu=float(J0),
                                      coefstr=costr,
                                      funcPath=sysdir3,
                                      binPath=BINBDGpath,
                                      mindists=mindisti,
                                      ini_grid_bounds=width_bnds,
                                      ini_dims=[nBV, nREL],
                                      minC=minCond)

    for cciv in new_civs:
        civs.append(cciv)
    print('>>> seed civilizations: %d/%d' % (len(civs), seed_civ_size))

civs.sort(key=lambda tup: np.abs(tup[3]))
civs = sortprint(civs, pr=True)

#ma = blunt_ev3(civs[-1][0],
#               civs[-1][1][0],
#               civs[-1][1][1],
#               basi,
#               funcPath=sysdir3,
#               nzopt=zop,
#               costring=costr,
#               bin_path=BINBDGpath,
#               mpipath=MPIRUN,
#               potNN='%s' % nnpot,
#               potNNN='%s' % nnnpot,
#               parall=-0,
#               anzcores=max(2, min(len(civs[-1][0]), MaxProc)),
#               tnnii=tnni,
#               jay=J0)
#
#try:
#    dim = int(np.sqrt(len(ma) * 0.5))
#    # read Norm and Hamilton matrices
#    normat = np.reshape(np.array(ma[:dim**2]).astype(float), (dim, dim))
#    hammat = np.reshape(np.array(ma[dim**2:]).astype(float), (dim, dim))
#    # diagonalize normalized norm (using "eigh(ermitian)" to speed-up the computation)
#    ewN, evN = eigh(normat)
#    ewH, evH = eigh(hammat, normat)
#    qualTWIN, gsTWIN, basCondTWIN = basQ(ewN, ewH, minCond)
#    print(qualTWIN, gsTWIN, basCondTWIN)
#
#except:
#    print('8472')
#exit()

#civs = []
#while len(civs) < civ_size:
#    basCond = -1
#    gsREF = 42.0
#
#    while ((basCond < minCond) | (gsREF == 0)):
#
#        seedMat = span_initial_basis3(fragments=channels,
#                                      coefstr=costr,
#                                      Jstreu=float(J0),
#                                      funcPath=sysdir3,
#                                      mindists=mindisti,
#                                      ini_grid_bounds=width_bnds,
#                                      ini_dims=[nBV, nREL],
#                                      binPath=BINBDGpath,
#                                      parall=parall)
#
#        seedMat = np.core.records.fromfile('MATOUTB', formats='f8', offset=4)
#
#        dim = int(np.sqrt(len(seedMat) * 0.5))
#
#        # read Norm and Hamilton matricesch
#        normat = np.reshape(
#            np.array(seedMat[:dim**2]).astype(float), (dim, dim))
#        hammat = np.reshape(
#            np.array(seedMat[dim**2:]).astype(float), (dim, dim))
#        # diagonalize normalized norm (using "eigh(ermitian)" to speed-up the computation)
#        # returns e-values in ascending order
#        try:
#            ewN, evN = eigh(normat)
#            ewH, evH = eigh(hammat, normat)
#
#            basCond = np.min(np.abs(ewN)) / np.max(np.abs(ewN))
#            gsREF = ewH[0]
#            gsvREF = evH[:, 0]
#            condREF = basCond
#            subprocess.call('cp -rf INQUA_N_V18 INQUA_N_V18_REF', shell=True)
#            subprocess.call('cp -rf INQUA_N_UIX INQUA_N_UIX_REF', shell=True)
#        except:
#            basCond = 0.0
#            gsREF = 42.
#            continue
#
#        qualREF, gsREF, basCond = basQ(ewN, ewH, minCond)
#
#        gsvREF = evH[:, 0]
#        condREF = basCond
#
#    print('%d -- ' % (civ_size - len(civs)), end='')
#    print('E0(seed) = %4.4f MeV' % gsREF, condREF)
#
#    # 2) rate each basis-vector block according to its contribution to the ground-state energy -------------------
#
#    intw = [
#        np.array(ln.split()).astype(float).tolist() for ln in open('intw.dat')
#    ]
#
#    relw = [
#        np.array(ln.split()).astype(float).tolist() for ln in open('relw.dat')
#    ]
#
#    cfgs = [con.split() for con in open('frags.dat')]
#
#    initialCiv = [cfgs, [intw, relw], qualREF, gsREF, basCond, gsvREF, normat]
#
#    civs.append(initialCiv)
#
#print(civs[-1][:-2])
#exit()

for nGen in range(anzGen):
    tic = time.time()

    qualCUT, gsCUT, basCondCUT = civs[-int(len(civs) / 2)][2:]
    qualREF, gsREF, basCondREF = civs[0][2:]

    # 3) select a subset of basis vectors which are to be replaced -----------------------------------------------

    civ_size = len(civs)
    weights = polynomial_sum_weight(civ_size, order=4)[1::][::-1]
    #print('selection weights: ', weights)
    # 4) select a subset of basis vectors from which replacements are generated ----------------------------------
    children = 0
    while children < anzNewBV:
        twins = []
        for ntwins in range(int(5 * anzNewBV)):
            parent_pair = np.random.choice(range(civ_size),
                                           size=2,
                                           replace=False,
                                           p=weights)

            mother = civs[parent_pair[0]]
            father = civs[parent_pair[1]]

            sbas = []
            bv = 1
            for n in range(len(mother[0])):
                off = np.mod(n, 2)
                for m in range(len(mother[1][0][n])):
                    sbas += [[
                        bv,
                        [
                            x for x in range(1 + off, 1 +
                                             len(mother[1][1][n]), 2)
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

            daughter = [mother[0], wdau, 0, 0, 0]
            son = [mother[0], wson, 0, 0, 0]
            twins.append(daughter)
            twins.append(son)

        # ---------------------------------------------------------------------
        ParaSets = [[
            twins[twinID][1][0], twins[twinID][1][1], sbas, nnpot, nnnpot,
            float(J0), twinID, BINBDGpath, costr
        ] for twinID in range(len(twins))]

        samp_list = []
        cand_list = []
        pool = ThreadPool(max(min(MaxProc, len(ParaSets)), 2))
        jobs = []
        for procnbr in range(len(ParaSets)):
            recv_end, send_end = multiprocessing.Pipe(False)
            pars = ParaSets[procnbr]
            p = multiprocessing.Process(target=end3, args=(pars, send_end))
            jobs.append(p)

            # sen_end returns [ intw, relw, qualREF, gsREF, basCond ]
            samp_list.append(recv_end)
            p.start()
            for proc in jobs:
                proc.join()

        samp_ladder = [x.recv() for x in samp_list]

        samp_ladder.sort(key=lambda tup: np.abs(tup[1]))

        #for el in samp_ladder:
        #    print(el[1:])

        fitchildren = 0
        for cand in samp_ladder[::-1]:
            if ((cand[1] > qualCUT) & (cand[3] > minCond)):
                cfgg = twins[0][0]

                civs.append([cfgg] + cand)
                fitchildren += 1
                if fitchildren + children > anzNewBV:
                    break
        children += fitchildren
        if fitchildren == 0:
            print('%d ' % children, end='')
        else:
            print('adding %d new children.' % children)

        #cand_list.sort(key=lambda tup: np.abs(tup[2]))
        # ---------------------------------------------------------------------

#        for twin in twins:
#
#            ma = blunt_ev3_parallel(twin[0],
#                                    twin[1][0],
#                                    twin[1][1],
#                                    sbas,
#                                    nzopt=zop,
#                                    costring=costr,
#                                    bin_path=BINBDGpath,
#                                    potNN='%s' % nnpot,
#                                    potNNN='%s' % nnnpot,
#                                    tnnii=tnni,
#                                    jay=J0)
#
#            ma = blunt_ev3(twin[0],
#                           twin[1][0],
#                           twin[1][1],
#                           sbas,
#                           funcPath=sysdir3,
#                           nzopt=zop,
#                           costring=costr,
#                           bin_path=BINBDGpath,
#                           mpipath=MPIRUN,
#                           potNN='%s' % nnpot,
#                           potNNN='%s' % nnnpot,
#                           parall=-0,
#                           anzcores=max(2, min(len(civs[0][0]), MaxProc)),
#                           tnnii=tnni,
#                           jay=J0)
#            try:
#                dim = int(np.sqrt(len(ma) * 0.5))
#                # read Norm and Hamilton matrices
#                normat = np.reshape(
#                    np.array(ma[:dim**2]).astype(float), (dim, dim))
#                hammat = np.reshape(
#                    np.array(ma[dim**2:]).astype(float), (dim, dim))
#                # diagonalize normalized norm (using "eigh(ermitian)" to speed-up the computation)
#                ewN, evN = eigh(normat)
#                ewH, evH = eigh(hammat, normat)
#                qualTWIN, gsTWIN, basCondTWIN = basQ(ewN, ewH, minCond)
#                twin[2:] = qualTWIN, gsTWIN, basCondTWIN
#                print(qualTWIN, gsTWIN, basCondTWIN)
#
#            except:
#                # ('unstable child!')
#                qualTWIN, gsTWIN, basCondTWIN = -42, 42, 0
#
#            if ((qualTWIN > qualCUT) & (basCondTWIN > minCond)):
#                civs.append(twin)
#                children += 1
#                if children == anzNewBV:
#                    break
#        exit()

    civs = sortprint(civs, pr=False)

    if len(civs) > target_pop_size:
        currentdim = len(civs)
        weights = polynomial_sum_weight(currentdim, order=4)[1::]
        #print('removal weights: ', weights)
        individual2remove = np.random.choice(range(currentdim),
                                             size=currentdim - target_pop_size,
                                             replace=False,
                                             p=weights)

        civs = [
            civs[n] for n in range(len(civs))
            if (n in individual2remove) == False
        ]
    toc = time.time() - tic
    print('>>> generation %d/%d (dt=%f)' % (nGen, anzGen, toc))
    civs = sortprint(civs, pr=False)

    nGen += 1

    outfile = 'civ_%d.dat' % nGen
    if civs[0][2] > qualREF:
        print('%d) New optimum.' % nGen)
        # wave-function printout (ECCE: in order to work, in addition to the civs[0] argument,
        # I need to hand over the superposition coeffs of the wfkt)
        #write_indiv3(civs[0], outfile)
        print('   opt E = %4.4f   opt cond. = %4.4e' %
              (civs[0][3], civs[0][4]),
              end='\n')

print('\n\n')

civs = sortprint(civs, pr=True)
#plotwidths3(sysdir3)

ma = blunt_ev3(civs[0][0],
               civs[0][1][0],
               civs[0][1][1],
               sbas,
               funcPath=sysdir3,
               nzopt=zop,
               costring=costr,
               bin_path=BINBDGpath,
               mpipath=MPIRUN,
               potNN='%s' % nnpot,
               potNNN='%s' % nnnpot,
               parall=-0,
               anzcores=max(2, min(len(civs[0]), MaxProc)),
               tnnii=tnni,
               jay=J0)

os.system('cp INQUA_N INQUA_N_%s' % lam)
os.system('cp OUTPUT bndg_out_%s' % lam)

dim = int(np.sqrt(len(ma) * 0.5))
# read Norm and Hamilton matrices
normat = np.reshape(np.array(ma[:dim**2]).astype(float), (dim, dim))
hammat = np.reshape(np.array(ma[dim**2:]).astype(float), (dim, dim))
# diagonalize normalized norm (using "eigh(ermitian)" to speed-up the computation)
ewN, evN = eigh(normat)
ewH, evH = eigh(hammat, normat)

# reformat the basis as input for the 4-body calculation
finCiv = [civs[0][0], civs[0][1][0], civs[0][1][1], sbas]
ob_strus, lu_strus, strus = condense_basis_3to4(finCiv,
                                                widthSet_relative,
                                                fn='inq_3to4_%s' % lam)
assert len(lu_strus) == len(ob_strus)

outl = ''
outs = ''
outst = ''

for st in range(len(lu_strus)):
    outl += lu_strus[st] + '\n'
    outs += ob_strus[st] + '\n'
    outst += str(strus[st]) + '\n'

with open('lustru_%s' % lam, 'w') as outfile:
    outfile.write(outl)
with open('obstru_%s' % lam, 'w') as outfile:
    outfile.write(outs)
with open('drei_stru_%s' % lam, 'w') as outfile:
    outfile.write(outst)