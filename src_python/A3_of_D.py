import subprocess
import os, fnmatch, copy, struct
import numpy as np
from scipy.linalg import eigh
from scipy.optimize import fmin
from numpy import asarray
from numpy import savetxt

from three_particle_functions import *
from PSI_parallel_M import *
from rrgm_functions import *
from genetic_width_growth import *
from parameters_and_constants import *

import multiprocessing
from multiprocessing.pool import ThreadPool
from four_particle_functions import from3to4

# flag to be set if after the optimization of the model space, a calibration within
# that space to an observable is ``requested''
fitt = False

# numerical stability
mindi = 0.3

width_bnds = [0.01, 18.15, 0.001, 24.25]
minCond = 10**-17

# genetic parameters
anzNewBV = 5
muta_initial = .035
anzGen = 8
seed_civ_size = 50
target_pop_size = 25

# number of width parameters used for the radial part of each
# (spin) angular-momentum-coupling block
nBV = 7
nREL = 6

J0 = 1 / 2

channel = '123'
sysdir3 = sysdir3base + '/' + channel + '/spect'
print('>>> working directory: ', sysdir3)
if id_chan == 0:
    refdir = sysdir3
if os.path.isdir(sysdir3) == False:
    subprocess.check_call(['mkdir', '-p', sysdir3])
os.chdir(sysdir3)
subprocess.call('cp %s .' % nnpot, shell=True)
subprocess.call('cp %s .' % nnnpot, shell=True)

DRange = np.linspace(start=10.95, stop=1.05, num=20, endpoint=True, dtype=None)

# [LECD,[ev0,ev1,...]]
results = []
dbg = False
for tnifac in DRange:

    nbrStatesOpti3 = [-2]
    costr = ''
    zop = nOperators if tnni == 11 else 14
    for nn in range(1, zop):
        if (nn == 1):
            cf = int(withCoul)
        elif (nn == 2):
            cf = twofac
        elif (nn == 14):
            cf = tnifac
        else:
            cf = 0.0

        costr += '%12.7f' % cf if (nn % 7 != 0) else '%12.7f\n' % cf

    prepare_einzel3(sysdir3, BINBDGpath)

    # 1) prepare an initial set of bases ----------------------------------------------------------------------------------
    civs = []
    while len(civs) < seed_civ_size:
        new_civs, basi = span_population3(anz_civ=int(3 * seed_civ_size),
                                          fragments=channels_3[channel],
                                          Jstreu=float(J0),
                                          coefstr=costr,
                                          nzo=nOperators,
                                          funcPath=sysdir3,
                                          binPath=BINBDGpath,
                                          mindists=mindi,
                                          ini_grid_bounds=width_bnds,
                                          ini_dims=[nBV, nREL],
                                          minC=minCond,
                                          evWin=evWindow,
                                          anzOptStates=nbrStatesOpti3)

        for cciv in new_civs:
            civs.append(cciv)
        print('>>> seed civilizations: %d/%d' % (len(civs), seed_civ_size))

        if ((id_chan == 1) & (len(civs) > 1)):
            break

    civs.sort(key=lambda tup: np.abs(tup[3]))
    civs = sortprint(civs, pr=False)

    for nGen in range(anzGen):

        if id_chan == 1:
            break

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
            while len(twins) < int(42 * anzNewBV):
                #for ntwins in range(int(5 * anzNewBV)):
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

                assert len(mother[1]) % 2 == 0

                for wset in range(len(mother[1])):
                    # 2) basis-dependent nbr. of cfgs
                    wdau.append([])
                    wson.append([])

                    # 3) evolve only half of the parameters as the other spin cfg must use the same
                    #    in case of SU(4) symmetry, anyway
                    enforceSym = 2 if bin_suffix == 'expl' else 1
                    for cfg in range(int(len(mother[0]) / enforceSym)):

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

                    # 4) enforce the same width parameters for the other half of spin cfgs
                    if enforceSym == 2:
                        for cfg in range(int(len(mother[0]) / 2)):
                            wdau[-1].append(wdau[-1][cfg])
                            wson[-1].append(wson[-1][cfg])

                daughter = [mother[0], wdau, 0, 0, 0]
                son = [mother[0], wson, 0, 0, 0]

                wa = sum(daughter[1][0] + daughter[1][1], [])
                wb = sum(son[1][0] + son[1][1], [])

                prox_check1 = check_dist(width_array1=wa, minDist=mindi)
                prox_check2 = check_dist(width_array1=wb, minDist=mindi)
                prox_checkr1 = np.all([
                    check_dist(width_array1=wa,
                               width_array2=wsr,
                               minDist=mindi) for wsr in widthSet_relative
                ])
                prox_checkr2 = np.all([
                    check_dist(width_array1=wb,
                               width_array2=wsr,
                               minDist=mindi) for wsr in widthSet_relative
                ])

                if ((prox_check1 * prox_check2 * prox_checkr1 * prox_checkr2
                     == True) & ((max(wa) <= max(width_bnds))
                                 & (max(wb) <= max(width_bnds)))):

                    twins.append(daughter)
                    twins.append(son)

            # ---------------------------------------------------------------------
            ParaSets = [[
                twins[twinID][1][0], twins[twinID][1][1], sbas, nnpotstring,
                nnnpotstring,
                float(J0), twinID, BINBDGpath, costr, minCond, evWindow,
                nOperators, nbrStatesOpti3
            ] for twinID in range(len(twins))]

            # x) the parallel environment is set up in sets(chunks) of bases
            #    in order to limit the number of files open simultaneously
            split_points = [
                n * maxParLen
                for n in range(1 + int(len(ParaSets) / maxParLen))
            ] + [len(ParaSets) + 1024]

            Parchunks = [
                ParaSets[split_points[i]:split_points[i + 1]]
                for i in range(len(split_points) - 1)
            ]

            samp_list = []
            cand_list = []

            for chunk in Parchunks:

                pool = ThreadPool(max(min(MaxProc, len(ParaSets)), 2))
                jobs = []

                for procnbr in range(len(chunk)):
                    recv_end, send_end = multiprocessing.Pipe(False)
                    pars = chunk[procnbr]
                    p = multiprocessing.Process(target=end3,
                                                args=(pars, send_end))
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
            if dbg:
                if fitchildren == 0:
                    print('%d ' % children, end='')
                else:
                    print('adding %d new children.' % children)

        civs = sortprint(civs, pr=dbg)

        if len(civs) > target_pop_size:
            currentdim = len(civs)
            weights = polynomial_sum_weight(currentdim, order=4)[1::]
            #print('removal weights: ', weights)
            individual2remove = np.random.choice(range(currentdim),
                                                 size=currentdim -
                                                 target_pop_size,
                                                 replace=False,
                                                 p=weights)

            civs = [
                civs[n] for n in range(len(civs))
                if (n in individual2remove) == False
            ]
        toc = time.time() - tic
        if dbg:
            print('>>> generation %d/%d (dt=%f)' % (nGen, anzGen, toc))
        civs = sortprint(civs, pr=dbg)

        nGen += 1

        outfile = 'civ_%d.dat' % nGen
        if civs[0][2] > qualREF:
            print(
                'Gen %d) New optimum with opt E = %4.4f   opt cond. = %4.4e' %
                (nGen, civs[0][3], civs[0][4]))

    civs = sortprint(civs, pr=dbg)

    ma = blunt_ev3(
        civs[0][0],
        civs[0][1][0],
        civs[0][1][1],
        sbas,
        funcPath=sysdir3,
        nzopt=zop,
        costring=costr,
        bin_path=BINBDGpath,
        mpipath=MPIRUN,
        potNN='%s' % nnpotstring,
        potNNN='%s' % nnnpotstring,
        # in order to pass superposition coefficients through bndg_out on to 4- and 5- body
        # scattering-calculation input, this function needs to run serial
        parall=-0,
        anzcores=max(2, min(len(civs[0]), MaxProc)),
        tnnii=tnni,
        jay=float(J0))

    smartEV, parCond, gsRatio = smart_ev(ma, threshold=10**-9)

    results.append([tnifac * d0, smartEV[-5::]])

#prepare output string
outs = ''
for res in results:
    outs += '%-f20.12' % res[0]
    outs += '  '
    for ev in res[1]:
        outs += '%-f20.12' % ev
        outs += '  '
    outs += '\n'

with open('Spect3_of_D.dat', 'w') as outfile:
    outfile.write(outs)

subprocess.call('rm -rf TQUAOUT.*', shell=True)
subprocess.call('rm -rf TDQUAOUT.*', shell=True)
subprocess.call('rm -rf DMOUT.*', shell=True)
subprocess.call('rm -rf DRDMOUT.*', shell=True)
subprocess.call('rm -rf matout_*.*', shell=True)