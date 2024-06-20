import subprocess
import os, fnmatch, copy, struct
import numpy as np
import sympy as sy
# CG(j1, m1, j2, m2, j3, m3)
from sympy.physics.quantum.cg import CG
from scipy.linalg import eigh

from four_particle_functions import *
from PSI_parallel_M import *
from rrgm_functions import *
from genetic_width_growth import *
from parameters_and_constants import *
from reduce_model_space_4 import redmod

import multiprocessing
from multiprocessing.pool import ThreadPool

# numerical stability
mindi = 1000.0
width_bnds = [0.06, 41.15, 0.08, 53.25]
minCond = 10**-24
maxRat = 10**29

grdTy = ['log_with_density_enhancement', 0.005, 0.001]

# genetic parameters
anzNewBV = 6
muta_initial = 0.004
anzGen = 5
seed_civ_size = 16
target_pop_size = 14

# number of width parameters used for the radial part of each
# (spin) angular-momentum-coupling block
nBV = 18
nREL = 18

J0 = 0

dbg = False
channel = 'alpha'

DRange = np.linspace(start=1.0, stop=-1.5, num=25, endpoint=True, dtype=None)

# read file with 3-body ground-state energies for this LEC range
threeBodyTHs = np.array([
    line.split() for line in open(
        '/home/kirscher/kette_repo/limit_cycles/systems/3_B2-05_B3-8.00/4.00/123/spect/Spect3_D-1.00--1.50.dat'
    )
]).astype(float)[:, [0, -1]]

# [LECD,[ev0,ev1,...]]
results = []
dbg = False
thdefault = 39.569694276328
for tnifac in [DRange[8]]:
    LastEx2opt = 1
    redfac = 0.9
    thEV = [
        redfac * three[1] for three in threeBodyTHs
        if np.abs(lec_set[la][tb][1] * tnifac - three[0]) < 1e-4
    ]
    if thEV == []:
        thEV = -float(thdefault) * redfac
        print('no 3-body threshold found for specific LEC, defaulting to: ',
              thEV)
    else:
        thEV = redfac * thEV[0]

    thEV = -float(thdefault) * redfac
    print('no 3-body threshold found for specific LEC, defaulting to: ', thEV)

    sysdir4o = sysdir4 + '/' + channel
    print('>>> working directory: ', sysdir4o)

    if os.path.isdir(sysdir4) == False:
        subprocess.check_call(['mkdir', '-p', sysdir4])
        prepare_einzel4(sysdir4, BINBDGpath, [channels_4[channel]])
    if os.path.isdir(sysdir4o) == False:
        subprocess.check_call(['mkdir', '-p', sysdir4o])

    os.chdir(sysdir4o)

    subprocess.call('cp %s .' % nnpot, shell=True)
    subprocess.call('cp %s .' % nnnpot, shell=True)
    subprocess.call('rm -rf *.dat', shell=True)

    costr = ''
    zop = 31 if tnni == 11 else 14
    for nn in range(1, zop):
        if ((nn == 1) & (withCoul == True)):
            cf = 1.0
        elif (nn == 2):
            cf = twofac
        elif (nn == 14):
            cf = tnifac
        else:
            cf = 0.0

        costr += '%12.7f' % cf if (nn % 7 != 0) else '%12.7f\n' % cf

    # 1) prepare an initial set of bases ----------------------------------------------------------------------------------
    civs = []
    findlowestEx = True
    while len(civs) < seed_civ_size:
        #nbrStatesOpti4 = list(range(-LastEx2opt, 0))
        nbrStatesOpti4 = [list(range(-LastEx2opt, 0))[0]]

        new_civs, basi = span_population4(anz_civ=int(seed_civ_size),
                                          fragments=[channels_4[channel]],
                                          Jstreu=float(J0),
                                          coefstr=costr,
                                          funcPath=sysdir4o,
                                          binPath=BINBDGpath,
                                          mindists=mindi,
                                          ini_grid_bounds=width_bnds,
                                          ini_dims=[nBV, nREL],
                                          gridType=grdTy,
                                          minC=minCond,
                                          maxR=maxRat,
                                          evWin=evWindow,
                                          nzo=zop,
                                          optRange=nbrStatesOpti4)
        for cciv in new_civs:
            civs.append(cciv)
        print('>>> seed civilizations: %d/%d' % (len(civs), seed_civ_size))

        # tup[2] = pulchritude tup[3] = Energy EV tup[4] = condition number

        civs.sort(key=lambda tup: np.linalg.norm(tup[3]))

        #print(civs[0][3][0])
        #exit()

        oo = np.array([civ[3][0] for civ in civs])

        # if the random bases set has at least one element close to the
        # lowest threshold, restart the seeding process and see if also
        # the next smallest value turns out negative; in this way we
        # will not miss to optimize an excited state below a threshold.

        if findlowestEx:
            if np.any(np.less(oo, thEV * np.ones(len(oo)))):
                LastEx2opt += 1
                civs = []
            else:
                LastEx2opt -= 1
                findlowestEx = False
                civs = []

    civs = sortprint(civs, pr=dbg)

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

            while len(twins) < int(5 * anzNewBV):
                #for ntwins in range(int(5 * anzNewBV)):
                parent_pair = np.random.choice(range(civ_size),
                                               size=2,
                                               replace=False,
                                               p=weights)

                mother = civs[parent_pair[0]]
                father = civs[parent_pair[1]]

                sbas = []
                bv = 1
                for n in range(int(len(mother[0]))):
                    off = np.mod(n, 2)
                    for m in range(int(len(mother[1][0][n]) / 2)):
                        sbas += [[
                            bv,
                            [
                                np.mod(x + off, 2)
                                for x in range(len(mother[1][1][n]))
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
                        #print(mother[1][wset][cfg])
                        #print(father[1][wset][cfg])
                        daughterson = [
                            intertwining(mother[1][wset][cfg][n],
                                         father[1][wset][cfg][n],
                                         mutation_rate=muta_initial,
                                         wMin=0.0001,
                                         wMax=140.,
                                         dbg=False,
                                         method='2point')
                            for n in range(len(mother[1][wset][cfg]))
                        ]
                        #print(daughterson)
                        rw1 = np.array(daughterson)[:, 0]  #.sort()
                        rw1.sort()
                        rw2 = np.array(daughterson)[:, 1]  #.sort()
                        rw2.sort()
                        wdau[-1].append(list(rw1)[::-1])
                        wson[-1].append(list(rw2)[::-1])
                        #print(rw1)
                        #print(rw2)
                        #exit()

                daughter = [mother[0], wdau, 0, 0, 0]
                son = [mother[0], wson, 0, 0, 0]

                #print(mother)
                #print(father)

                wa = sum(daughter[1][0] + daughter[1][1], [])
                wb = sum(son[1][0] + son[1][1], [])

                wai = sum(daughter[1][0], [])
                wbi = sum(son[1][0], [])

                #np.max(wa + wb)
                #print(len(wa))
                #exit()

                # check whether all widths are dufficiently distant
                # (ecce) in most cases, this condition always fails
                #        except for relatively small bases => the seemingly
                prox_check1 = check_dist(width_array1=wai, minDist=mindi * 100)
                prox_check2 = check_dist(width_array1=wbi, minDist=mindi * 100)

                if (prox_check1 == prox_check2 == False):

                    twins.append(daughter)
                    twins.append(son)
                #else:
                #    print('children too close...', end='')

            #print('Gen %d) offspring created and is now rated.' % nGen)
            # ---------------------------------------------------------------------
            ParaSets = [[
                twins[twinID][1][0], twins[twinID][1][1], sbas, nnpotstring,
                nnnpotstring,
                float(J0), twinID, BINBDGpath, costr, minCond, evWindow,
                maxRat, nOperators, nbrStatesOpti4
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

            #print('rating offspring...')
            for chunk in Parchunks:

                pool = ThreadPool(max(min(MaxProc, len(ParaSets)), 2))
                jobs = []

                for procnbr in range(len(chunk)):
                    recv_end, send_end = multiprocessing.Pipe(False)
                    pars = chunk[procnbr]
                    p = multiprocessing.Process(target=end4,
                                                args=(pars, send_end))
                    jobs.append(p)

                    # sen_end returns [ intw, relw, qualREF, gsREF, basCond ]
                    samp_list.append(recv_end)
                    p.start()
                for proc in jobs:
                    proc.join()

            #print('Gen %d) offspring rated.' % nGen)

            samp_ladder = [x.recv() for x in samp_list]

            samp_ladder.sort(key=lambda tup: np.linalg.norm(tup[2]))

            #for el in samp_ladder:
            #    print(el[1:])

            for cand in samp_ladder[::-1]:
                if ((cand[1] > qualCUT) & (cand[3] > minCond)):
                    cfgg = twins[0][0]

                    civs.append([cfgg] + cand)
                    children += 1

                    if children > anzNewBV:
                        break
            print('number of prodigies/target ', children, '/', anzNewBV)

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
            # wave-function printout (ECCE: in order to work, in addition to the civs[0] argument,
            # I need to hand over the superposition coeffs of the wfkt)
            #write_indiv3(civs[0], outfile)
            print(
                '(Gen., Opt cond., Opt lowest EVs) = %d , %4.4e' %
                (nGen, civs[0][4]), civs[0][3])

    print('\n\n')

    civs = sortprint(civs, pr=dbg)

    ma = blunt_ev4t(civs[0][0],
                    civs[0][1][0],
                    civs[0][1][1],
                    sbas,
                    funcPath=sysdir4o,
                    nzopt=zop,
                    costring=costr,
                    bin_path=BINBDGpath,
                    mpipath=MPIRUN,
                    potNN='%s' % nnpotstring,
                    potNNN='%s' % nnnpotstring,
                    parall=-0,
                    anzcores=max(2, min(len(civs[0]), MaxProc)),
                    tnnii=tnni,
                    jay=float(J0))

    smartEV, parCond, gsRatio = smart_ev(ma, threshold=10**-9)
    results.append([tnifac * d0, smartEV[-LastEx2opt::]])

    subprocess.call('rm -rf TQUAOUT.*', shell=True)
    subprocess.call('rm -rf TDQUAOUT.*', shell=True)
    subprocess.call('rm -rf DMOUT.*', shell=True)
    subprocess.call('rm -rf DRDMOUT.*', shell=True)
    subprocess.call('rm -rf matout_*.*', shell=True)

#prepare output string
outs = ''
for res in results:
    outs += '%-20.12f' % res[0]
    outs += '  '
    for ev in res[1]:
        outs += '%-20.12f' % ev
        outs += '  '
    outs += '\n'

outf = 'Spect4_of_D.dat'

if os.path.exists(outf):
    with open(outf, 'a') as outfile:
        outfile.write(outs)
    print('results appended to: ', outf)
else:
    outf = 'Spect4_D-%2.2f-%2.2f.dat' % (DRange[0], DRange[-1])
    with open(outf, 'w') as outfile:
        outfile.write(outs)
    print('results written in: ', outf)