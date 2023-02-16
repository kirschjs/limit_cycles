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
from plot_dist import *
from parameters_and_constants import *
from reduce_model_space_4 import redmod

import multiprocessing
from multiprocessing.pool import ThreadPool

# numerical stability
mindi = 0.2
width_bnds = [0.001, 618.15, 0.002, 751.25]
minCond = 10**-14
maxRat = 10**19

# genetic parameters
anzNewBV = 4
muta_initial = .002
anzGen = 10
seed_civ_size = 20
target_pop_size = 20

# number of width parameters used for the radial part of each
# (spin) angular-momentum-coupling block
nBV = 12
nREL = 12

einzel4 = True

J0 = 0

for channel in channels_4:
    sysdir4 = sysdir4 + '/' + channel
    print('>>> working directory: ', sysdir4)

    if os.path.isdir(sysdir4) == False:
        subprocess.check_call(['mkdir', '-p', sysdir4])
    if einzel4 == True:
        prepare_einzel4(sysdir4, BINBDGpath, [channels_4[channel]])

    os.chdir(sysdir4)

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
    while len(civs) < seed_civ_size:
        new_civs, basi = span_population4(anz_civ=int(seed_civ_size),
                                          fragments=[channels_4[channel]],
                                          Jstreu=float(J0),
                                          coefstr=costr,
                                          funcPath=sysdir4,
                                          binPath=BINBDGpath,
                                          mindists=mindi,
                                          ini_grid_bounds=width_bnds,
                                          ini_dims=[nBV, nREL],
                                          minC=minCond,
                                          maxR=maxRat,
                                          evWin=evWindow)
        for cciv in new_civs:
            civs.append(cciv)
        print('>>> seed civilizations: %d/%d' % (len(civs), seed_civ_size))

    civs.sort(key=lambda tup: np.abs(tup[3]))
    civs = sortprint(civs, pr=True)

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
            while len(twins) < int(anzNewBV):
                #for ntwins in range(int(5 * anzNewBV)):
                parent_pair = np.random.choice(range(civ_size),
                                               size=2,
                                               replace=False,
                                               p=weights)

                mother = civs[parent_pair[0]]
                father = civs[parent_pair[1]]

                sbas = []
                bv = 1
                for n in range(int(len(mother[0]) / 2)):
                    off = np.mod(n, 2)
                    for m in range(len(mother[1][0][n])):
                        sbas += [[
                            bv, [1 for x in range(len(mother[1][1][n]))]
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
                        #rw1.sort()
                        rw2 = np.array(daughterson)[:, 1]  #.sort()
                        #rw2.sort()
                        wdau[-1].append(list(rw1)[::-1])
                        wson[-1].append(list(rw2)[::-1])

                daughter = [mother[0], wdau, 0, 0, 0]
                son = [mother[0], wson, 0, 0, 0]

                #print(mother)
                #print(father)

                wa = sum(daughter[1][0] + daughter[1][1], [])
                wb = sum(son[1][0] + son[1][1], [])

                #print(wa)
                #print(wb)
                #exit()

                prox_check1 = check_dist(width_array1=wa, minDist=mindi)
                prox_check2 = check_dist(width_array1=wb, minDist=mindi)
                prox_checkr1 = check_dist(width_array1=wa,
                                          width_array2=widthSet_relative,
                                          minDist=mindi)
                prox_checkr2 = check_dist(width_array1=wb,
                                          width_array2=widthSet_relative,
                                          minDist=mindi)

                if prox_check1 * prox_check2 * prox_checkr1 * prox_checkr2 == True:

                    twins.append(daughter)
                    twins.append(son)

            # ---------------------------------------------------------------------
            ParaSets = [[
                twins[twinID][1][0], twins[twinID][1][1], sbas, nnpotstring,
                nnnpotstring,
                float(J0), twinID, BINBDGpath, costr, minCond, evWindow, maxRat
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

            print('rating offspring...')
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

            print('offspring rated.')
            samp_ladder = [x.recv() for x in samp_list]

            samp_ladder.sort(key=lambda tup: np.abs(tup[1]))

            #for el in samp_ladder:
            #    print(el[1:])

            for cand in samp_ladder[::-1]:
                if ((cand[1] > qualCUT) & (cand[3] > minCond)):
                    cfgg = twins[0][0]

                    civs.append([cfgg] + cand)
                    children += 1
                    print('another prodigy added.')

                    if children > anzNewBV:
                        break

        civs = sortprint(civs, pr=False)

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

    ma = blunt_ev4t(civs[0][0],
                    civs[0][1][0],
                    civs[0][1][1],
                    sbas,
                    funcPath=sysdir4,
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
    gsEnergy = smartEV[-1]

    print('\n> basType %s : C-nbr = %4.4e E0 = %4.4e\n\n' %
          (channels_4[channel], parCond, gsEnergy))

    print(civs[0])

    redmod(BINBDGpath)

    tt = get_bas()
    ttt = get_bsv_rw_idx()
    # reformat the basis as input for the 5-body calculation

    finCiv = [civs[0][0], civs[0][1][0], civs[0][1][1], ttt]
    ob_strus, lu_strus, strus = condense_basis_4to5(finCiv,
                                                    widthSet_relative,
                                                    fn='inq_4to5_%s' % lam)

    print(ob_strus, lu_strus, strus)
    # reformat the basis as input for the 5-body calculation

    assert len(lu_strus) == len(ob_strus)

    os.system('cp INQUA_N INQUA_N_%s' % lam)
    os.system('cp OUTPUT bndg_out_%s' % lam)

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
    with open('vier_stru_%s' % lam, 'w') as outfile:
        outfile.write(outst)

    subprocess.call('rm -rf TQUAOUT.*', shell=True)
    subprocess.call('rm -rf TDQUAOUT.*', shell=True)
    subprocess.call('rm -rf DMOUT.*', shell=True)
    subprocess.call('rm -rf DRDMOUT.*', shell=True)
    subprocess.call('rm -rf matout_*.*', shell=True)