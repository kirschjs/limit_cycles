import subprocess
import os, fnmatch, copy, struct
import numpy as np
import sympy as sy
# CG(j1, m1, j2, m2, j3, m3)
from sympy.physics.quantum.cg import CG
from scipy.linalg import eigh
from scipy.optimize import fmin

from three_particle_functions import *
from PSI_parallel_M import *
from rrgm_functions import *
from genetic_width_growth import *
from plot_dist import *
from parameters_and_constants import *

import multiprocessing
from multiprocessing.pool import ThreadPool
from four_particle_functions import from3to4

# flag to be set if after the optimization of the model space, a calibration within
# that space to an observable is ``requested''
fitt = False

# numerical stability
mindi = 0.2

width_bnds = [0.1, 41.15, 0.2, 22.25]
minCond = 10**-14

# genetic parameters
anzNewBV = 6
muta_initial = .02
anzGen = 6
seed_civ_size = 10
target_pop_size = 8

# number of width parameters used for the radial part of each
# (spin) angular-momentum-coupling block
nBV = 6
nREL = 4

J0 = 1 / 2

for channel in channels_3:
    sysdir3 = sysdir3base + '/' + channel
    print('>>> working directory: ', sysdir3)

    if os.path.isdir(sysdir3) == False:
        subprocess.check_call(['mkdir', '-p', sysdir3])
    os.chdir(sysdir3)

    subprocess.call('cp %s .' % nnpot, shell=True)
    subprocess.call('cp %s .' % nnnpot, shell=True)

    subprocess.call('rm -rf *.dat', shell=True)

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

                wa = sum(daughter[1][0] + daughter[1][1], [])
                wb = sum(son[1][0] + son[1][1], [])

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
                float(J0), twinID, BINBDGpath, costr, minCond, evWindow,
                nOperators
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
            if fitchildren == 0:
                print('%d ' % children, end='')
            else:
                print('adding %d new children.' % children)

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

    civs = sortprint(civs, pr=False)
    #plotwidths3(sysdir3)

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

    os.system('cp INQUA_N INQUA_N_%s' % lam)
    os.system('cp OUTPUT bndg_out_%s' % lam)

    smartEV, parCond, gsRatio = smart_ev(ma, threshold=10**-9)
    gsEnergy = smartEV[-1]

    print('\n> basType %s : C-nbr = %4.4e E0 = %4.4e\n\n' %
          (channels_3[channel], parCond, gsEnergy))

    output_nbr(outfi='E0', outval=gsEnergy)

    # reformat the basis as input for the 4-body calculation
    finCiv = [civs[0][0], civs[0][1][0], civs[0][1][1], sbas]
    ob_strus, lu_strus, strus, bvwidthString = condense_basis_3to4(
        finCiv, widthSet_relative, fn='inq_3to4_%s' % lam)

    expC = parse_ev_coeffs_normiert(mult=0,
                                    infil='OUTPUT',
                                    outf='COEFF_NORMAL')

    for wn in range(len(bvwidthString.split('\n'))):

        if bvwidthString.split('\n')[wn] != '':

            print('{%12.8f , %12.8f , %12.8f },' %
                  (float(expC[wn]),
                   float(bvwidthString.split('\n')[wn].split()[0]),
                   float(bvwidthString.split('\n')[wn].split()[1])))

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

    print(">>> End of 3-body day in channel %s\n" % channel)

    if fitt:

        def fitti(fac3, fitb, fix=-1):
            repl_line(
                'INEN', 3,
                '%+12.6f%+12.6f%+12.6f%+12.6f%+12.6f%+12.6f%+12.6f\n' %
                (1.0, 1.0, 1.0, 1.0, 1.0, 1.0, fac3))

            subprocess.run([BINBDGpath + spectralEXE_serial])

            matout = np.core.records.fromfile('MATOUTB',
                                              formats='f8',
                                              offset=4)

            smartEV, parCond, gsRatio = smart_ev(matout, threshold=10**-10)

            print(np.real(smartEV[-4:]))
            E_0 = np.real(smartEV[fix])
            print(abs(float(E_0) + fitb))
            return abs(float(E_0) + fitb)

        # energy to fit to
        trib = 8.48
        # initial scaling factor from which the root-finding algorithm commences its search
        fac = 1.015

        ft_lo = fmin(fitti, fac, args=(trib, -1), disp=False)

        res_lo = fitti(ft_lo[0], 0.0, -1)
        print('L = %2.2f:  D = %12.4f => B(3)= %8.4f   ;  D_start = %12.4f' %
              (lam, d0 * ft_lo[0], res_lo, d0))
        exit()

    subprocess.call('rm -rf TQUAOUT.*', shell=True)
    subprocess.call('rm -rf TDQUAOUT.*', shell=True)
    subprocess.call('rm -rf DMOUT.*', shell=True)
    subprocess.call('rm -rf DRDMOUT.*', shell=True)
    subprocess.call('rm -rf matout_*.*', shell=True)
