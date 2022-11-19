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

# numerical stability
minCond = 10**-10
minidi_seed = 0.05
minidi_breed = 0.25
denseEVinterval = [-2, 2]
width_bnds = [0.005, 12.25]
deutDim = 12

# genetic parameters
anzNewBV = 6
muta_initial = 0.013
anzGen = 4
civ_size = 10
target_pop_size = civ_size
zop = 14

for channel in channels_2:
    J0 = channels_2[channel][1]

    sysdir2 = sysdir2base + '/' + channel

    if os.path.isdir(sysdir2) == False:
        subprocess.check_call(['mkdir', '-p', sysdir2])

    subprocess.call('rm -rf %s/civ_*' % sysdir2, shell=True)

    os.chdir(sysdir2base)

    prep_pot_file_2N(lam=lam, wiC=cloW, baC=cloB, ps2=nnpot)
    prep_pot_file_3N(lam=la, d10=d0, ps3=nnnpot)

    os.chdir(sysdir2)
    subprocess.call('cp %s .' % nnpot, shell=True)

    costr = ''
    for nn in range(1, zop):
        cf = tnf if (1 <= nn <= 28) else 0.0
        costr += '%12.7f' % cf if (nn % 7 != 0) else '%12.7f\n' % cf

    # 1) prepare an initial set of bases ----------------------------------------------------------------------------------
    civs = []
    while len(civs) < civ_size:

        new_civs, basi = span_population2(anz_civ=int(3 * civ_size),
                                          fragments=channel,
                                          Jstreu=float(J0),
                                          coefstr=costr,
                                          funcPath=sysdir2,
                                          binPath=BINBDGpath,
                                          mindist=minidi_seed,
                                          ini_grid_bounds=width_bnds,
                                          ini_dims=deutDim,
                                          minC=minCond,
                                          evWin=evWindow)
        for cciv in new_civs:
            civs.append(cciv)
        print('>>> seed civilizations: %d/%d' % (len(civs), civ_size))

    civs.sort(key=lambda tup: np.abs(tup[3]))
    civs = sortprint(civs, pr=True)

    for nGen in range(anzGen):

        try:
            qualCUT, gsCUT, basCondCUT = civs[-int(len(civs) / 2)][2:]
            qualREF, gsREF, basCondREF = civs[0][2:]
        except:
            print('exception:')
            print(civs[-int(len(civs) / 2)][2:])
            print(civs[0][2:])
        # 3) select a subset of basis vectors which are to be replaced -----------------------------------------------

        civ_size = len(civs)
        weights = polynomial_sum_weight(civ_size, order=2)[1::]
        # 4) select a subset of basis vectors from which replacements are generated ----------------------------------
        children = 0

        while children < anzNewBV:
            twins = []
            while len(twins) < int(5 * anzNewBV):
                parent_pair = np.random.choice(range(civ_size),
                                               size=2,
                                               replace=False,
                                               p=weights)

                mother = civs[parent_pair[0]]
                father = civs[parent_pair[1]]

                wson = []
                wdau = []
                for wset in range(len(mother[1])):
                    # 2) basis-dependent nbr. of cfgs
                    daughterson = [
                        intertwining(mother[1][wset][n],
                                     father[1][wset][n],
                                     mutation_rate=muta_initial)
                        for n in range(len(mother[1][wset]))
                    ]
                    rw1 = np.array(daughterson)[:, 0]  #.sort()
                    rw1.sort()
                    rw2 = np.array(daughterson)[:, 1]  #.sort()
                    rw2.sort()

                    if ((check_dist(width_array=rw1, minDist=minidi_breed)
                         == False)
                            & (check_dist(width_array=rw2,
                                          minDist=minidi_breed) == False)):

                        wdau.append(list(rw1)[::-1])
                        wson.append(list(rw2)[::-1])

                        daughter = [mother[0], wdau, 0, 0, 0]
                        son = [mother[0], wson, 0, 0, 0]
                        twins.append(daughter)
                        twins.append(son)

            sbas = []
            bv = two_body_channels[channel]
            bv = 0
            for n in range(len(twins[-1][0])):
                sbas += [[
                    two_body_channels[channel] + bv,
                    [1 for x in range(1, 1 + len(twins[-1][1][n]))]
                ]]
                bv += len(two_body_channels)

            ParaSets = [[
                twins[twinID][1], sbas, nnpotstring,
                float(J0), BINBDGpath, costr, twinID, minCond, evWindow
            ] for twinID in range(len(twins))]

            samp_list = []
            cand_list = []

            pool = ThreadPool(max(min(MaxProc, len(ParaSets)), 2))
            jobs = []
            for procnbr in range(len(ParaSets)):
                recv_end, send_end = multiprocessing.Pipe(False)
                pars = ParaSets[procnbr]
                p = multiprocessing.Process(target=end2, args=(pars, send_end))
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
            #if fitchildren == 0:
            #    print('%d ' % children, end='')
            #else:
            #    print('adding %d new children.' % children)

        civs = sortprint(civs, pr=False, ordn=2)

        if len(civs) > target_pop_size:
            currentdim = len(civs)
            weights = polynomial_sum_weight(currentdim, order=2)[1::]
            individual2remove = np.random.choice(range(currentdim),
                                                 size=currentdim -
                                                 target_pop_size,
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
            print('   opt E = %4.4f   opt cond. = %4.4e' %
                  (civs[0][3], civs[0][4]),
                  end='\n')

    print('\n\n')

    civs = sortprint(civs, pr=True, ordn=2)

    ma = blunt_ev2(cfgs=civs[0][0],
                   widi=civs[0][1],
                   basis=sbas,
                   nzopt=zop,
                   costring=costr,
                   binpath=BINBDGpath,
                   potNN=nnpotstring,
                   jay=J0,
                   funcPath=sysdir2)

    smartEV, parCond = smart_ev(ma, threshold=minCond)
    gsEnergy = smartEV[-1]

    print('\n> basType %s : C-nbr = %4.4e E0 = %4.4e\n\n' %
          (channel, parCond, gsEnergy))

    os.system('cp INQUA_N INQUA_N_%s' % (lam))
    os.system('cp OUTPUT bndg_out_%s' % (lam))
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
    os.system('cp PHAOUT phaout_%s' % (lam))