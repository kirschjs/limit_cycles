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

# flag to be set if after the optimization of the model space, a calibration within
# that space to an observable is ``requested''
fitt = False

# numerical stability
minCond = 10**-10
minidi_breed = 0.3
minidi_seed = minidi_breed
minidi_breed_rel = minidi_breed
denseEVinterval = [-2, 2]
width_bnds = [0.1, 7.25]

deutDim = 5

miniE_breed = -0.1

# genetic parameters
anzNewBV = 6
muta_initial = 0.02
anzGen = 2
civ_size = 20
target_pop_size = 15

zop = 14 if bin_suffix == '_v18-uix' else 11

for channel in channels_2:
    J0 = channels_2[channel][1]

    sysdir2 = sysdir2base + '/' + channel

    if os.path.isdir(sysdir2) == False:
        subprocess.check_call(['mkdir', '-p', sysdir2])

    subprocess.call('rm -rf %s/civ_*' % sysdir2, shell=True)

    os.chdir(sysdir2base)
    print('>>> working directory: ', sysdir2)

    if bin_suffix == '_v18-uix':
        prep_pot_file_2N(lam=(2 * np.sqrt(float(lam))),
                         wiC=cloW,
                         baC=cloB,
                         ps2=nnpot)
    elif bin_suffix == '_eft-cib':
        prep_pot_file_2N_pp(lam=2 * np.sqrt(float(lam)),
                            wiC=cloW,
                            baC=cloB,
                            ppC=cpp,
                            ps2=nnpot)
    else:
        print('no potential structure assigned to suffix.')
        exit()
    prep_pot_file_3N(lam=2 * np.sqrt(float(lam)), d10=d0, ps3=nnnpot)
    #continue

    os.chdir(sysdir2)
    if id_chan == 0:
        refdir = sysdir2
    subprocess.call('cp %s .' % nnpot, shell=True)

    prescat = False
    if prescat:
        os.system('cp PHAOUT phaout_%s' % (lam))
        print(">>> 2-body phases calculated. End of day for channel %s\n" %
              channel)

        phaa = read_phase()

        redmass = mn['137'] / 2
        if channel[:2] == 'ppXXXXX':
            print('proton-proton channel:\n')
            a_aa = [
                appC(phaa[n][2] * np.pi / 180.,
                     np.sqrt(2 * redmass * phaa[n][0]),
                     redmass,
                     q1=1,
                     q2=1) for n in range(len(phaa))
            ]
        else:
            a_aa = [
                -MeVfm * np.tan(phaa[n][2] * np.pi / 180.) /
                np.sqrt(2 * redmass * phaa[n][0]) for n in range(len(phaa))
            ]
        print(
            'a_aa(E_min) = (%4.4f+i%4.4f) fm   a_aa(E_max) = (%4.4f+i%4.4f) fm'
            % (a_aa[0].real, a_aa[0].imag, a_aa[-1].real, a_aa[-1].imag))
        plotarray([float(a.real) for a in a_aa],
                  [phaa[n][0] for n in range(len(phaa))], 'a_atom-atom.pdf')
        exit()

    costr = ''
    for nn in range(1, zop):
        cf = twofac if (1 <= nn <= 28) else 0.0
        if (nn == 1):
            cf = int(withCoul)
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
                                          min_seedE=miniE_breed,
                                          ini_grid_bounds=width_bnds,
                                          ini_dims=deutDim,
                                          minC=minCond,
                                          evWin=evWindow,
                                          anzOptStates=nbrStatesOpti2)
        for cciv in new_civs:
            civs.append(cciv)
        print('>>> seed civilizations: %d/%d' % (len(civs), civ_size))
        if ((id_chan == 1) & (len(civs) > 1)):
            break

    civs.sort(key=lambda tup: np.abs(tup[3]))
    civs = sortprint(civs, pr=False)

    for nGen in range(anzGen):

        try:
            qualCUT, gsCUT, basCondCUT = civs[-int(len(civs) / 2)][2:]
            qualREF, gsREF, basCondREF = civs[0][2:]
        except:
            print('exception:')
            print(civs[-int(len(civs) / 2)][2:])
            print(civs[0][2:])
        # 3) select a subset of basis vectors which are to be replaced -----------------------------------------------

        civi_size = len(civs)
        weights = polynomial_sum_weight(civi_size, order=2)[1::]
        # 4) select a subset of basis vectors from which replacements are generated ----------------------------------
        children = 0

        while children < anzNewBV:
            twins = []
            while len(twins) < int(42 * anzNewBV):
                parent_pair = np.random.choice(range(civi_size),
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

                    prox_check1 = check_dist(width_array1=rw1,
                                             minDist=minidi_breed)
                    prox_check2 = check_dist(width_array1=rw2,
                                             minDist=minidi_breed)
                    prox_checkr1 = np.all([
                        check_dist(width_array1=rw1,
                                   width_array2=wsr,
                                   minDist=minidi_breed)
                        for wsr in widthSet_relative
                    ])

                    prox_checkr2 = np.all([
                        check_dist(width_array1=rw2,
                                   width_array2=wsr,
                                   minDist=minidi_breed)
                        for wsr in widthSet_relative
                    ])

                    #print(prox_check1, prox_check2, prox_checkr1, prox_checkr2)
                    if prox_check1 * prox_check2 * prox_checkr1 * prox_checkr2 == True:

                        wdau.append(list(rw1)[::-1])
                        wson.append(list(rw2)[::-1])

                        daughter = [mother[0], wdau, 0, 0, 0]
                        son = [mother[0], wson, 0, 0, 0]
                        twins.append(daughter)
                        twins.append(son)
                        #print('new')
                        #exit()

            sbas = []
            bv = two_body_channels[channel]
            bv = 0
            for n in range(len(twins[-1][0])):
                sbas += [[
                    two_body_channels[channel] + bv,
                    [1 for x in range(1, 1 + len(twins[-1][1][n]))]
                ]]
                bv += len(two_body_channels)

            if id_chan == 1:
                break

            ParaSets = [[
                twins[twinID][1], sbas, nnpotstring,
                float(J0), BINBDGpath, costr, twinID, minCond, evWindow,
                nbrStatesOpti2
            ] for twinID in range(len(twins))]

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
                    p = multiprocessing.Process(target=end2,
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
            #if fitchildren == 0:
            #    print('%d ' % children, end='')
            #else:
            #    print('adding %d new children.' % children)

        if id_chan == 1:
            break

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

    civs = sortprint(civs, pr=False, ordn=2)

    ma = blunt_ev2(cfgs=civs[0][0],
                   widi=civs[0][1],
                   basis=sbas,
                   nzopt=zop,
                   costring=costr,
                   binpath=BINBDGpath,
                   potNN=nnpotstring,
                   jay=float(J0),
                   funcPath=sysdir2)

    smartEV, parCond, smartRAT = smart_ev(ma, threshold=minCond)
    gsEnergy = smartEV[-1]

    print('\n> basType %s : C-nbr = %4.4e E0 = %4.4e   cMax/cMin = %e\n\n' %
          (channel, parCond, gsEnergy, smartRAT))

    #wset = get_quaf_width_set()
    #cof = parse_ev_coeffs()
    #cof2 = parse_ev_coeffs_2()
    #print('exp. coeff.   exp. coeff. (normalized)  width')
    #for bsb in range(len(wset)):
    #    print('{%12.6f , %12.6f , %12.6f}' %
    #          (float(cof[bsb]), float(cof2[bsb]), float(wset[bsb])))

    os.system('cp INQUA_N INQUA_N_%s' % (lam))
    os.system('cp OUTPUT bndg_out_%s' % (lam))
    os.system('cp INEN INEN_BDG')

    if id_chan == 1:
        os.system('cp %s/INQUA_N* .' % (refdir))
        subprocess.run([BINBDGpath + NNhamilEXE_serial])
        subprocess.run([BINBDGpath + spectralEXE_serial])
        os.system('cp OUTPUT bndg_out_%s' % (lam))

    os.system('cp INEN_STR INEN')
    subprocess.run([BINBDGpath + spectralEXE_serial])

    print(">>> calculating 2-body phases.")
    spole_2(nzen=100,
            e0=0.0001,
            d0=0.01,
            eps=[0.01 for n in range(len(channels_2))],
            bet=[2.1 for n in range(len(channels_2))],
            nzrw=100,
            frr=0.06,
            rhg=8.0,
            rhf=1.0,
            pw=0)

    subprocess.run([BINBDGpath + smatrixEXE])

    os.system('cp PHAOUT phaout_%s' % (lam))
    print(">>> 2-body phases calculated. End of day for channel %s\n" %
          channel)

    if deg_channs:
        id_chan = 1

    phaa = read_phase()
    if channel[:2] == 'pp':
        print('proton-proton channel:\n')
        a_aa = [
            appC(phaa[n][2] * np.pi / 180.,
                 np.sqrt(mn['137'] * phaa[n][0]),
                 0.5 * mn['137'],
                 q1=1,
                 q2=1) for n in range(len(phaa))
        ]
    else:
        a_aa = [
            -MeVfm * np.tan(phaa[n][2] * np.pi / 180.) /
            np.sqrt(mn['137'] * phaa[n][0]) for n in range(len(phaa))
        ]
    print('a_aa(E_min) = (%4.4f+i%4.4f) fm   a_aa(E_max) = (%4.4f+i%4.4f) fm' %
          (a_aa[0].real, a_aa[0].imag, a_aa[-1].real, a_aa[-1].imag))
    plotarray([float(a.real) for a in a_aa],
              [phaa[n][0] for n in range(len(phaa))], 'a_atom-atom.pdf')