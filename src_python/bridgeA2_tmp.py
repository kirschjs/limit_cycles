import subprocess
import os, fnmatch, copy, struct
import numpy as np
import sympy as sy
# CG(j1, m1, j2, m2, j3, m3)
from sympy.physics.quantum.cg import CG
from scipy.linalg import eigh
from scipy.stats import truncnorm, norm

from three_particle_functions import *
from PSI_parallel_M import *
from rrgm_functions import *
from genetic_width_growth import *
from plot_dist import *
from parameters_and_constants import *

from scipy.optimize import fmin
import multiprocessing
from multiprocessing.pool import ThreadPool

# flag to be set if after the optimization of the model space, a calibration within
# that space to an observable is ``requested''
fitt = 0

# which eigenstate whould have the specified target value? fixi=-1 = ground-state fitting
fixi = -2
# energy to fit to
deub = 2.22

gTy = ['log_with_density_enhancement', 0.0001, 0.0002]  #'log',  #

lecFile = '/home/kirscher/kette_repo/limit_cycles/manuscript/graphs/LECS.dat'
lecFile = '/home/kirscher/Documents/vault/Vorlesungen/num_methods/lec_ex1_b2.22.dat'
lecFile = '/home/kirscher/Documents/vault/Vorlesungen/num_methods/lec_ex0_b2.22_gs3.dat'
lec_set = np.array([line.split() for line in open(lecFile)
                    if line[0] != '#']).astype(float)

# 0:0.5 1:1.0 2:2.22 3:5 4:7 5:10
bdimer = 0
# 0:4 1:6 2:8 3:10
las = lec_set[:, 0]
lamstart, lamend = 0, 3  #len(las) - 1

# numerical stability
minCond = 10**-16
minidi_breed = 410.1
minidi_seed = minidi_breed
minidi_breed_rel = minidi_breed
denseEVinterval = [-2, 2]
width_bnds = [0.001, 430.25]

deutDim = 10

miniE_breed = -0.0

# genetic parameters
anzNewBV = 4
muta_initial = 0.015
anzGen = 35
civ_size = 22
target_pop_size = 32

# define a random distribution from which width parameters are chose if and only if
# the binary intertwining operation yields values outside the acceptable interval
loc, scale = 1.3, 100.5  # TODO, loc should be where choosen s.t. the Gaussian having the same width as the exponential prop. density

a_transformed, b_transformed = (width_bnds[0] - loc) / scale, (width_bnds[1] -
                                                               loc) / scale
rv = truncnorm(a_transformed, b_transformed, loc=loc, scale=scale)
x = np.linspace(truncnorm.ppf(0.01, width_bnds[0], width_bnds[1]),
                truncnorm.ppf(1, width_bnds[0], width_bnds[1]), 100)

r = rv.rvs(size=10000)

fig, ax = plt.subplots(1, 1)
ax.plot(x, rv.pdf(x), 'k-', lw=2, label='frozen pdf')
ax.hist(r, density=True, bins='auto', histtype='stepfilled', alpha=0.2)
ax.set_xlim(width_bnds[0], width_bnds[1])
ax.legend(loc='best', frameon=False)

fig.savefig("default_breeding_widths_dimer.pdf")

zop = 14 if bin_suffix == '_v18-uix' else 11

ftFac = []

for nlam in range(lamstart, lamend):

    nbrStatesOpti2 = list(range(fixi, fixi + 1))

    lam = las[nlam]
    channel = 'np3s'
    nnpott = nnpot + str(lam)
    nnnpott = nnnpot + str(lam)
    nnpotstringt = nnpotstring + str(lam)
    nnnpotstringt = nnnpotstring + str(lam)

    J0 = channels_2[channel][1]

    sysdir2 = sysdir2base + '/' + channel

    if os.path.isdir(sysdir2) == False:
        subprocess.check_call(['mkdir', '-p', sysdir2])

    subprocess.call('rm -rf %s/civ_*' % sysdir2, shell=True)

    os.chdir(sysdir2base)
    print('>>> working directory: ', sysdir2)

    cloB = 0.0
    cloW = lec_set[nlam, 2 + bdimer]

    if bin_suffix == '_v18-uix':
        prep_pot_file_2N(lam=(2 * np.sqrt(float(lam))),
                         wiC=cloW,
                         baC=cloB,
                         ps2=nnpott)
    elif bin_suffix == '_eft-cib':
        prep_pot_file_2N_pp(lam=2 * np.sqrt(float(lam)),
                            wiC=cloW,
                            baC=cloB,
                            ppC=cpp,
                            ps2=nnpott)
    else:
        print('no potential structure assigned to suffix.')
        exit()

    d00 = lec_set[nlam, 2 + bdimer]
    print('>>> l = %f fm  C(l) = %f MeV  D(l) = %f MeV' % (lam, cloW, d00))
    prep_pot_file_3N(lam=2 * np.sqrt(float(lam)), d10=d00, ps3=nnnpott)

    os.chdir(sysdir2)
    if id_chan == 0:
        refdir = sysdir2
    subprocess.call('cp %s .' % nnpott, shell=True)
    subprocess.call('cp %s .' % nnnpott, shell=True)

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

        nbrStatesOpti2 = list(range(fixi, -0))
        new_civs, basi = span_population2(anz_civ=int(3 * civ_size),
                                          fragments=channel,
                                          Jstreu=float(J0),
                                          coefstr=costr,
                                          funcPath=sysdir2,
                                          binPath=BINBDGpath,
                                          mindist=minidi_seed,
                                          min_seedE=miniE_breed,
                                          gridType=gTy,
                                          ini_grid_bounds=width_bnds,
                                          ini_dims=deutDim,
                                          minC=minCond,
                                          evWin=evWindow,
                                          optRange=nbrStatesOpti2)
        for cciv in new_civs:
            civs.append(cciv)
        print('>>> seed civilizations: %d/%d' % (len(civs), civ_size))
        if ((id_chan == 1) & (len(civs) > 1)):
            break

    civs.sort(key=lambda tup: np.linalg.norm(tup[3]))
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
            while len(twins) < int(2 * anzNewBV):
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
                                     mutation_rate=muta_initial,
                                     wMin=0.5 * width_bnds[0],
                                     wMax=2.5 * width_bnds[1],
                                     dbg=False,
                                     def1=rv.rvs(),
                                     def2=rv.rvs(),
                                     method='2point')
                        for n in range(len(mother[1][wset]))
                    ]
                    rw1 = np.array(daughterson)[:, 0]  #.sort()
                    rw1.sort()
                    rw2 = np.array(daughterson)[:, 1]  #.sort()
                    rw2.sort()

                    #print(rw1, rw1, rw2, rw2)
                    #exit()

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
                    if prox_check1 == prox_check2 == prox_checkr1 == prox_checkr2 == False:

                        wdau.append(list(rw1)[::-1])
                        wson.append(list(rw2)[::-1])

                        daughter = [mother[0], wdau, 0, 0, 0]
                        son = [mother[0], wson, 0, 0, 0]
                        twins.append(daughter)
                        twins.append(son)
                        #exit()
                    else:
                        #print('sibblings too close!')
                        continue

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
                twins[twinID][1], sbas, nnpotstringt,
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
            # wave-function printout (ECCE: in order to work, in addition to the civs[0] argument,
            # I need to hand over the superposition coeffs of the wfkt)
            #write_indiv3(civs[0], outfile)
            print(
                '(Gen., Opt cond., Opt lowest EVs) = %d , %4.4e' %
                (nGen, civs[0][4]), civs[0][3])

    print('\n\n')

    civs = sortprint(civs, pr=False, ordn=2)

    ma = blunt_ev2(cfgs=civs[0][0],
                   widi=civs[0][1],
                   basis=sbas,
                   nzopt=zop,
                   costring=costr,
                   binpath=BINBDGpath,
                   potNN=nnpotstringt,
                   jay=float(J0),
                   funcPath=sysdir2)

    smartEV, parCond, smartRAT = smart_ev(ma, threshold=minCond)
    gsEnergy = smartEV[-1]

    print('\n> basType %s : C-nbr = %4.4e E0 = %4.4e   cMax/cMin = %e\n\n' %
          (channel, parCond, gsEnergy, smartRAT))

    mmout = 1
    if mmout:
        wset = get_quaf_width_set()
        cof = parse_ev_coeffs()
        cof2 = parse_ev_coeffs_2()
        print(
            'exp. coeff.   exp. coeff. (normalized)  width   (for lambda = %f fm^-1)'
            % float(lam))
        wstr = '{'
        for bsb in range(len(wset)):
            wstr += '%12.6f ,' % float(wset[bsb])
            print('{%12.6f , %12.6f , %12.6f}' %
                  (float(cof[bsb]), float(cof2[bsb]), float(wset[bsb])))
        wstr = wstr[:-2] + '}'
        print(wstr)

    os.system('cp INQUA_N INQUA_N_%s' % (lam))
    os.system('cp OUTPUT bndg_out_%s' % (lam))
    os.system('cp INEN INEN_BDG')

    def fitti(fac, fitb, fix=-1):
        repl_line(
            'INEN', 2, '%+12.6f%+12.6f%+12.6f%+12.6f%+12.6f%+12.6f%+12.6f\n' %
            (0.0, fac, 1.0, 1.0, 1.0, 1.0, 1.0))
        subprocess.run([BINBDGpath + spectralEXE_serial])
        matout = np.core.records.fromfile('MATOUTB', formats='f8', offset=4)
        smartEV, parCond, gsRatio = smart_ev(matout, threshold=10**-10)
        print(np.real(smartEV[-4:]))
        E_0 = np.real(smartEV[fix])
        print(abs(float(E_0) + fitb))
        return abs(float(E_0) + fitb)

    if fitt:

        # initial scaling factor from which the root-finding algorithm commences its search
        fac = 0.97

        ft_lo = fmin(fitti, fac, args=(deub, fixi), disp=False)

        ftFac.append([lam, cloW * ft_lo[0]])
        res_lo = fitti(fac=ft_lo[0], fitb=0.0, fix=fixi)
        print(
            'L = %2.2f:  C_opt = %12.4f => B(2,%d-ex) = %8.4f   ;  C_start = %12.4f'
            % (lam, cloW * ft_lo[0], np.abs(fixi) - 1, res_lo, cloW))

if fitt:
    outs = ''
    for n in range(len(ftFac)):
        outs += '%4.4f  %20.8f\n' % (ftFac[n][0], ftFac[n][1])
        print('%4.4f  %20.8f' % (ftFac[n][0], ftFac[n][1]))

    with open('lec_ex%d_b%2.2f.dat' % (np.abs(fixi) - 1, deub),
              'w') as outfile:
        outfile.write(outs)