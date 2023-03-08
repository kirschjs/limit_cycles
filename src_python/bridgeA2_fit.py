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

from scipy.optimize import fmin

las = lec_set.keys()
for lam in las:

    la = ('%-4.2f' % float(lam))[:4]

    sysdir2 = sysdir2base + '/fit'

    if os.path.isdir(sysdir2) == False:
        subprocess.check_call(['mkdir', '-p', sysdir2])

    os.chdir(sysdir2base)
    print('>>> working directory: ', sysdir2base)

    if os.path.isfile(sysdir2 + 'INQUA_N') == True:
        os.system('rm ' + sysdir2 + 'INQUA_N')

    pots = 'pot_nn_%02d' % int(float(la))

    if bin_suffix == '_v18-uix':
        prep_pot_file_2N(lam=lam, wiC=cloW, baC=cloB, ps2=nnpot)
    elif bin_suffix == '_eft-cib':
        prep_pot_file_2N_pp(lam=lam, wiC=cloW, baC=cloB, ppC=cpp, ps2=nnpot)
    else:
        print('no potential structure assigned to suffix.')
        exit()
    prep_pot_file_3N(lam=la, d10=d0, ps3=nnnpot)

    os.chdir(sysdir2)
    subprocess.call('cp %s .' % nnpot, shell=True)

    exit()
    chs = {
        # DEUTERON
        #'np-3SD1': [1, 1, [2, 10], [], [], 'SD', 'blue', sizeFrag, 1.0],
        #'np-3S1': [1, 1, [2], [], [], 'S', 'blue', sizeFrag, 1.01],
        #'np-1S0': [0, 0, [1], [], [], 'S', 'red', sizeFrag, 1.02],
        'np-1P1': [1, 0, [8], [], [], 'P', 'gray', sizeFrag, 1.0],
        'np-3P0': [0, 1, [9], [], [], 'P', 'red', sizeFrag, 0.5],
        #'np-3P1': [1, 1, [9], [], [], 'P', 'green', sizeFrag, 0.98],
        #'np-3P2': [2, 1, [9], [], [], 'P', 'blue', sizeFrag, 1.0],
        #'nn-1S0': [0, 0, [4], [], [], 'S', 'orange', sizeFrag, 0.9],
        #'nn-3P0': [0, 1, [5], [], [], 'P', 'red', sizeFrag, 0.81],
        #'nn-3P1': [1, 1, [5], [], [], 'P', 'green', sizeFrag, .99],
        #'nn-3P2': [2, 1, [5], [], [], 'P', 'blue', sizeFrag, 0.95]
        # NEUTERON
        #'nn-3PF2': [2, 1, [5, 7], [], [], 'PF', 'blue'],
        #'pp-1S0': [0, 0, [3], [], [], 'S', 'red', sizeFrag, 2.0]
        #'nn-3F2': [2, 1, 7, [], [], 'F']
        #'pp-3P0': [0, 1, 12, [], [], 'P', 'red'],
        #'pp-3P1': [1, 1, 12, [], [], 'P', 'green'],
        #'pp-3P2': [2, 1, 12, [], [], 'P', 'blue']
    }

    scale = 1.
    addw = 2

    plo = 0
    verb = 0
    h2_inlu(anzo=7)
    os.system(BINpath + 'LUDW_EFT_new.exe')
    h2_inob(anzo=5)
    os.system(BINpath + 'KOBER_EFT_nn.exe')

    phasSeum = [np.zeros(anze) for n in range(anzs)]

    for chan in chs:
        jay = chs[chan][0]
        stot = chs[chan][1]
        tmp1 = []
        tmp2 = []

        for n in range(len(eps_space)):

            if (('bdg' in cal) | ('reduce' in cal)):
                coeff = np.array(
                    #  coul, cent,p^2,r^2,            LS,        TENSOR, TENSOR_p
                    [1., pot_scale * eps_space[n], 0., 0, 0, 0, 0.])
                costr = ''
                for fac in coeff:
                    costr += '%12.6f' % float(fac)
                rw = wid_gen(add=addw,
                             w0=w120,
                             ths=[1e-6, 3e2, 0.2],
                             sca=chs[chan][8])
                h2_inqua(rw, pots)
                os.system(BINpath + 'QUAFL_' + mpii + '.exe')
                h2_inen_bs(relw=rw, costr=costr, j=jay, ch=chs[chan][2])
                os.system(BINpath + 'DR2END_' + mpii + '.exe')

                if dbg:
                    print(
                        'LS-scheme: B(2,%s,eps=%2.2f) = %4.4f MeV [' %
                        (chan, eps_space[n], get_h_ev()[0]), get_h_ev(n=4),
                        ']')
                rrgm_functions.parse_ev_coeffs()
                os.system('cp OUTPUT end_out_b && cp INEN inen_b')

            if 'scatt' in cal:
                h2_inen_str_pdp(relw=rw,
                                costr=costr,
                                j=jay,
                                sc=stot,
                                ch=chs[chan][2])
                os.system(BINpath + 'DR2END_' + mpii + '.exe')
                if verb: os.system('cp OUTPUT end_out_s && cp INEN inen_s')
                h2_spole(nzen=anze,
                         e0=0.001,
                         d0=.5,
                         eps=0.01,
                         bet=2.1,
                         nzrw=400,
                         frr=0.06,
                         rhg=8.0,
                         rhf=1.0,
                         pw=0)
                os.system(BINpath + 'S-POLE_PdP.exe')

                if verb:
                    os.system('cp PHAOUT pho_j=%d_l=%s_e=%f' %
                              (jay, lam, eps_space[n]))

                for cr in range(1, len(chs[chan][5]) + 1):
                    for cl in range(1, cr + 1):

                        phases = read_phase(phaout='PHAOUT',
                                            ch=[cl, cr],
                                            meth=1,
                                            th_shift='')
                        chs[chan][3].append(phases)
                        phasSeum[n] += np.array(phases)[:, 2]
                        chs[chan][4].append(eps_space[n])
                        print(
                            'a(NN,%s,eps=%2.2f) = %17f fm   @ E = %6.4f\n' %
                            (chan, eps_space[n],
                             -(float(chs[chan][3][0][1][0]) * mn[mpii] /
                               MeVfm**2)**(-1.5) * np.tan(
                                   float(chs[chan][3][0][1][2]) * np.pi / 180),
                             float(chs[chan][3][0][1][0])),
                            end='\n')

    if 'plot' in cal:
        if 'scatt' not in cal:
            print('ECCE >>> plotting phases without a scattering calc.')

        fig = plt.figure()
        chss = []
        ax1 = fig.add_subplot(1, 1, 1)
        ax1.set_xlabel(r'$k_{cm}\;\;[MeV]$', fontsize=21)
        ax1.set_ylabel(r'$\delta(nn)\;\; [deg]$', fontsize=21)
        ax1.set_title(
            r'$\Lambda=%d fm^{-1}:\;\;\;\;c(LS)=%2.3f\;\;\;\;\;c(S_{12})=%2.3f$'
            % (int(lam), ls * eps_space[-1], ten * eps_space[-1]),
            fontsize=21)

        if phasSeum:
            ax1.plot([np.sqrt(ph[0] * mn[mpii]) for ph in chs[chan][3][-1]],
                     phasSeum[0],
                     label=r'$\sum_\delta(\epsilon=0)$',
                     color='black')
            ax1.plot([np.sqrt(mn[mpii] * ph[0]) for ph in chs[chan][3][-1]],
                     phasSeum[-1],
                     label=r'$\sum_\delta(\epsilon=max.)$',
                     color='black',
                     linestyle='dotted')

        nn = 1

        for chan in chs:

            ax1.plot([np.sqrt(mn[mpii] * ph[0]) for ph in chs[chan][3][0]
                      ][:int(len(chs[chan][3][0]) * (10 - nn) / 10)],
                     [ph[2] for ph in chs[chan][3][0]
                      ][:int(len(chs[chan][3][0]) * (10 - nn) / 10)],
                     label=r'$C_2(%s)=%.2E$' % (chan, chs[chan][4][0]),
                     color=chs[chan][-3],
                     linewidth=5 - nn)
            ax1.plot([np.sqrt(mn[mpii] * ph[0]) for ph in chs[chan][3][-1]],
                     [ph[2] for ph in chs[chan][3][-1]],
                     label=r'$C_2(^%1d%s_%1d)=%.2E$' %
                     (int(2 * chs[chan][1] + 1), chs[chan][-4], chs[chan][0],
                      chs[chan][4][-1]),
                     color=chs[chan][-3],
                     linestyle='dashed',
                     linewidth=2,
                     alpha=0.5)
            nn += 1
            [
                ax1.plot(
                    [np.sqrt(mn[mpii] * ph[0]) for ph in chs[chan][3][-1]],
                    [ph[2] for ph in chs[chan][3][n]],
                    color=chs[chan][-3],
                    alpha=0.25,
                    lw=1) for n in range(1,
                                         len(chs[chan][3]) - 1)
            ]
            leg = ax1.legend(loc='best')
            #plt.ylim(-.1, .1)

plt.show()