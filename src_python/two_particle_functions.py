from pathlib import Path

import os, re
import numpy as np
import random
import rrgm_functions, parameters_and_constants


def h2_inen_str_pdp(relw, costr, j=0, sc=0, ch=[1]):
    s = ''
    s += ' 10  2 12  9  1  1 -0  0  0 -1\n'
    s += '  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1\n'
    #
    s += '%s\n' % costr
    #     2*J #ch s/b
    s += '%4d%4d   0   0   2\n' % (int(2 * j), len(ch))
    for c in ch:
        s += '  1  1%3d\n' % int(2 * sc)
        s += '   1%4d\n' % int(c)
        for rr in range(1, 24):
            s += '  1'
        s += '\n'
    with open('INEN', 'w') as outfile:
        outfile.write(s)
    return


def spole_2(nzen=20,
            e0=0.01,
            d0=0.075,
            eps=[0.01],
            bet=1.1,
            nzrw=400,
            frr=0.06,
            rhg=8.0,
            rhf=1.0,
            pw=1,
            nbrCH=5,
            adaptweightUP=1.0,
            adaptweightLOW=0.009,
            adaptweightL=0.5,
            GEW=1.0,
            QD=1.8,
            QS=0.0):
    if nbrCH > 5:
        print(
            'ECCE (S-POLE): calculation for > 5 physical channels untested!\nViewer discretion is strongly advised.'
        )

    if len(eps) != nbrCH:
        print(
            'ECEE(INPUTSPOLE): number of channels does not match number of eps parameters!'
        )

    s = ''
    s += ' 11  3  0  0  0 +0\n'
    s += '%3d  0  1\n' % int(nzen)
    s += '%12.7f%12.7f\n' % (float(e0), float(d0))
    epsline = ''.join(['%12.8f' % float(epsI) for epsI in eps]) + '\n'
    betline = ''.join(['%12.8f' % float(betI) for betI in bet]) + '\n'
    s += epsline + betline
    #    OUT
    s += ' +1  0 +1  0  1  0  2  0\n'
    s += '%3d\n' % int(nzrw)
    s += '%12.8f%12.8f%12.8f\n' % (float(frr), float(rhg), float(rhf))
    channelDescriptorline = ''.join(['%3d' % n
                                     for n in range(1, nbrCH + 1)]) + '\n'
    s += channelDescriptorline
    adaptIntervalWeightlineUP = nbrCH * ('%12.8f' %
                                         float(adaptweightUP)) + '\n'
    adaptIntervalWeightlineLOW = nbrCH * ('%12.8f' %
                                          float(adaptweightLOW)) + '\n'
    adaptIntervalWeightlineL = nbrCH * ('%12.8f' % float(adaptweightL)) + '\n'
    s += adaptIntervalWeightlineUP + adaptIntervalWeightlineLOW + adaptIntervalWeightlineL
    s += '%12.8f%12.8f%12.8f\n' % (GEW, QD, QS)
    with open('INPUTSPOLE', 'w') as outfile:
        outfile.write(s)
    return


def inlu_2(anzo=5, anzf=1):
    s = ''
    s += '  9\n'
    for n in range(anzo):
        s += '  1'
    s += '\n%3d\n' % anzf
    for n in range(anzf):
        s += '  4  2\n'
    for n in range(anzf):
        s += '  0\n  1\n  2\n  3\n'
    with open('INLUCN', 'w') as outfile:
        outfile.write(s)
    return


def inob_2exp(anzo=5, anzf=1):
    s = ''
    s += '  0  0\n'
    for n in range(anzo):
        s += '  1'
    s += '\n  4\n'
    s += '%3d  2\n' % anzf
    for n in range(anzf):
        s += '  2 15  6  1\n'
        s += '  1  1\n'
        s += '  1  3\n'  #  1) p-up, n-up
        s += '  1  2\n'  #  2) NN - singlet
        s += '  3  4\n'
        s += '  1  4\n'
        s += '  3  2\n'
        s += '  4  1\n'
        s += '  2  3\n'
        s += '  2  1\n'
        s += '  4  3\n'
        s += '  4  3\n'
        s += '  1  2\n'
        s += '  2  1\n'
        s += '  3  4\n'
        s += '  4  3\n'
        s += '  3  4\n'  #  6)  n up n down
        s += '  1  1\n'  # p-up, p-up
        s += '  0  1  1  2\n'
        s += '  0  1  1  2\n'
        s += '  0  1  1  2\n'
        s += '  0  1  1  2\n'
        s += '  0  1 -1  2\n'
        s += '  0  1 -1  2\n'
        s += '  0  1 -1  2\n'
        s += '  0  1 -1  2\n'
        s += '  0  1  0  1  1  4\n'
        s += '  0  1  0  1 -1  4\n'
        s += '  0  1  0  1  0  1  1  2\n'
        s += '  0  1  0  1  0  1 -1  2\n'
        s += '  0  1  0  1  0  1  0  1  1  1\n'
        s += '  0  1  0  1  0  1  0  1  0  1  1  1\n'
    with open('INOB', 'w') as outfile:
        outfile.write(s)
    return


def inob_2(anzo=5, anzf=1):
    s = ''
    s += '  0  0\n'
    for n in range(anzo):
        s += '  1'
    s += '\n  4\n'
    s += '%3d  2\n' % anzf
    for n in range(anzf):
        s += '  2  9  6  1\n'
        s += '  1  1\n'
        s += '  1  3\n'  #  1) p-up, n-up
        s += '  1  4\n'  #  2) NN - singlet
        s += '  2  3\n'
        s += '  4  3\n'
        s += '  1  2\n'
        s += '  2  1\n'
        s += '  3  4\n'
        s += '  4  3\n'
        s += '  3  4\n'  #  6)  n up n down
        s += '  1  1\n'  # p-up, p-up
        s += '  0  1  1  2\n'
        s += '  0  1 -1  2\n'
        s += '  0  1  0  1  1  4\n'
        s += '  0  1  0  1 -1  4\n'
        s += '  0  1  0  1  0  1  1  2\n'
        s += '  0  1  0  1  0  1 -1  2\n'
        s += '  0  1  0  1  0  1  0  1  1  1\n'
        s += '  0  1  0  1  0  1  0  1  0  1  1  1\n'
    with open('INOB', 'w') as outfile:
        outfile.write(s)
    return


def inqua_2(relw, ps2, inquaout='INQUA_N'):
    s = ''
    s += ' 10  8  9  3 00  0  0  0  0\n'
    #s += pot_dir + ps2 + '\n'
    s += ps2
    for n in range(len(relw)):
        s += '\n 13'
        s += '\n  1%3d\n' % int(len(relw[n]))
        s += '.0          .0\n'

        for rw in range(0, len(relw[n])):
            s += '%12.6f' % float(relw[n][rw])
            if ((rw != (len(relw[n]) - 1)) & ((rw + 1) % 6 == 0)):
                s += '\n'
        s += '\n'
        s += '  2  1\n1.\n'  # 1:  n-p 1S0
        s += '  1  1\n1.\n'  # 2:  n-p 3S1
        # ------------
        s += '  5  1\n1.\n'  # 3:  n-n 1S0
        s += '  5  2\n1.\n'  # 4:  n-n 3P0,1,2
        s += '  4  3\n1.\n'  # 5:  n-n 1D2
        s += '  5  4\n1.\n'  # 6:  n-n 3F2,3,4
        # ------------
        s += '  2  2\n1.\n'  # 7:  n-p 1P1
        s += '  1  2\n1.\n'  # 8:  n-p 3P0,1,2
        s += '  1  3\n1.\n'  # 9: n-p 3D1
        # ------------
        s += '  3  1\n1.\n'  # 10:  p-p 1S0
        s += '  6  2\n1.\n'  # 11:  p-p 3P0,1,2
        s += '  3  3\n1.\n'  # 12:  p-p 1D2
        s += '  6  4\n1.'  # 13:  p-p 3F2,3,4
    with open(inquaout, 'w') as outfile:
        outfile.write(s)
    return
    # r7 c2:   S  L           S_c
    #  1   :   0  0  1S0         0
    #  2   :   1  0  3S1         2
    #  3   :   0  0  1S0         0          p-p
    #  4   :   0  0  1S0         0          n-n
    #  5   :   1  1  3P0,3P1,3P2 2          n-n
    #  6   :   0  2  1D2         2          n-n
    #  7   :   0  1  1P1         0
    #  8   :   1  1  3P0,3P1,3P2 2
    #  9   :   1  2  3D1         2


def inen_bdg_2(bas, costr, j, ch=1, anzo=14, fn='INEN', pari=0, tni=10):
    s = ''
    s += ' 10  3 11%3d  1  1  0  0  0 -1\n' % int(anzo)
    #       N  T Co CD^2 LS  T
    s += '  1  1  1  1  1  1  1  1  1  1\n'

    s += '%s\n' % costr

    #     2*J #ch s/b
    s += '%3d%3d  1  0  2\n' % (int(2 * j), len(bas))

    for bv in bas:
        s += '%3d%3d\n' % (1, bv[0])

        tmp = ''

        for n in range(1, 1 + len(bv[1])):
            tmp += '%3d' % int(1)

        tmp += '\n'
        s += tmp

    with open(fn, 'w') as outfile:
        outfile.write(s)
    return


def inen_str_2(costr, anzrelw=20, j=0, sc=0, ch=1, anzo=14, fn='INEN'):
    s = ''
    s += ' 10  2 12%3d  1  1 -0  0  0 -1\n' % int(anzo)
    #      N  T Co  CD^2 LS  T
    s += '  1  1  1  1  1  1  1  1  1  1\n'
    #
    s += '%s\n' % costr
    #     2*J #ch s/b
    s += '%4d   1   0   0   2\n' % int(2 * j)
    s += '  1  1%3d\n' % int(2 * sc)
    s += '   1%4d\n' % int(ch)
    for rr in range(1, anzrelw + 1):
        if ((rr % 30 == 0)):
            s += '  1'
            s += '\n'
        s += '  1'
    with open(fn, 'w') as outfile:
        outfile.write(s)
    return