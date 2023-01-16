from pathlib import Path

import os, re
import numpy as np
import random
import rrgm_functions, parameters_and_constants

elem_spin_prods_5 = {
    # (he4-n)
    'tpn_1s0h':
    '  5 12  1  4        tp-n   No1: S=0\n  2  4  1\n  1  1  1  1  1\n  4  2  3  1  3\n  4  1  4  1  3\n  4  1  3  2  3\n  2  4  3  1  3\n  2  3  4  1  3\n  2  3  3  2  3\n  3  2  4  1  3\n  3  2  3  2  3\n  3  1  4  2  3\n  1  4  4  1  3\n  1  4  3  2  3\n  1  3  4  2  3\n  1 12\n -1 48\n -1 48\n -1 12\n  1 48\n  1 48\n -1 48\n -1 48\n  1 12\n  1 48\n  1 48\n -1 12\n',
    'tpn_6s0h':
    '  5 12  1  4        tp-n   No6: S=0\n  2  4  1\n  1  1  1  1  1\n  4  3  2  1  3\n  4  3  1  2  3\n  4  1  4  1  3\n  4  1  3  2  3\n  2  3  4  1  3\n  2  3  3  2  3\n  3  4  2  1  3\n  3  4  1  2  3\n  3  2  4  1  3\n  3  2  3  2  3\n  1  4  4  1  3\n  1  4  3  2  3\n  1 12\n -1 12\n -1 48\n  1 48\n -1 48\n  1 48\n -1 12\n  1 12\n  1 48\n -1 48\n  1 48\n -1 48\n',
    'henn_1s0h':
    '  5 12  1  4        hen-n  No1: S=0\n  2  4  1\n  1  1  1  1  1\n  4  2  1  3  3\n  4  1  2  3  3\n  4  1  1  4  3\n  2  4  1  3  3\n  2  3  2  3  3\n  2  3  1  4  3\n  3  2  2  3  3\n  3  2  1  4  3\n  3  1  2  4  3\n  1  4  2  3  3\n  1  4  1  4  3\n  1  3  2  4  3\n -1 12\n  1 48\n  1 48\n  1 12\n -1 48\n -1 48\n  1 48\n  1 48\n -1 12\n -1 48\n -1 48\n  1 12\n',
    'henn_6s0h':
    '  5 12  1  4        hen-n  No6: S=0\n  2  4  1\n  1  1  1  1  1\n  4  1  2  3  3\n  4  1  1  4  3\n  2  3  2  3  3\n  2  3  1  4  3\n  2  1  4  3  3\n  2  1  3  4  3\n  3  2  2  3  3\n  3  2  1  4  3\n  1  4  2  3  3\n  1  4  1  4  3\n  1  2  4  3  3\n  1  2  3  4  3\n -1 48\n  1 48\n -1 48\n  1 48\n  1 12\n -1 12\n  1 48\n -1 48\n  1 48\n -1 48\n -1 12\n  1 12\n',
    'dist_5':
    '  5  1  1  4        polarized 5-par\n  2  4  1\n  1  1  1  1  1\n  1  2  3  4  5\n  1  1\n',
}


def inlu_5(anzO, fn='INLU', fr=[], indep=0):
    out = '  0  0  0  0  0%3d\n' % indep
    for n in range(anzO):
        out += '  1'
    out += '\n%d\n' % len(fr)
    for n in range(0, len(fr)):
        out += '  1  5\n'

    zerle = {
        '000-0-0': '  0  0  0  0\n  0  1  2\n  0  5  3\n  0  6  4\n',
        '000-0-1': '  0  0  0  1\n  0  1  2\n  0  5  3\n  1  6  4\n',
    }

    for n in fr:
        out += zerle[n]

    for n in range(len(fr)):
        for m in range(len(fr)):
            out += '%s_%s\n' % (fr[n], fr[m])

    with open(fn, 'w') as outfile:
        outfile.write(out)


def inob_5(fr, anzO, fn='INOB', indep=0):
    #                IBOUND => ISOSPIN coupling allowed
    out = '  0  0  0  0%3d\n' % indep
    for n in range(anzO):
        out += '  1'
    out += '\n  4\n%3d  5\n' % len(fr)

    for n in fr:
        out += elem_spin_prods_5[n]

    for n in range(len(fr)):
        for m in range(len(fr)):
            out += '%s_%s\n' % (fr[n], fr[m])

    with open(fn, 'w') as outfile:
        outfile.write(out)


def inqua_5(intwi=[], relwi=[], potf='', inquaout='INQUA_M'):
    s = ''
    # NBAND1,NBAND2,NBAND3,NBAND4,NBAND5,NAUS,MOBAUS,LUPAUS,NBAUS
    s += ' 10  8  9  3 00  0  0  0  0\n%s\n' % potf
    zerl_counter = 0
    bv_counter = 1
    for n in range(len(intwi)):

        zerl_counter += 1
        nrel = len(relwi[n])
        nb = int(len(intwi[n]) / 2)
        s += '%3d%60s%s\n%3d%3d\n' % (
            nb, '', 'Z%d  BVs %d - %d' %
            (zerl_counter, bv_counter, bv_counter - 1 + nb), nb, nrel)

        bv_counter += nb
        for bv in range(nb):
            s += '%48s%-12.6f%-12.6f\n' % ('', float(
                intwi[n][2 * bv]), float(intwi[n][1 + 2 * bv]))

        for rw in range(0, len(relwi[n])):
            s += '%12.6f' % float(relwi[n][rw])
            if ((rw != (len(relwi[n]) - 1)) & ((rw + 1) % 6 == 0)):
                s += '\n'
        s += '\n'

        tmpln = np.ceil(nb / 6.)
        for bb in range(0, nb):
            s += '  1  1\n'
            for i in range(int(bb / 6)):
                s += '\n'
            s += '1.'.rjust(12 * (bb % 6 + 1))

            for ii in range(int(tmpln - int(bb / 6))):
                s += '\n'

    with open(inquaout, 'w') as outfile:
        outfile.write(s)

    return


def inen_bdg_5(bas, jay, co, fn='INEN', pari=0, nzop=31, tni=11, idum=2):
    # idum=2 -> I4 for all other idum's -> I3
    # NBAND1,IDUM,NBAND3,NZOP,IFAKD,IGAK,NZZ,IAUW,IDRU,IPLO,IDUN,ICOPMA(=1 -> stop after N,H output)
    head = '%3d%3d 12%3d  1  0 +0  0  0 -1  0 +0\n' % (tni, idum, nzop)
    head += '  1  1  1  1  0  0  0  0  0  0  0  0  0  0  1  1\n'

    head += co + '\n'

    out = ''
    if idum == 2:
        out += '%4d%4d   1  -1%4d\n' % (int(2 * jay), len(bas), pari)
    else:
        out += '%3d%3d  1 -1%3d\n' % (int(2 * jay), len(bas), pari)

    relset = False

    for bv in bas:
        if idum == 2:
            out += '%4d%4d\n' % (1, bv[0])
        else:
            out += '%3d%3d\n' % (1, bv[0])

        tmp = ''

        for n in bv[1]:
            tmp += '%3d' % int(n)

        tmp += '\n'
        out += tmp

    with open(fn, 'w') as outfile:
        outfile.write(head + out)


def inen_str_5(coeff,
               wr,
               bvs,
               uec,
               phys_chan,
               dma=[1, 0, 1, 0, 1, 0, 1],
               jay=0,
               anzch=1,
               pari=0,
               nzop=9,
               tni=10,
               fn='INEN_STR'):

    s = '%3d  2 12%3d  1  1 +2  0  0 -1\n' % (tni, nzop)

    s += '  1  1  1  1  0  0  0  0  0  0  0  0  0  0  1  1\n'

    s += coeff + '\n'

    # SPIN #CHANNELS
    s += '%4d%4d   0   0%4d   1\n' % (int(2 * jay), anzch, pari)

    # FRAGMENT-EXPANSION COEFFICIENTS

    s += '%4d\n' % len(uec)

    for cf in uec:
        s += '%-20.9f\n' % cf

    # ------------------------------------ phys chan
    chanstrs = []
    stmp = ''
    stmp += '%3d%3d%3d\n' % (phys_chan[0], phys_chan[1], phys_chan[2])
    stmp += '%4d' % len(uec)
    di = 1
    for i in range(len(uec)):
        di += 1
        stmp += '%4d' % (i + 1)
        if ((di % 20 == 0) | (int(i + 1) == len(uec))):
            stmp += '\n'
            di = 0
    di = 1
    for i in range(1, 1 + len(uec)):
        stmp += '%4d' % i
        if ((di % 20 == 0) | (di == len(uec))):
            stmp += '\n'
        di += 1

    nbr_relw_phys_chan = len(wr)
    for i in range(1, nbr_relw_phys_chan + 1):
        stmp += '%3d' % int(1)
        if ((int(i) % 50 == 0) | (int(i) == nbr_relw_phys_chan)):
            stmp += '\n'
    chanstrs.insert(0, stmp)

    s += ''.join(chanstrs)

    distuec = [
        nc + 1 for nc in range(len(uec)) if 10**2 > np.abs(uec[nc]) > 0.1
    ]

    fd = True
    for nphy_chan in range(len(phys_chan)):
        relwoffset = ''
        for i in range(1, len(uec) - 4):
            s += '%3d%3d%3d' % (phys_chan[0], phys_chan[1], phys_chan[2])
            if fd:
                s += ' -1\n'
                fd = False
            else:
                s += '\n'
            s += '   1%4d\n' % i
            s += '%-4d\n' % (np.random.choice(distuec))
            s += relwoffset
            for relw in dma:
                s += '%3d' % relw
            s += '\n'

    with open(fn, 'w') as outfile:
        outfile.write(s)
    return


def spole_5(nzen=20,
            e0=0.05,
            d0=0.5,
            eps=0.01,
            bet=1.1,
            nzrw=100,
            frr=0.06,
            rhg=8.0,
            rhf=1.0,
            pw=0):
    s = ''
    s += ' 11  3  0  0  0  1\n'
    s += '%3d  0  0\n' % int(nzen)
    s += '%12.4f%12.4f\n' % (float(e0), float(d0))
    s += '%12.4f%12.4f%12.4f%12.4f%12.4f%12.4f\n' % (
        float(eps), float(eps), float(eps), float(eps), float(eps), float(eps))
    s += '%12.4f%12.4f%12.4f%12.4f%12.4f%12.4f\n' % (
        float(bet), float(bet), float(bet), float(bet), float(bet), float(bet))
    #    OUT
    s += '  0  0  1  0  1  0 -0  0\n'
    s += '%3d\n' % int(nzrw)
    s += '%12.4f%12.4f%12.4f\n' % (float(frr), float(rhg), float(rhf))
    s += '  1  2  3  4\n'
    s += '0.0         0.0         0.0         0.0         0.0         0.0\n'
    s += '.001        .001        .001        .001        .001        .001\n'
    for weight in pw:
        s += '%12.4f' % float(weight)
    s += '\n'
    s += '1.          1.          0.\n'
    with open('INPUTSPOLE', 'w') as outfile:
        outfile.write(s)
    return