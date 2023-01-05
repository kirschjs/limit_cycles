from pathlib import Path

import os, re
import numpy as np
import random
import rrgm_functions, parameters_and_constants

elem_spin_prods_5 = {
    # (he4-n)
    'tpn_1s0h':
    '  5 12  1  4        tp-n   No1: S=0\n  2  4  1\n  1  1  1  1  1\n  4  2  3  1  3\n  4  1  4  1  3\n  4  1  3  2  3\n  2  4  3  1  3\n  2  3  4  1  3\n  2  3  3  2  3\n  3  2  4  1  3\n  3  2  3  2  3\n  3  1  4  2  3\n  1  4  4  1  3\n  1  4  3  2  3\n  1  3  4  2  3\n  1 12\n -1 48\n -1 48\n -1 12\n  1 48\n  1 48\n -1 48\n -1 48\n  1 12\n  1 48\n  1 48\n -1 12',
    'tpn_6s0h':
    '  5 12  1  4        tp-n   No6: S=0\n  2  4  1\n  1  1  1  1  1\n  4  3  2  1  3\n  4  3  1  2  3\n  4  1  4  1  3\n  4  1  3  2  3\n  2  3  4  1  3\n  2  3  3  2  3\n  3  4  2  1  3\n  3  4  1  2  3\n  3  2  4  1  3\n  3  2  3  2  3\n  1  4  4  1  3\n  1  4  3  2  3\n  1 12\n -1 12\n -1 48\n  1 48\n -1 48\n  1 48\n -1 12\n  1 12\n  1 48\n -1 48\n  1 48\n -1 48',
    'henn_1s0h':
    '  5 12  1  4        hen-n  No1: S=0\n  2  4  1\n  1  1  1  1  1\n  4  2  1  3  3\n  4  1  2  3  3\n  4  1  1  4  3\n  2  4  1  3  3\n  2  3  2  3  3\n  2  3  1  4  3\n  3  2  2  3  3\n  3  2  1  4  3\n  3  1  2  4  3\n  1  4  2  3  3\n  1  4  1  4  3\n  1  3  2  4  3\n -1 12\n  1 48\n  1 48\n  1 12\n -1 48\n -1 48\n  1 48\n  1 48\n -1 12\n -1 48\n -1 48\n  1 12',
    'henn_6s0h':
    '  5 12  1  4        hen-n  No6: S=0\n  2  4  1\n  1  1  1  1  1\n  4  1  2  3  3\n  4  1  1  4  3\n  2  3  2  3  3\n  2  3  1  4  3\n  2  1  4  3  3\n  2  1  3  4  3\n  3  2  2  3  3\n  3  2  1  4  3\n  1  4  2  3  3\n  1  4  1  4  3\n  1  2  4  3  3\n  1  2  3  4  3\n -1 48\n  1 48\n -1 48\n  1 48\n  1 12\n -1 12\n  1 48\n -1 48\n  1 48\n -1 48\n -1 12\n  1 12',
}


def inlu_5(anzO, fn='INLU', fr=[], indep=0):
    out = '  0  0  0  0  0%3d\n' % indep
    for n in range(anzO):
        out += '  1'
    out += '\n%d\n' % len(fr)
    for n in range(0, len(fr)):
        out += '  1  4\n'

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
    out += '\n  4\n%3d  4\n' % len(fr)

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