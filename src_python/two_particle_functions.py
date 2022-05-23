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


def h2_spole(nzen=20,
             e0=0.01,
             d0=0.075,
             eps=0.01,
             bet=1.1,
             nzrw=400,
             frr=0.06,
             rhg=8.0,
             rhf=1.0,
             pw=1):
    s = ''
    s += ' 11  3  0  0  0  1\n'
    s += '%3d  0  0\n' % int(nzen)
    s += '%12.4f%12.4f\n' % (float(e0), float(d0))
    s += '%12.4f%12.4f%12.4f\n' % (float(eps), float(eps), float(eps))
    s += '%12.4f%12.4f%12.4f\n' % (float(bet), float(bet), float(bet))
    #    OUT
    s += '  0  0  1  0  1  0 -0  0\n'
    s += '%3d\n' % int(nzrw)
    s += '%12.4f%12.4f%12.4f\n' % (float(frr), float(rhg), float(rhf))
    s += '  1  2  3  4\n'
    s += '0.0         0.0         0.0\n'
    s += '.001        .001        .001\n'
    if pw == 0:
        s += '.5          .5          .5          .5\n'
    elif pw == 1:
        s += '.3          .3          .3          .3\n'
    elif pw == 2:
        s += '.15         .15         .15         .15\n'
    s += '1.          1.          0.\n'
    with open('INPUTSPOLE', 'w') as outfile:
        outfile.write(s)
    return


def h2_inlu(anzo=5):
    s = ''
    s += '  9\n'
    for n in range(anzo):
        s += '  1'
    s += '\n  1\n'
    s += '  4  2\n'
    s += '  0\n  1\n  2\n  3'
    with open('INLUCN', 'w') as outfile:
        outfile.write(s)
    return


def h2_inob(anzo=5):
    s = ''
    s += '  0  0\n'
    for n in range(anzo):
        s += '  1'
    s += '\n  4\n'
    s += '  1  2\n'
    s += '  2  9  6  1\n'
    s += '  1  1\n'
    s += '  1  3\n'  #  p-up, n-up
    s += '  1  4\n'  #  ...
    s += '  2  3\n'
    s += '  1  2\n'
    s += '  2  1\n'
    s += '  3  4\n'
    s += '  4  3\n'
    s += '  3  3\n'  # n-up, n-up
    s += '  1  1\n'  # p-up, p-up
    s += '  1  1\n'
    s += '  0  1  1  2\n'
    s += '  0  1 -1  2\n'
    s += '  0  1  0  1  1  2\n'
    s += '  0  1  0  1 -1  2\n'
    s += '  0  1  0  1  0  1  1  2\n'
    s += '  0  1  0  1  0  1 -1  2\n'
    s += '  0  1  0  1  0  1  0  1  1  1\n'
    s += '  0  1  0  1  0  1  0  1  0  1  1  1'
    with open('INOB', 'w') as outfile:
        outfile.write(s)
    return


def h2_inqua(relw, ps2):
    s = ''
    s += ' 10  8  9  3 00  0  0  0  0\n'
    #s += pot_dir + ps2 + '\n'
    s += ps2 + '\n'
    s += ' 14\n'
    s += '  1%3d\n' % int(len(relw))
    s += '.0          .0\n'
    for relwl in range(0, int(np.ceil(float(len(relw)) / float(6)))):
        for rr in range(0, 6):
            if (relwl * 6 + rr) < len(relw):
                s += '%12.7f' % float(relw[relwl * 6 + rr])
        s += '\n'
    s += '  2  1\n1.\n'  # 1:  n-p 1S0
    s += '  1  1\n1.\n'  # 2:  n-p 3S1
    # ------------
    s += '  3  1\n1.\n'  # 3:  p-p 1S0
    # ------------
    s += '  4  1\n1.\n'  # 4:  n-n 1S0
    s += '  5  2\n1.\n'  # 5:  n-n 3P0,1,2
    s += '  4  3\n1.\n'  # 6:  n-n 1D2
    s += '  5  4\n1.\n'  # 7:  n-n 3F2,3,4
    # ------------
    s += '  2  2\n1.\n'  # 8:  n-p 1P1
    s += '  1  2\n1.\n'  # 9:  n-p 3P0,1,2
    s += '  1  3\n1.\n'  # 10: n-p 3D1
    # ------------
    s += '  3  1\n1.\n'  # 4:  p-p 1S0
    s += '  6  2\n1.\n'  # 5:  p-p 3P0,1,2
    s += '  3  3\n1.\n'  # 6:  p-p 1D2
    s += '  6  4\n1.'  # 7:  p-p 3F2,3,4
    with open('INQUA_N', 'w') as outfile:
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


def h2_inen_bs(relw, costr, j=0, ch=1, anzo=9, nzz=0, EVein=[]):
    s = ''
    s += ' 10  2 12%3d  1  1%3d  0  0 -1\n' % (int(anzo), nzz)
    #       N  T Co CD^2 LS  T
    s += '  1  1  1  1  1  1  1  1  1  1\n'

    s += '%s\n' % costr

    #     2*J #ch s/b
    s += '%4d%4d   1   0   2\n' % (int(2 * j), len(ch))
    for c in ch:
        s += '   1%4d\n' % int(c)
        for rr in range(len(relw)):
            s += '  1'
        s += '\n'

    if nzz < 0:
        for c in EVein:
            s += c

    with open('INEN', 'w') as outfile:
        outfile.write(s)
    return


def h2_inen_str(relw, costr, j=0, sc=0, ch=1, anzo=7):
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
    for rr in range(1, len(relw) + 1):
        if ((rr % 30 == 0)):
            s += '  1'
            s += '\n'
        s += '  1'
    with open('INEN', 'w') as outfile:
        outfile.write(s)
    return