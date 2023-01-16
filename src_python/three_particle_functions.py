from pathlib import Path

import os, re
import numpy as np
import random
import rrgm_functions, parameters_and_constants

NEWLINE_SIZE_IN_BYTES = -1

elem_spin_prods_3 = {
    'no6':
    '  3  2  1  2            3n: s12=0 S=1/2 z\n  1  1  1\n  3  4  3\n  4  3  3\n  1  2\n -1  2\n',
    'no1':
    '  3  3  1  2            3n: s12=1 S=1/2 z\n  1  1  1\n  3  3  4\n  4  3  3\n  3  4  3\n  2  3\n -1  6\n -1  6\n',
    'no2':
    '  3  1  1  2            3n: s12=1 S=3/2 z\n  1  1  1\n  3  3  3\n  1  1\n',
    'he_no0':
    '  3  1  1  2            No1: t=0,T=1/2;s=1,S=1/2, l=even\n  1  1  1\n  3  4  2\n  1  1\n',
    'he_no1':
    '  3  6  1  2            No1: t=0,T=1/2;s=1,S=1/2, l=even\n  1  1  1\n  1  3  2\n  1  4  1\n  2  3  1\n  3  1  2\n  3  2  1\n  4  1  1\n  1  3\n -1 12\n -1 12\n -1  3\n +1 12\n +1 12\n',
    'he_no1y':
    '  3  6  1  1            No1: t=0,T=1/2;s=1,S=1/2, l=even\n  1  1  1\n  1  3  2\n  1  4  1\n  2  3  1\n  3  1  2\n  3  2  1\n  4  1  1\n  1  3\n -1 12\n -1 12\n -1  3\n +1 12\n +1 12\n',
    'he_no2':
    '  3  2  1  2            No2: t=0, S=3/2, l=even\n  1  1  1\n  1  3  1\n  3  1  1\n  1  2\n -1  2\n',
    'he_no2y':
    '  3  2  1  1            No2: t=0, S=3/2, l=even\n  1  1  1\n  1  3  1\n  3  1  1\n  1  2\n -1  2\n',
    'he_no3':
    '  3  4  1  2            No3: t=0,T=1/2;s=0,S=1/2, l=odd\n  1  1  1\n  1  4  1\n  2  3  1\n  3  2  1\n  4  1  1\n  1  4\n -1  4\n -1  4\n  1  4\n',
    'he_no3y':
    '  3  4  1  1            No3: t=0,T=1/2;s=0,S=1/2, l=odd\n  1  1  1\n  1  4  1\n  2  3  1\n  3  2  1\n  4  1  1\n  1  4\n -1  4\n -1  4\n  1  4\n',
    'he_no3ii':
    '  3  9  1  2            No1: t=1,T=3/2;s=1,S=1/2, l=odd\n  1  1  1\n  1  1  4\n  1  2  3\n  2  1  3\n  1  3  2\n  1  4  1\n  2  3  1\n  3  1  2\n  3  2  1\n  4  1  1\n  2  9\n -1 18\n -1 18\n  2  9\n -1 18\n -1 18\n  2  9\n -1 18\n -1 18\n',
    'he_no5':
    '  3  3  1  2            No5: t=1,T=1/2 S=3/2, l=odd\n  1  1  1\n  3  1  1\n  1  3  1\n  1  1  3\n -1  6\n -1  6\n  2  3\n',
    'he_no5y':
    '  3  3  1  2            No5: t=1,T=1/2 S=3/2, l=odd\n  1  1  1\n  3  1  1\n  1  3  1\n  1  1  3\n -1  6\n -1  6\n  2  3\n',
    'he_no5ii':
    '  3  3  1  2            No5: t=1,T=3/2 S=3/2, l=odd\n  1  1  1\n  3  1  1\n  1  3  1\n  1  1  3\n  1  3\n  1  3\n  1  3\n',
    'he_no6':
    '  3  6  1  2            No6: t=1,T=1/2;s=0,S=1/2, l=even\n  1  1  1\n  1  2  3\n  2  1  3\n  1  4  1\n  2  3  1\n  3  2  1\n  4  1  1\n  1  3\n -1  3\n -1 12\n  1 12\n -1 12\n  1 12\n',
    'he_no6y':
    '  3  6  1  1            No6: t=1,T=1/2;s=0,S=1/2, l=even\n  1  1  1\n  1  2  3\n  2  1  3\n  1  4  1\n  2  3  1\n  3  2  1\n  4  1  1\n  1  3\n -1  3\n -1 12\n  1 12\n -1 12\n  1 12\n',
    'he_no6ii':
    '  3  6  1  2            No6: t=1,T=3/2;s=0,S=1/2, l=even\n  1  1  1\n  1  2  3\n  2  1  3\n  1  4  1\n  2  3  1\n  3  2  1\n  4  1  1\n  1  6\n -1  6\n  1  6\n -1  6\n  1  6\n -1  6\n',
    't_no1':
    '  3  6  1  2            No1: t=0, S=1/2, l=even\n  1  1  1\n  1  3  4\n  1  4  3\n  2  3  3\n  3  1  4\n  3  2  3\n  4  1  3\n  1  3\n -1 12\n -1 12\n -1  3\n  1 12\n  1 12\n',
    't_no6':
    '  3  6  1  2            No6: t=1, S=1/2, l=even\n  1  1  1\n  1  4  3\n  2  3  3\n  3  2  3\n  4  1  3\n  3  4  1\n  4  3  1\n  1 12\n -1 12\n  1 12\n -1 12\n -1  3\n  1  3\n',
    'dist_3':
    '  3  1  1  2                    polarized 3-par\n  1  1  1\n  1  2  3\n  1  1\n',
}


def insam(anz, fn='INSAM'):
    out = '  1  1\n  1%3d\n' % anz

    with open(fn, 'w') as outfile:
        outfile.write(out)


def inlu_3(anzO, fn='INLU', fr=[], indep=0):
    out = '  0  0  0  0  0%3d\n' % indep
    for n in range(anzO):
        out += '  1'
    out += '\n%d\n' % len(fr)
    for n in range(0, len(fr)):
        out += '  1  3\n'

    for n in fr:
        out += '%3d%3d\n%3d\n' % (int(n[0]), int(n[1]), int(n[2]))

    for n in range(len(fr)):
        for m in range(len(fr)):
            out += '%s_%s\n' % (fr[n], fr[m])

    with open(fn, 'w') as outfile:
        outfile.write(out)


def inob_3(fr, anzO, fn='INOB', indep=0):
    #                IBOUND => ISOSPIN coupling allowed
    out = '  0  2  2  1%3d\n' % indep
    for n in range(anzO):
        out += '  1'
    out += '\n  4\n%3d  3\n' % len(fr)

    for n in fr:
        out += elem_spin_prods_3[n]

    for n in range(len(fr)):
        for m in range(len(fr)):
            out += '%s_%s\n' % (fr[n], fr[m])

    with open(fn, 'w') as outfile:
        outfile.write(out)


def inen_bdg_3(bas, jay, co, fn='INEN', pari=0, nzop=31, tni=11, idum=2):
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

        for n in range(1, int(max(1, 1 + max(bv[1])))):
            if n in bv[1]:
                tmp += '%3d' % int(1)
            else:
                tmp += '%3d' % int(0)

        tmp += '\n'
        out += tmp

    with open(fn, 'w') as outfile:
        outfile.write(head + out)


def parallel_mod_of_3inqua(frlu=[],
                           frob=[],
                           infile='INQUA',
                           outfile='INQUA_PAR',
                           tni=0,
                           einzel_path=''):

    pref = '-tni' if tni else ''
    s = ''
    #   NBAND1,NBAND2,NBAND3,NBAND4,NBAND5,naufset,NAUS,MOBAUS,LUPAUS,NBAUS,NLES
    s += ' 10  8  9  3 00  0  0  0  0\n'
    inq = [line for line in open(infile)][1:]
    for ln in range(len(inq)):
        if (re.search('eob', inq[ln]) != None):
            break
        else:
            s += inq[ln]

    # lower triangular BV matrix
    for bra in range(len(frlu)):
        for ket in range(bra + 1):
            s += einzel_path + 'eob%s/%s_%s\n' % (pref, frob[bra], frob[ket])
            s += einzel_path + 'elu%s/%s_%s\n' % (pref, frlu[bra], frlu[ket])

    appe = 'w'
    with open(outfile, appe) as outfi:
        outfi.write(s)
    return
    # r7 c2:   S  L           S_c
    #  1   :   0  0  1S0         0
    #  2   :   1  0  3S1         2


def inqua_3(intwi=[], relwi=[], potf='', inquaout='INQUA_M'):
    s = ''
    # NBAND1,NBAND2,NBAND3,NBAND4,NBAND5,NAUS,MOBAUS,LUPAUS,NBAUS
    s += ' 10  8  9  3 00  0  0  0  0\n%s\n' % potf
    zerl_counter = 0
    bv_counter = 1
    for n in range(len(intwi)):
        zerl_counter += 1
        nrel = len(relwi[n])
        s += '%3d%60s%s\n%3d%3d\n' % (len(intwi[n]), '', 'Z%d  BVs %d - %d' %
                                      (zerl_counter, bv_counter, bv_counter -
                                       1 + len(intwi[n])), len(intwi[n]), nrel)

        bv_counter += len(intwi[n])
        for bv in range(len(intwi[n])):
            s += '%36s%-12.6f\n' % ('', float(intwi[n][bv]))

        for rw in range(0, len(relwi[n])):
            s += '%12.6f' % float(relwi[n][rw])
            if ((rw != (len(relwi[n]) - 1)) & ((rw + 1) % 6 == 0)):
                s += '\n'
        s += '\n'

        tmpln = np.ceil(len(intwi[n]) / 6.)
        for bb in range(0, len(intwi[n])):
            s += '  1  1\n'
            for i in range(int(bb / 6)):
                s += '\n'
            s += '1.'.rjust(12 * (bb % 6 + 1))

            for ii in range(int(tmpln - int(bb / 6))):
                s += '\n'

    with open(inquaout, 'w') as outfile:
        outfile.write(s)

    return


def read_inob(infile='INOB'):
    fi = [line for line in open(infile)]
    ob_stru = []
    for ll in fi:
        if 'he' in ll.split()[-1]:
            ob_stru.append(ll.split()[-1].strip())
    return ob_stru


def read_inlu(infile='INLU'):
    fi = [line for line in open(infile)][2:]
    lu_stru = []
    for n in range(int(fi[0]) + 1, (int(fi[0]) + 1) + 2 * int(fi[0]), 2):
        lu_stru.append(fi[n].split()[0].strip() + fi[n].split()[1].strip() +
                       fi[n + 1].split()[0].strip())
    return lu_stru


def retrieve_he3_M(inqua):

    relw = []
    intw = []
    frgm = []
    inq = [line for line in open(inqua)]

    lineNR = 0
    while lineNR < len(inq):
        if ((re.search('Z', inq[lineNR]) != None) |
            (re.search('z', inq[lineNR]) != None)):
            break
        lineNR += 1
    if lineNR == len(inq):
        print('no <Z> qualifier found in <INQUA>!')
        exit()

    while ((lineNR < len(inq)) & (inq[lineNR][0] != '/')):
        try:
            anziw = int(inq[lineNR].split()[0])
        except:
            break

        anzbvLN = int(1 + np.ceil(anziw / 6)) * anziw
        anzrw = int(inq[lineNR + 1].split()[1])

        frgm.append([anziw, anzrw])
        intwtmp = []
        relwtmp = []
        for iws in range(0, 2 * anziw, 2):
            intwtmp += [float(inq[lineNR + 2 + iws].strip())]

            relwtmp.append(
                [float(rrw) for rrw in inq[lineNR + 3 + iws].split()])
        intw += [intwtmp]
        relw += [relwtmp]

        lineNR += 2 * anziw + anzbvLN + 2

    iw = intw
    rw = relw

    with open('intw3he.dat', 'w') as f:
        for ws in iw:
            np.savetxt(f, [ws], fmt='%12.4f', delimiter=' ; ')
    f.close()
    with open('relw3he.dat', 'w') as f:
        for wss in rw:
            for ws in wss:
                np.savetxt(f, [ws], fmt='%12.4f', delimiter=' ; ')
    f.close()

    return iw, rw, frgm