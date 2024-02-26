from pathlib import Path

import os, re
import shutil
import numpy as np
import random
import rrgm_functions, parameters_and_constants
import more_itertools

elem_spin_prods_4 = {
    # (2-2)
    #'np1s_np1s_S0':
    #'  4 24  1  2        dqdq S=0        z5   \n  1  1  1  1\n  1  4  1  4\n  2  3  2  3\n  3  2  1  4\n  4  1  2  3\n  1  4  3  2\n  2  3  4  1\n  3  2  3  2\n  4  1  4  1\n  4  3  1  2\n  4  3  2  1\n  1  2  3  4\n  2  1  4  3\n  1  4  2  3\n  2  3  1  4\n  3  2  2  3\n  4  1  1  4\n  1  4  4  1\n  2  3  3  2\n  3  2  4  1\n  4  1  3  2\n  3  4  2  1\n  4  3  1  2\n  1  2  4  3\n  2  1  3  4\n  1  4\n  1  4\n  1  4\n  1  4\n  1  4\n  1  4\n  1  4\n  1  4\n  1  4\n  1  4\n  1  4\n  1  4\n -1  4\n -1  4\n -1  4\n -1  4\n -1  1\n -1  1\n -1  1\n -1  1\n -1  1\n -1  1\n -1  1\n -1  1\n',
    'np1s_np1s_S0':
    '  4 24  1  2        dd S=0          z5   \n  1  1  1  1\n  1  4  1  4\n  2  3  1  4\n  1  4  2  3\n  2  3  2  3\n  3  2  1  4\n  4  1  1  4\n  3  2  2  3\n  4  1  2  3\n  1  4  3  2\n  2  3  3  2\n  1  4  4  1\n  2  3  4  1\n  3  2  3  2\n  4  1  3  2\n  3  2  4  1\n  4  1  4  1\n  1  3  2  4\n  2  4  1  3\n  3  1  2  4\n  4  2  1  3\n  1  3  4  2\n  2  4  3  1\n  3  1  4  2\n  4  2  3  1\n -1  4\n -1  4\n -1  4\n -1  4\n  1  4\n  1  4\n  1  4\n  1  4\n  1  4\n  1  4\n  1  4\n  1  4\n -1  4\n -1  4\n -1  4\n -1  4\n  1  1\n  1  1\n  1  1\n  1  1\n  1  1\n  1  1\n  1  1\n  1  1\n',
    #'np3s_np3s_S0':
    #'  4 36  1  2        dd S=0          z5   \n  1  1  1  1\n  1  4  1  4\n  2  3  1  4\n  1  4  2  3\n  2  3  2  3\n  3  2  1  4\n  4  1  1  4\n  3  2  2  3\n  4  1  2  3\n  1  4  3  2\n  2  3  3  2\n  1  4  4  1\n  2  3  4  1\n  3  2  3  2\n  4  1  3  2\n  3  2  4  1\n  4  1  4  1\n  1  3  2  4\n  2  4  1  3\n  3  1  2  4\n  4  2  1  3\n  1  3  4  2\n  2  4  3  1\n  3  1  4  2\n  4  2  3  1\n  1  2  3  4\n  3  4  1  2\n  4  3  1  2\n  2  1  3  4\n  3  4  2  1\n  1  2  4  3\n  4  3  2  1\n  2  1  4  3\n  3  3  2  2\n  4  4  1  1\n  1  1  4  4\n  2  2  3  3\n -1  4\n -1  4\n -1  4\n -1  4\n  1  4\n  1  4\n  1  4\n  1  4\n  1  4\n  1  4\n  1  4\n  1  4\n -1  4\n -1  4\n -1  4\n -1  4\n  1  1\n  1  1\n  1  1\n  1  1\n  1  1\n  1  1\n  1  1\n  1  1\n -1  4\n -1  4\n -1  4\n -1  4\n -1  4\n -1  4\n -1  4\n -1  4\n  1  1\n  1  1\n  1  1\n  1  1\n',
    #'np3s_np3s_S0':
    #'  4  6  1  2        dd S=0          z5   \n  1  1  1  1\n  1  3  2  4\n  2  4  1  3\n  1  4  1  4\n  2  3  1  4\n  1  4  2  3\n  2  3  2  3\n  1  1\n  1  1\n -1  4\n -1  4\n -1  4\n -1  4\n',
    #'np1s_np1s_S0':
    #'  4  4  1  2        dq-dq S=0       z    \n  1  1  1  1\n  1  4  1  4\n  2  3  2  3\n  1  4  2  3\n  2  3  1  4\n  1  4\n  1  4\n -1  4\n -1  4\n',
    'np1s_np1s_S0':
    '  4 16  1  2        dq-dq S=0       z    \n  1  1  1  1\n  1  4  1  4\n  1  4  2  3\n  1  4  3  2\n  1  4  4  1\n  2  3  1  4\n  2  3  2  3\n  2  3  3  2\n  2  3  4  1\n  3  2  1  4\n  3  2  2  3\n  3  2  3  2\n  3  2  4  1\n  4  1  1  4\n  4  1  2  3\n  4  1  3  2\n  4  1  4  1\n  1  1\n -1  1\n  1  1\n -1  1\n -1  1\n  1  1\n -1  1\n  1  1\n  1  1\n -1  1\n  1  1\n -1  1\n -1  1\n  1  1\n -1  1\n  1  1\n',
    'pp1s_nn1s_S0':
    '  4  8  1  2        pp-nn S=0       z    \n  1  1  1  1\n  4  3  1  2\n  4  3  2  1\n  1  2  3  4\n  2  1  4  3\n  3  4  2  1\n  4  3  1  2\n  1  2  4  3\n  2  1  3  4\n  1  1\n  1  1\n  1  1\n  1  1\n -1  1\n -1  1\n -1  1\n -1  1\n',
    'nn1s_nn1s_S0':
    '  4  4  1  2     nn0-nn0 S=0  z\n  1  1  1  1\n  3  4  1  2\n  4  3  2  1\n  4  3  1  2\n  3  4  2  1\n  1  4\n  1  4\n  1  4\n  1  4\n',
    # (3-1)
    'tp_1s0':
    '  4 12  1  3   tp No1  S=0,T=0    z1\n  1  1  1  1\n  4  2  3  1\n  4  1  4  1\n  4  1  3  2\n  2  4  3  1\n  2  3  4  1\n  2  3  3  2\n  3  2  4  1\n  3  2  3  2\n  3  1  4  2\n  1  4  4  1\n  1  4  3  2\n  1  3  4  2\n  1 12\n -1 48\n -1 48\n -1 12\n  1 48\n  1 48\n -1 48\n -1 48\n  1 12\n  1 48\n  1 48\n -1 12\n',
    'tp_6s0':
    '  4 12  1  3   tp No6  S=0,T=0    z3\n  1  1  1  1\n  4  3  2  1\n  4  3  1  2\n  4  1  4  1\n  4  1  3  2\n  2  3  4  1\n  2  3  3  2\n  3  4  2  1\n  3  4  1  2\n  3  2  4  1\n  3  2  3  2\n  1  4  4  1\n  1  4  3  2\n  1 12\n -1 12\n -1 48\n  1 48\n -1 48\n  1 48\n -1 12\n  1 12\n  1 48\n -1 48\n  1 48\n -1 48\n',
    'hen_1s0':
    '  4 12  1  3   he3n No1 S=0, T=0  z2\n  1  1  1  1\n  4  2  1  3\n  4  1  2  3\n  4  1  1  4\n  2  4  1  3\n  2  3  2  3\n  2  3  1  4\n  3  2  2  3\n  3  2  1  4\n  3  1  2  4\n  1  4  2  3\n  1  4  1  4\n  1  3  2  4\n -1 12\n  1 48\n  1 48\n  1 12\n -1 48\n -1 48\n  1 48\n  1 48\n -1 12\n -1 48\n -1 48\n  1 12\n',
    'hen_6s0':
    '  4 12  1  3   he3n No6 S=0, T=0  z1\n  1  1  1  1\n  4  1  2  3\n  4  1  1  4\n  2  3  2  3\n  2  3  1  4\n  2  1  4  3\n  2  1  3  4\n  3  2  2  3\n  3  2  1  4\n  1  4  2  3\n  1  4  1  4\n  1  2  4  3\n  1  2  3  4\n -1 48\n  1 48\n -1 48\n  1 48\n  1 12\n -1 12\n  1 48\n -1 48\n  1 48\n -1 48\n -1 12\n  1 12\n',
    'tp_123-4':
    '  4 24  1  3                        \n  1  1  1  1\n  1  2  3  4\n  1  2  4  3\n  1  3  2  4\n  1  3  4  2\n  1  4  2  3\n  1  4  3  2\n  2  1  3  4\n  2  1  4  3\n  2  3  1  4\n  2  3  4  1\n  2  4  1  3\n  2  4  3  1\n  3  1  2  4\n  3  1  4  2\n  3  2  1  4\n  3  2  4  1\n  3  4  1  2\n  3  4  2  1\n  4  1  2  3\n  4  1  3  2\n  4  2  1  3\n  4  2  3  1\n  4  3  1  2\n  4  3  2  1\n -1  1\n  1  1\n  1  1\n -1  1\n -1  1\n  1  1\n  1  1\n -1  1\n -1  1\n  1  1\n  1  1\n -1  1\n -1  1\n  1  1\n  1  1\n -1  1\n -1  1\n  1  1\n  1  1\n -1  1\n -1  1\n  1  1\n  1  1\n -1  1\n',
    'np3s_np3s_12-34':
    '  4 24  1  2                        \n  1  1  1  1\n  1  2  3  4\n  1  2  4  3\n  1  3  2  4\n  1  3  4  2\n  1  4  2  3\n  1  4  3  2\n  2  1  3  4\n  2  1  4  3\n  2  3  1  4\n  2  3  4  1\n  2  4  1  3\n  2  4  3  1\n  3  1  2  4\n  3  1  4  2\n  3  2  1  4\n  3  2  4  1\n  3  4  1  2\n  3  4  2  1\n  4  1  2  3\n  4  1  3  2\n  4  2  1  3\n  4  2  3  1\n  4  3  1  2\n  4  3  2  1\n -1  1\n  1  1\n  1  1\n -1  1\n -1  1\n  1  1\n  1  1\n -1  1\n -1  1\n  1  1\n  1  1\n -1  1\n -1  1\n  1  1\n  1  1\n -1  1\n -1  1\n  1  1\n  1  1\n -1  1\n -1  1\n  1  1\n  1  1\n -1  1\n',
}


def tn_inqua_31(basis4,
                stru3,
                relw=parameters_and_constants.w120,
                fn_inq='INQUA_N',
                fn_inen='INEN',
                fn='pot_dummy',
                typ='0p-n'):

    if os.path.isfile(os.getcwd() + '/INQUA_N') == True:
        appendd = True
    else:
        appendd = False

    bvinstruct_full_3 = rrgm_functions.determine_struct(fn_inq)
    print(bvinstruct_full_3)
    stru3_label = stru3[0][:-6].split('-')
    width_blocks = {}

    for s in stru3_label:
        width_blocks[s] = []

    print('3n-fragment structure:\n', stru3_label)
    exit()

    if len(bvinstruct_full_3) != len(stru3_label):
        print('3-body structure string inconsistent with INQUA_N fragments!')
        exit()

    bvinstruct_3 = np.zeros(len(bvinstruct_full_3)).astype(int)

    outs = ''
    bvs = []
    head = ' 10  8  9  3 00  0  0  0\n%s\n' % fn

    inen = fn_inen
    inqua = fn_inq

    lines_inen = [line for line in open(inen)]

    bnr_bv = int(lines_inen[3][4:8])
    anzr = int([line for line in open(inqua)][3][3:6])

    # read list of fragment basis vectors bvs = {[BV,rel]}
    bvs = []
    for anz in range(bnr_bv):
        nr = int(lines_inen[4 + 2 * anz].split()[1])
        for bv in range(0, anzr):
            try:
                if int(lines_inen[5 + 2 * anz].split()[bv]) == 1:
                    bvs.append([nr, bv])
                else:
                    pass
            except:
                pass

    lines_inqua = [line for line in open(inqua)]
    lines_inqua = lines_inqua[2:]
    bbv = []

    #print(bvs, len(bvs))

    # read width set for all v in bvs bbv = {[w1,w2]}
    # 2 widths specify the 3-body vector
    for bv in bvs:
        lie = 0
        maxbv = 0
        zerl_not_found = True
        while zerl_not_found == True:
            bvinz = int(lines_inqua[lie][:4])
            maxbv = maxbv + bvinz
            rel = int(lines_inqua[lie + 1][4:7])
            nl = int(rel / 6)
            if rel % 6 != 0:
                nl += 1
            if maxbv >= bv[0]:
                if maxbv >= bv[0]:
                    rell = []
                    [[
                        rell.append(float(a))
                        for a in lines_inqua[lie + 1 + bvinz + 1 +
                                             n].rstrip().split()
                    ] for n in range(0, nl)]
                    bbv.append([
                        float(lines_inqua[lie + 1 + bvinz - maxbv +
                                          bv[0]].strip().split()[0]),
                        rell[bv[1]]
                    ])

                    # how do the basis elements sort into the coupling-scheme structure?
                    nnn = 0
                    while bvinstruct_full_3[nnn] < bv[0]:
                        nnn += 1

                    bvinstruct_3[nnn] += 1

                    # assign the width set to an entry in the label-width dictionary
                    width_blocks[stru3_label[nnn]].append([
                        float(lines_inqua[lie + 1 + bvinz - maxbv +
                                          bv[0]].strip().split()[0]),
                        rell[bv[1]]
                    ])
                    zerl_not_found = False
            else:
                if bvinz < 7:
                    lie = lie + 2 + bvinz + nl + 2 * bvinz
                else:
                    lie = lie + 2 + bvinz + nl + 3 * bvinz

    #print(bbv)
    #print(width_blocks)
    # CAREFUL: rjust might place widths errorously in file!

    zmax = 8

    tm = []
    zerlegungs_struct_3 = [[zmax for i in range(int(bvinstruct_3[j] / zmax))]
                           for j in range(len(bvinstruct_3))]

    for j in range(len(bvinstruct_3)):
        if bvinstruct_3[j] % zmax != 0:
            zerlegungs_struct_3[j] += [bvinstruct_3[j] % zmax]

    if dbg: print(bvinstruct_3, ' --> ', zerlegungs_struct_3)

    zerlegungs_struct_4 = []
    zerl_counter = 0
    for s4 in range(len(basis4)):

        # retrieve width block and its structure for 4-body vector
        for zs in range(len(zerlegungs_struct_3)):

            label3 = basis4[s4][0][4:]

            if label3[0] == stru3_label[zs]:
                zerlegungs_struct_4.append([zerlegungs_struct_3[zs], label3])
                for n in range(len(zerlegungs_struct_3[zs])):
                    zerl_counter += 1
                    outs += '%3d%60s%s\n%3d%3d\n' % (
                        zerlegungs_struct_3[zs][n], '', 'Z%d' % zerl_counter,
                        zerlegungs_struct_3[zs][n], len(relw))

                    for bv in width_blocks[
                            label3[0]][sum(zerlegungs_struct_3[zs][:n]
                                           ):sum(zerlegungs_struct_3[zs][:n +
                                                                         1])]:

                        outs += '%48s%-12.6f%-12.6f\n' % ('', float(
                            bv[0]), float(bv[1]))

                    for rw in range(0, len(relw)):
                        outs += '%12.6f' % float(relw[rw])
                        if ((rw != (len(relw) - 1)) & ((rw + 1) % 6 == 0)):
                            outs += '\n'
                    outs += '\n'
                    for bb in range(0, zerlegungs_struct_3[zs][n]):
                        outs += '  1  1\n'
                        if zerlegungs_struct_3[zs][n] < 7:
                            outs += '1.'.rjust(12 * (bb + 1))
                            outs += '\n'
                        else:
                            if bb < 6:
                                outs += '1.'.rjust(12 * (bb + 1))
                                outs += '\n\n'
                            else:
                                outs += '\n'
                                outs += '1.'.rjust(12 * (bb % 6 + 1))
                                outs += '\n'

    if appendd:
        with open('INQUA_N', 'a') as outfile:
            outfile.write(outs)
    else:
        outs = head + outs
        with open('INQUA_N', 'w') as outfile:
            outfile.write(outs)

    return [fr for fr in zerlegungs_struct_4 if fr[0] != []]


def from3to4(stru3,
             relw=parameters_and_constants.w120,
             fn_outq='INQUA_TMP',
             zmax=8):

    outs = ''
    block_stru = []
    zerl_counter = 1

    for z3 in range(len(stru3[0])):

        stru = np.array(
            np.meshgrid(np.array(stru3[1][0][z3]),
                        np.array(stru3[1][1][z3]))).T.reshape(-1, 2)

        anzBV = len(stru)

        tmp = [zmax for i in range(int(anzBV / zmax))]
        if anzBV % zmax != 0:
            tmp += [anzBV % zmax]

        block_stru.append(tmp)

        bnds = np.insert(np.cumsum(tmp), 0, 0)
        stru = [stru[bnds[i]:bnds[i + 1]] for i in range(len(bnds) - 1)]

        label3 = stru3[0][z3][0]

        for nz in range(len(stru)):

            outs += '%3d%60s%s\n%3d%3d\n' % (len(stru[nz]), '', 'Z%d - %s' %
                                             (zerl_counter, label3),
                                             len(stru[nz]), len(relw))
            for bv in stru[nz]:
                outs += '%48s%-12.6f%-12.6f\n' % ('', float(bv[0]), float(
                    bv[1]))
            for rw in range(0, len(relw)):
                outs += '%12.6f' % float(relw[rw])
                if ((rw != (len(relw) - 1)) & ((rw + 1) % 6 == 0)):
                    outs += '\n'
            outs += '\n'
            for bb in range(0, len(stru[nz])):
                outs += '  1  1\n'
                if len(stru[nz]) < 7:
                    outs += '1.'.rjust(12 * (bb + 1))
                    outs += '\n'
                else:
                    if bb < 6:
                        outs += '1.'.rjust(12 * (bb + 1))
                        outs += '\n\n'
                    else:
                        outs += '\n'
                        outs += '1.'.rjust(12 * (bb % 6 + 1))
                        outs += '\n'

            zerl_counter += 1

    with open(fn_outq, 'w') as outfile:
        outfile.write(outs)

    return block_stru


def inlu_4(anzO, fn='INLU', fr=[], indep=0):
    out = '  0  0  0  0  0%3d\n' % indep
    for n in range(anzO):
        out += '  1'
    out += '\n%d\n' % len(fr)
    for n in range(0, len(fr)):
        out += '  1  4\n'

    zerle = {
        '000-0': '  0  0  0\n  0  1  2\n  0  4  3\n',
        #
        '001-1': '  0  0  1\n  0  1  2\n  1  4  3\n',
        '010-1': '  0  1  0\n  1  1  2\n  1  4  3\n',
        '100-1': '  1  0  0\n  1  1  2\n  1  4  3\n',
        #
        '011-0': '  0  1  1\n  1  1  2\n  0  4  3\n',
        '011-1': '  0  1  1\n  1  1  2\n  1  4  3\n',
        '011-2': '  0  1  1\n  1  1  2\n  2  4  3\n',
        '101-0': '  1  0  1\n  1  1  2\n  0  4  3\n',
        '101-1': '  1  0  1\n  1  1  2\n  1  4  3\n',
        '101-2': '  1  0  1\n  1  1  2\n  2  4  3\n',
        '110-0': '  1  1  0\n  0  1  2\n  0  4  3\n',
        '110-1': '  1  1  0\n  1  1  2\n  1  4  3\n',
        '110-2': '  1  1  0\n  2  1  2\n  2  4  3\n',
        #
        '111-1-0': '  1  1  1\n  1  1  2\n  0  4  3\n',
        '111-1-2': '  1  1  1\n  1  1  2\n  2  4  3\n',
        '111-2-2': '  1  1  1\n  2  1  2\n  2  4  3\n',
    }

    for n in fr:
        out += zerle[n]

    for n in range(len(fr)):
        for m in range(len(fr)):
            out += '%s_%s\n' % (fr[n], fr[m])

    with open(fn, 'w') as outfile:
        outfile.write(out)


def inob_4(fr, anzO, fn='INOB', indep=0):
    #                IBOUND => ISOSPIN coupling allowed
    out = '  0  2  2  1%3d\n' % indep
    for n in range(anzO):
        out += '  1'
    out += '\n  4\n%3d  4\n' % len(fr)

    for n in fr:
        out += elem_spin_prods_4[n]

    for n in range(len(fr)):
        for m in range(len(fr)):
            out += '%s_%s\n' % (fr[n], fr[m])

    with open(fn, 'w') as outfile:
        outfile.write(out)


def inqua_4(intwi=[], relwi=[], potf='', inquaout='INQUA_M'):
    s = ''
    # NBAND1,NBAND2,NBAND3,NBAND4,NBAND5,NAUS,MOBAUS,LUPAUS,NBAUS
    s += ' 10  8  9  3 00  0  0  0  0\n%s\n' % potf
    zerl_counter = 0
    bv_counter = 1
    for n in range(len(intwi)):

        zerl_counter += 1
        nrel = len(relwi[n])

        #print(intwi[n])
        # construct a 2-column list ordered wrt. the first one
        thh = intwi[n]
        th2 = np.reshape(thh, (2, -1))

        #print(th2)
        idxt = th2[0].argsort()[::-1]
        col1 = [eww for eww in th2[0][idxt]]
        col2 = [eww for eww in th2[1][idxt]]
        iiw = np.array([col1, col2])
        #print(iiw)
        #exit()
        nb = int(len(intwi[n]) / 2)
        s += '%3d%60s%s\n%3d%3d\n' % (
            nb, '', 'Z%d  BVs %d - %d' %
            (zerl_counter, bv_counter, bv_counter - 1 + nb), nb, nrel)

        bv_counter += nb
        for bv in range(int(nb)):
            s += '%48s%-12.6f%-12.6f\n' % ('', float(
                iiw[0][bv]), float(iiw[1][bv]))

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


def inen_bdg_4(bas,
               jay,
               co,
               fn='INEN',
               pari=0,
               nzop=31,
               tni=11,
               idum=2,
               nzz=4):
    # idum=2 -> I4 for all other idum's -> I3
    # NBAND1,IDUM,NBAND3,NZOP,IFAKD,IGAK,NZZ,IAUW,IDRU,IPLO,IDUN,ICOPMA(=1 -> stop after N,H output)
    head = '%3d%3d 12%3d  1  0%3d  0  0 -1  0 +0\n' % (tni, idum, nzop, nzz)

    if nzop == 31:
        head += '  1  1  1  1  0  0  0  0  0  0  0  0  0  0  1  1\n'
    elif nzop == 28:
        head += '  1  1  1  1  1  1  0  0  0  0  0  0  1\n'

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


def inen_str_4(coeff,
               wr,
               bvs,
               uec,
               phys_chan,
               dma=[1, 0, 1, 0, 1, 0, 0],
               jay=0,
               anzch=1,
               pari=0,
               nzop=9,
               tni=10,
               fn='INEN_STR',
               diCh=''):

    s = '%3d  2 12%3d  1  1 +2  0  0 -1\n' % (tni, nzop)

    if nzop == 31:
        s += '  1  1  1  1  0  0  0  0  0  0  0  0  0  0  1  1\n'
    elif nzop == 28:
        s += '  1  1  1  1  1  1  0  0  0  0  0  0  1\n'

    s += coeff + '\n'

    sumuec = list(more_itertools.collapse(uec))

    cumc = [0]
    for cset in phys_chan[2]:
        tmp = len(cset) if np.ndim(cset) == 1 else len(cset[0])
        tmp += cumc[-1]
        cumc += [tmp]

    #cumc = np.insert(np.cumsum([len(cset) for cset in uec]), 0, 0)

    # SPIN #CHANNELS
    s += '%4d%4d   0   0%4d   1\n' % (int(2 * jay), anzch, pari)

    # FRAGMENT-EXPANSION COEFFICIENTS
    s += '%4d\n' % len(list(more_itertools.collapse(phys_chan[2])))

    for cf in list(more_itertools.collapse(phys_chan[2])):
        s += '%-20.9f\n' % cf

    # ------------------------------------ phys chan
    chanstrs = []
    ncof = 0

    for nphy_chan in range(len(phys_chan[0])):

        stmp = ''
        stmp += '%3d%3d%3d\n' % (phys_chan[0][nphy_chan][0],
                                 phys_chan[0][nphy_chan][1],
                                 phys_chan[0][nphy_chan][2])
        stmp += '%4d' % len(phys_chan[2][nphy_chan])
        di = 1
        for i in range(phys_chan[1][nphy_chan][0], phys_chan[1][nphy_chan][1]):
            di += 1
            stmp += '%4d' % (i + 1)
            if ((di % 20 == 0) | (int(i + 1) == phys_chan[1][nphy_chan][1])):
                stmp += '\n'
                di = 0
        di = 1
        ncof = cumc[nphy_chan]
        for i in range(1, 1 + len(phys_chan[2][nphy_chan])):
            stmp += '%4d' % (i + ncof)
            if ((di % 20 == 0) | (di == len(phys_chan[2][nphy_chan]))):
                stmp += '\n'
            di += 1
        nbr_relw_phys_chan = len(wr)
        for i in range(1, nbr_relw_phys_chan + 1):
            stmp += '%3d' % int(1)
            if ((int(i) % 50 == 0) | (int(i) == nbr_relw_phys_chan)):
                stmp += '\n'
        # channels are put in reverse order!
        # chanstrs.insert(0, stmp)
        chanstrs.append(stmp)

    s += ''.join(chanstrs)

    sumuec = list(more_itertools.collapse(phys_chan[2]))

    distuec = [
        nc + 1 for nc in range(len(sumuec)) if 10**2 > np.abs(sumuec[nc]) > 0.1
    ]

    #print(sumuec)
    #print(distuec)

    fd = True
    unique_chs = [[phys_chan[0][0]], [phys_chan[1][0]], [phys_chan[2][0]]]

    for nn in range(len(phys_chan[0])):
        newch = True
        for mm in range(len(unique_chs[0])):
            if phys_chan[0][nn] == unique_chs[0][mm]:
                newch = False
                break
        if newch:
            unique_chs[0].append(phys_chan[0][nn])
            unique_chs[1].append(phys_chan[1][nn])
            unique_chs[2].append(phys_chan[2][nn])

    for nphy_chan in range(len(unique_chs[0])):
        relwoffset = ''
        for i in range(unique_chs[1][nphy_chan][0],
                       unique_chs[1][nphy_chan][1] - 4):
            s += '%3d%3d%3d' % (unique_chs[0][nphy_chan][0],
                                unique_chs[0][nphy_chan][1],
                                unique_chs[0][nphy_chan][2])
            if fd:
                s += ' -1\n'
                fd = False
            else:
                s += '\n'
            s += '   1%4d\n' % (i + 1)
            nc = np.random.choice(distuec)
            s += '%-4d\n' % nc
            assert np.abs(sumuec[nc - 1]) > 0.1
            s += relwoffset
            for relw in dma:
                s += '%3d' % relw
            s += '\n'

    # here we attach the additional distortion channels which do not contribute
    # to the expansion of the asymptotic ones
    s += diCh

    if os.path.exists(fn):
        shutil.copy(fn, 'inen.bkp')
    with open(fn, 'w') as outfile:
        outfile.write(s)

    return


def spole_4(nzen=20,
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


def from2to4(relw, zwei_inq, vier_dir, fn, app=False):

    # input 1: a two-body basis is read from "zwei_inq" as a set if width parameters
    # for one(!) particular LST configuration
    # input 2: a set of radial widths which expands the relative motion between the two dimers

    # output: input file for the 4-body channel corresponding to a dimer-dimer partition

    outs = ''
    if app == False:
        outs += ' 10  8  9  3 00  0  0  0\n%s\n' % fn
    dstru = []
    # -----------------------------------------------------
    inquas = zwei_inq

    lines_inquas = [line for line in open(inquas)]

    arw = int(lines_inquas[3].split()[1])
    lines_inquas = lines_inquas[5:5 + int(arw / 6) + int(arw % 6 > 0)]
    wws = []
    for ll in lines_inquas:
        for l in ll.strip().split():
            wws.append(float(l))
    wws = wws[::-1]

    bvPerZ = 8
    anzBV = int(arw * (arw + 1) / 2)
    zstruct = [bvPerZ for n in range(0, int(anzBV / bvPerZ))]
    if anzBV % bvPerZ != 0:
        zstruct.append(anzBV % bvPerZ)

    #print(zstruct, len(zstruct), 'x nn-nn')

    bvl = arw - 1
    bvr = arw - 1
    for z in range(0, len(zstruct)):
        outs += '%3d\n%3d%3d\n' % (zstruct[z], zstruct[z], len(relw))
        for bv in range(zstruct[z]):
            outs += '%48s%-12.6f%-12.6f\n' % ('', float(
                wws[bvl]), float(wws[bvr]))
            if bvr > 0:
                bvr = bvr - 1
            else:
                bvl = bvl - 1
                bvr = bvl
        for rw in range(0, len(relw)):
            outs += '%12.6f' % float(relw[rw])
            if (((rw + 1) % 6 == 0) & (rw + 1 != len(relw))):
                outs += '\n'
        outs += '\n'
        for bb in range(0, zstruct[z]):
            outs += '  1  1\n'
            if zstruct[z] < 7:
                outs += '1.'.rjust(12 * (bb + 1))
                outs += '\n'
            else:
                if bb < 6:
                    outs += '1.'.rjust(12 * (bb + 1))
                    outs += '\n\n'
                else:
                    outs += '\n'
                    outs += '1.'.rjust(12 * (bb % 6 + 1))
                    outs += '\n'

    #writemode = 'a' if app else 'w'
    #with open(vier_dir + '/INQUA_N', writemode) as outfile:
    #    outfile.write(outs)

    return zstruct, outs


def from22to4(relw, zwei_inq_1, zwei_inq_2, vier_dir, fn, app=False):

    # input 1: two(!) two-body bases are read from "zwei_inq_1/2" as sets of width parameters
    # for particular LST configuration
    # input 2: a set of radial widths which expands the relative motion between the two dimers

    # output: input file for the 4-body channel corresponding to a dimer-dimer partition

    outs = ''
    if app == False:
        outs += ' 10  8  9  3 00  0  0  0\n%s\n' % fn
    dstru = []

    # dimer 1 --------------------------------------------
    inquas = zwei_inq_1

    lines_inquas = [line for line in open(inquas)]

    arw1 = int(lines_inquas[3].split()[1])
    lines_inquas = lines_inquas[5:5 + int(arw1 / 6) + int(arw1 % 6 > 0)]
    wws1 = []
    for ll in lines_inquas:
        for l in ll.strip().split():
            wws1.append(float(l))
    wws1 = wws1[::-1]

    # dimer 2 --------------------------------------------
    inquas = zwei_inq_2

    lines_inquas = [line for line in open(inquas)]

    arw2 = int(lines_inquas[3].split()[1])
    lines_inquas = lines_inquas[5:5 + int(arw2 / 6) + int(arw2 % 6 > 0)]
    wws2 = []
    for ll in lines_inquas:
        for l in ll.strip().split():
            wws2.append(float(l))
    wws2 = wws2[::-1]

    bvPerZ = 8
    anzBV = int(arw1 * arw2)
    zstruct = [bvPerZ for n in range(0, int(anzBV / bvPerZ))]
    if anzBV % bvPerZ != 0:
        zstruct.append(anzBV % bvPerZ)

    #print(zstruct, len(zstruct), 'x dimer-dimer')
    bvl = arw1 - 1
    bvr = arw2 - 1
    for z in range(0, len(zstruct)):
        outs += '%3d\n%3d%3d\n' % (zstruct[z], zstruct[z], len(relw))
        for bv in range(zstruct[z]):
            outs += '%48s%-12.6f%-12.6f\n' % ('', float(
                wws1[bvl]), float(wws2[bvr]))
            if bvr > 0:
                bvr = bvr - 1
            else:
                bvl = bvl - 1
                bvr = arw2 - 1
        for rw in range(0, len(relw)):
            outs += '%12.6f' % float(relw[rw])
            if (((rw + 1) % 6 == 0) & (rw + 1 != len(relw))):
                outs += '\n'
        outs += '\n'
        for bb in range(0, zstruct[z]):
            outs += '  1  1\n'
            if zstruct[z] < 7:
                outs += '1.'.rjust(12 * (bb + 1))
                outs += '\n'
            else:
                if bb < 6:
                    outs += '1.'.rjust(12 * (bb + 1))
                    outs += '\n\n'
                else:
                    outs += '\n'
                    outs += '1.'.rjust(12 * (bb % 6 + 1))
                    outs += '\n'

    #writemode = 'a' if app else 'w'
    #with open(vier_dir + '/INQUA_N', writemode) as outfile:
    #    outfile.write(outs)

    return zstruct, outs