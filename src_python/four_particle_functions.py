from pathlib import Path

import os, re
import numpy as np
import random
import rrgm_functions, parameters_and_constants

elem_spin_prods_4 = {
    # (2-2)
    'ddS0':
    '  4 24  1  2        dd S=0          z5   \n  1  1  1  1\n  4  2  3  1\n  4  2  1  3\n  4  1  4  1\n  4  1  2  3\n  4  1  3  2\n  4  1  1  4\n  2  4  3  1\n  2  4  1  3\n  2  3  4  1\n  2  3  2  3\n  2  3  3  2\n  2  3  1  4\n  3  2  4  1\n  3  2  2  3\n  3  2  3  2\n  3  2  1  4\n  3  1  4  2\n  3  1  2  4\n  1  4  4  1\n  1  4  2  3\n  1  4  3  2\n  1  4  1  4\n  1  3  4  2\n  1  3  2  4\n  1 12\n -1 12\n -1 48\n  1 48\n -1 48\n  1 48\n -1 12\n  1 12\n  1 48\n -1 48\n  1 48\n -1 48\n -1 48\n  1 48\n -1 48\n  1 48\n  1 12\n -1 12\n  1 48\n -1 48\n  1 48\n -1 48\n -1 12\n  1 12\n',
    'dqdqS0':
    '  4 16  1  2        dq-dq S=0       z    \n  1  1  1  1\n  1  4  1  4  0  0\n  1  4  2  3  0  0\n  1  4  3  2  0  0\n  1  4  4  1  0  0\n  2  3  1  4  0  0\n  2  3  2  3  0  0\n  2  3  3  2  0  0\n  2  3  4  1  0  0\n  3  2  1  4  0  0\n  3  2  2  3  0  0\n  3  2  3  2  0  0\n  3  2  4  1  0  0\n  4  1  1  4  0  0\n  4  1  2  3  0  0\n  4  1  3  2  0  0\n  4  1  4  1  0  0\n -1 48\n  1 48\n -1 48\n  1 48\n  1 48\n -1 48\n  1 48\n -1 48\n -1 48\n  1 48\n -1 48\n  1 48\n  1 48\n -1 48\n  1 48\n -1 48\n',
    'dndnS0':
    '  4  4  1  2     nn0-nn0 S=0  z\n  1  1  1  1\n  3  4  3  4\n  4  3  4  3\n  4  3  3  4\n  3  4  4  3\n  1  4\n  1  4\n -1  4\n -1  4\n',
    'nnnnS0':
    '  4  6  1  2     nn1-nn1 S=0  z\n  1  1  1  1\n  3  3  4  4\n  4  4  3  3\n  4  3  3  4\n  3  4  4  3\n  3  4  3  4\n  4  3  4  3\n  1  3\n  1  3\n -1 12\n -1 12\n -1 12\n -1 12\n',
    'nnnnS1':
    '  4  4  1  2     nn1-nn1 S=1  z\n  1  1  1  1\n  3  3  3  4\n  3  3  4  3\n  4  3  3  3\n  3  4  3  3\n  1  4\n  1  4\n -1  4\n -1  4\n',
    'nnnnS2':
    '  4  1  1  2     nn1-nn1 S=2  z\n  1  1  1  1\n  3  3  3  3\n  1  1\n',
    'nnnnS0t':
    '  4  1  1  2     nn1-nn1 S=2  z\n  1  1  1  1\n  3  4  3  4\n  1  1\n',
    'nndnS1':
    '  4  2  1  2     nn1-nn0 S=1  z\n  1  1  1  1\n  3  3  3  4\n  3  3  4  3\n  1  2\n -1  2\n',
    # (3-1)
    'No6S0':
    '  4  4  1  3     (nn0n1/2)n S=0  z\n  1  1  1  1\n  4  3  4  3\n  4  3  3  4\n  3  4  4  3\n  3  4  3  4\n  1  4\n -1  4\n -1  4\n  1  4\n',
    'No6S1':
    '  4  2  1  3     (nn0n1/2)n S=1  z\n  1  1  1  1\n  4  3  3  3\n  3  4  3  3\n -1  2\n  1  2\n',
    'No1S0':
    '  4  6  1  3     (nn1n1/2)n S=0  z\n  1  1  1  1\n  4  4  3  3\n  4  3  4  3\n  4  3  3  4\n  3  4  4  3\n  3  4  3  4\n  3  3  4  4\n  1  3\n -1 12\n -1 12\n -1 12\n -1 12\n  1  3\n',
    'No1S1':
    '  4  3  1  3     (nn1n1/2)n S=1  z\n  1  1  1  1\n  4  3  3  3\n  3  4  3  3\n  3  3  4  3\n -1  6\n -1  6\n  2  3\n',
    'No2S1':
    '  4  4  1  3     (nn1n3/2)n S=1  z\n  1  1  1  1\n  4  3  3  3\n  3  4  3  3\n  3  3  4  3\n  3  3  3  4\n -1 12\n -1 12\n -1 12\n  1 12\n',
    'No2S2':
    '  4  1  1  3     (nn1n3/2)n S=2  z\n  1  1  1  1\n  3  3  3  3\n  1  1\n',
    #
    'n3n_6s0':
    '  4  4  1  3     (nn0n1/2)n S=0\n  1  1  1  1\n  4  3  4  3\n  4  3  3  4\n  3  4  4  3\n  3  4  3  4\n  1  4\n -1  4\n -1  4\n  1  4\n',
    'n3n_6s1':
    '  4  2  1  3     (nn0n1/2)n S=1\n  1  1  1  1\n  4  3  3  3\n  3  4  3  3\n -1  2\n  1  2\n',
    'n3n_1s0':
    '  4  6  1  3     (nn1n1/2)n S=0\n  1  1  1  1\n  4  4  3  3\n  4  3  4  3\n  4  3  3  4\n  3  4  4  3\n  3  4  3  4\n  3  3  4  4\n  1  3\n -1 12\n -1 12\n -1 12\n -1 12\n  1  3\n',
    'n3n_1s1':
    '  4  3  1  3     (nn1n1/2)n S=1\n  1  1  1  1\n  4  3  3  3\n  3  4  3  3\n  3  3  4  3\n -1  6\n -1  6\n  2  3\n',
    'n3n_2s1':
    '  4  4  1  3     (nn1n3/2)n S=1\n  1  1  1  1\n  3  3  3  4\n  4  3  3  3\n  3  4  3  3\n  3  3  4  3\n  3  4\n -1 12\n -1 12\n -1 12\n',
    'n3n_2s21':
    '  4  1  1  3     (nn1n3/2)n S=2\n  1  1  1  1\n  3  3  3  3\n  1  1\n',
    'n3n_2s22':
    '  4  1  1  3     (nn1n3/2)n S=2\n  1  1  1  1\n  3  3  3  3\n  1  1\n',
    'tp_1s0':
    '  4 12  1  3   tp No1  S=0,T=0    z1\n  1  1  1  1\n  4  2  3  1\n  4  1  4  1\n  4  1  3  2\n  2  4  3  1\n  2  3  4  1\n  2  3  3  2\n  3  2  4  1\n  3  2  3  2\n  3  1  4  2\n  1  4  4  1\n  1  4  3  2\n  1  3  4  2\n  1 12\n -1 48\n -1 48\n -1 12\n  1 48\n  1 48\n -1 48\n -1 48\n  1 12\n  1 48\n  1 48\n -1 12\n',
    'tp_6s0':
    '  4 12  1  3   tp No6  S=0,T=0    z3\n  1  1  1  1\n  4  3  2  1\n  4  3  1  2\n  4  1  4  1\n  4  1  3  2\n  2  3  4  1\n  2  3  3  2\n  3  4  2  1\n  3  4  1  2\n  3  2  4  1\n  3  2  3  2\n  1  4  4  1\n  1  4  3  2\n  1 12\n -1 12\n -1 48\n  1 48\n -1 48\n  1 48\n -1 12\n  1 12\n  1 48\n -1 48\n  1 48\n -1 48\n',
    'tp_1s1':
    '  4  4  1  3   tp No3  S=1          \n  1  1  1  1\n  4  1  3  1\n  2  3  3  1\n  3  2  3  1\n  1  4  3  1\n -1  8\n  1  8\n  1  8\n -1  8\n',
    'tp_5s2':
    '  4  3  1  3   tp No5  S=2          \n  1  1  1  1\n  3  3  1  1\n  3  1  3  1\n  1  3  3  1\n  1  3\n -1 12\n -1 12\n',
    'tp_5s1':
    '  4 12  1  3   tp No5  S=1          \n  1  1  1  1\n  4  3  1  1\n  4  1  3  1\n  2  3  3  1\n  3  4  1  1\n  3  2  3  1\n  3  3  2  1\n  3  3  1  2\n  3  1  4  1\n  3  1  3  2\n  1  4  3  1\n  1  3  4  1\n  1  3  3  2\n -1 36\n  1144\n  1144\n -1 36\n  1144\n -1 36\n  1  4\n  1144\n -1 16\n  1144\n  1144\n -1 16\n',
    'hen_1s0':
    '  4 12  1  3   he3n No1 S=0, T=0  z2\n  1  1  1  1\n  4  2  1  3\n  4  1  2  3\n  4  1  1  4\n  2  4  1  3\n  2  3  2  3\n  2  3  1  4\n  3  2  2  3\n  3  2  1  4\n  3  1  2  4\n  1  4  2  3\n  1  4  1  4\n  1  3  2  4\n -1 12\n  1 48\n  1 48\n  1 12\n -1 48\n -1 48\n  1 48\n  1 48\n -1 12\n -1 48\n -1 48\n  1 12\n',
    'hen_6s0':
    '  4 12  1  3   he3n No6 S=0, T=0  z1\n  1  1  1  1\n  4  1  2  3\n  4  1  1  4\n  2  3  2  3\n  2  3  1  4\n  2  1  4  3\n  2  1  3  4\n  3  2  2  3\n  3  2  1  4\n  1  4  2  3\n  1  4  1  4\n  1  2  4  3\n  1  2  3  4\n -1 48\n  1 48\n -1 48\n  1 48\n  1 12\n -1 12\n  1 48\n -1 48\n  1 48\n -1 48\n -1 12\n  1 12\n',
    'hen_1s1':
    '  4  4  1  3   he3 No3  S=1         \n  1  1  1  1\n  4  1  1  3\n  2  3  1  3\n  3  2  1  3\n  1  4  1  3\n  1  8\n -1  8\n -1  8\n  1  8\n',
    'hen_5s2':
    '  4  3  1  3   he3 No5  S=2         \n  1  1  1  1\n  3  1  1  3\n  1  3  1  3\n  1  1  3  3\n -1 12\n -1 12\n  1  3\n',
    'hen_5s1':
    '  4 12  1  3   he3 No5  S=1         \n  1  1  1  1\n  4  1  1  3\n  2  3  1  3\n  2  1  3  3\n  3  2  1  3\n  3  1  2  3\n  3  1  1  4\n  1  4  1  3\n  1  2  3  3\n  1  3  2  3\n  1  3  1  4\n  1  1  4  3\n  1  1  3  4\n  1144\n  1144\n -1 36\n  1144\n  1144\n -1 16\n  1144\n -1 36\n 1144\n -1 16\n -1 36\n  1  4\n',
    'tn_1s1':
    '  4  6  1  3   tn No1  S=1,T12,S12=1    z1\n  1  1  1  1\n  4  1  3  3\n  2  3  3  3\n  3  2  3  3\n  3  1  4  3\n  1  4  3  3\n  1  3  4  3\n  1 12\n -1 12\n  1 12\n -1  3\n -1 12\n  1  3\n',
    'tn_6s1':
    '  4  6  1  3   tn No1  S=1,T12,S12=0    z1\n  1  1  1  1\n  4  3  1  3\n  4  1  3  3\n  2  3  3  3\n  3  4  1  3\n  3  2  3  3\n  1  4  3  3\n  1  3\n -1 12\n -1 12\n -1  3\n  1 12\n  1 12\n',
}


def tn_inqua_31(basis4,
                stru3,
                dicti,
                relw=parameters_and_constants.w120,
                fn_inq='INQUA_N',
                fn_inen='INEN',
                fn='pot_dummy',
                typ='0p-n'):

    if os.path.isfile(os.getcwd() + '/INQUA_N') == True:
        appendd = True
    else:
        appendd = False

    width_blocks = {}

    for partition3 in stru3:
        part_str = ''
        for spin_stru in partition3:
            part_str += spin_stru + '-'
        part_str = part_str[:-1] + '__0_5P'

        inen = fn_inen + part_str
        inqua = fn_inq + part_str

        bvinstruct_part_3 = rrgm_functions.determine_struct(inqua)
        for s in partition3:
            width_blocks[s] = []

        if dbg:
            print('3n-fragment structure:\n', partition3)

        if len(bvinstruct_part_3) != len(partition3):
            print(
                '3-body structure string inconsistent with INQUA_N fragments!')
            exit()

        bvinstruct_tmp_3 = np.zeros(len(bvinstruct_part_3)).astype(int)

        outs = ''
        bvs = []
        head = ' 10  8  9  3 00  0  0  0\n%s\n' % fn

        lines_inen = [line for line in open(inen)]

        offss = 0 if tni == False else 3

        bnr_bv = int(lines_inen[3 + offss][4:8])
        anzr = int([line for line in open(inqua)][3][3:6])

        # read list of fragment basis vectors bvs = {[BV,rel]}
        bvs = []
        for anz in range(bnr_bv):
            nr = int(lines_inen[4 + offss + 2 * anz].split()[1])
            for bv in range(0, anzr):
                try:
                    if int(lines_inen[5 + offss + 2 * anz].split()[bv]) == 1:
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
                        while bvinstruct_part_3[nnn] < bv[0]:
                            nnn += 1

                        bvinstruct_tmp_3[nnn] += 1

                        # assign the width set to an entry in the label-width dictionary
                        width_blocks[partition3[nnn]].append([
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
        # CAREFUL: rjust might place widths errorously in file!

    zmax = 8

    tm = []
    block_stru = {}
    for block3 in width_blocks:
        tmp = [zmax for i in range(int(len(width_blocks[block3]) / zmax))]
        if len(width_blocks[block3]) % zmax != 0:
            tmp += [len(width_blocks[block3]) % zmax]
        block_stru[block3] = tmp

    if dbg: print(block_stru)

    zerlegungs_struct_4 = []
    zerl_counter = 0
    for s4 in range(len(basis4)):

        label3 = dicti[basis4[s4][0]]
        zerlegungs_struct_3 = block_stru[label3]
        zerlegungs_struct_4.append([zerlegungs_struct_3, label3])
        for n in range(len(zerlegungs_struct_3)):
            zerl_counter += 1
            outs += '%3d%60s%s\n%3d%3d\n' % (zerlegungs_struct_3[n], '',
                                             'Z%d' % zerl_counter,
                                             zerlegungs_struct_3[n], len(relw))
            for bv in width_blocks[label3][sum(zerlegungs_struct_3[:n]
                                               ):sum(zerlegungs_struct_3[:n +
                                                                         1])]:
                outs += '%48s%-12.6f%-12.6f\n' % ('', float(bv[0]), float(
                    bv[1]))
            for rw in range(0, len(relw)):
                outs += '%12.6f' % float(relw[rw])
                if ((rw != (len(relw) - 1)) & ((rw + 1) % 6 == 0)):
                    outs += '\n'
            outs += '\n'
            for bb in range(0, zerlegungs_struct_3[n]):
                outs += '  1  1\n'
                if zerlegungs_struct_3[n] < 7:
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


def inen_bdg_4(bas, jay, co, fn='INEN', pari=0, nzop=31, tni=11, idum=2):
    # idum=2 -> I4 for all other idum's -> I3
    # NBAND1,IDUM,NBAND3,NZOP,IFAKD,IGAK,NZZ,IAUW,IDRU,IPLO,IDUN,ICOPMA(=1 -> stop after N,H output)
    head = '%3d%3d 12%3d  1  0 +0  0  0 -1  0 +0\n' % (tni, idum, nzop)
    head += '  1  1  0  1  0  0  0  0  0  0  0  0  0  0  1  1\n'

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


def inen_str_4(coeff,
               wr,
               bvs,
               uec,
               phys_chan=[2, 2, 0],
               dma=[1, 0, 1, 0, 1, 0, 1],
               jay=0,
               anzch=-1,
               pari=0,
               nzop=9,
               tni=10,
               fn='INEN_STR'):

    s = '%3d  2 12%3d  1  1 +2  0  0 -1\n' % (tni, nzop)

    s += '  1  1  0  1  0  0  0  0  0  0  0  0  0  0  1  1\n'

    s += coeff + '\n'

    anzch = 1

    # SPIN #CHANNELS
    s += '%4d%4d   0   0%4d   1\n' % (int(2 * jay), anzch, pari)

    # FRAGMENT-EXPANSION COEFFICIENTS

    s += '%4d\n' % len(uec)

    for cf in uec:
        s += '%-20.9f\n' % cf

    # ------------------------------------ phys chan

    s += '%3d%3d%3d\n' % (phys_chan[0], phys_chan[1], phys_chan[2])
    s += '%4d' % len(bvs)
    di = 1
    for i in range(len(bvs)):
        di += 1
        s += '%4d' % (i + 1)
        if ((di % 20 == 0) | (int(i + 1) == len(bvs))):
            s += '\n'
            di = 0
    di = 1
    for i in range(1, 1 + len(uec)):
        s += '%4d' % i
        if ((di % 20 == 0) | (di == len(bvs))):
            s += '\n'
        di += 1

    nbr_relw_phys_chan = len(wr)
    for i in range(1, nbr_relw_phys_chan + 1):
        s += '%3d' % int(1)
        if ((int(i) % 50 == 0) | (int(i) == nbr_relw_phys_chan)):
            s += '\n'

    # ------------------------------------ DIST chan
    #rand_coef = []
    #thl = 1e-2
    #uclist = [float(c) for c in uec]

    #while rand_coef == []:
    #    rand_ind = np.random.randint(1, len(uec) - 3, size=len(jj_chans))
    #    rand_coef = abs(np.take(uclist, rand_ind))

    #    if ((min(rand_coef) < thl) |
    #        (np.unique(np.unique(rand_ind) == np.sort(rand_ind))[0] == False)):
    #        rand_coef = []

    #nch = 0
    #for n in range(len(jj_chans)):
    #    bv_in_chan = len(np.unique(jj_chans[n][1]))
    relwoffset = ''
    fd = True
    for i in range(2, len(bvs) - 2):
        s += '%3d%3d%3d' % (phys_chan[0], phys_chan[1], phys_chan[2])
        if fd:
            s += ' -1\n'
            fd = False
        else:
            s += '\n'
        s += '   1%4d\n' % bvs[i][0]
        s += '%-4d\n' % (np.random.randint(1, len(uec) - 2))
        s += relwoffset
        for relw in dma:
            s += '%3d' % relw
        s += '\n'

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


def fromNNto4(relw, zwei_dir, vier_dir, fn):

    # input 1: a two-body basis is read from "zwei_dir" as a set if width parameters
    # for one(!) particular LST configuration
    # input 2: a set of radial widths which expands the relative motion between the two dimers

    # output: input file for the 4-body channel corresponding to a dimer-dimer partition

    outs = ''
    outs += ' 10  8  9  3 00  0  0  0\n%s\n' % fn
    dstru = []
    # -----------------------------------------------------
    inquas = zwei_dir + '/INQUA_N'

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
            if ((rw + 1) % 6 == 0):
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

    with open(vier_dir + '/INQUA_N', 'w') as outfile:
        outfile.write(outs)
    return zstruct