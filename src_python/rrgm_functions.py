import os, re, itertools, math
import numpy as np
import random

from parameters_and_constants import *

from sympy.physics.quantum.cg import CG
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import mpmath


def output_nbr(outfi='tmpout', outval=0):
    s = '%12.12f' % float(outval)
    with open(outfi, 'w') as outfile:
        outfile.write(s)


def plotphas(infi='PHAOUT', oufi='tmp.pdf', chs=[], titl=''):

    phases = {}
    etas = {}

    method = '1'
    tmp = np.array([
        line.split() for line in open(infi) if (line.split()[-1] == method)
    ]).astype(float)

    for line in tmp:
        for ch in chs:
            if ((ch[0] == line[2]) & (ch[1] == line[3])):

                chastr = '%d-%d' % (line[2], line[3])
                chastrTH = chastr if int(line[2]) == 1 else '%d-%d' % (1, 1)
                try:
                    Exx = float(line[0]) if int(
                        line[2]) == 1 else phases[chastrTH][-1][
                            0]  #+ float(line[1])
                    tmp_phas = float(line[10])
                    tmp_eta = float(line[9])
                    phases[chastr].append([Exx, tmp_phas])
                    etas[chastr].append([Exx, tmp_eta])
                except:
                    phases[chastr] = []
                    etas[chastr] = []
                    Exx = float(line[0]) if int(
                        line[2]) == 1 else phases[chastrTH][-1][
                            0]  #+ float(line[1])
                    phases[chastr].append([Exx, float(line[10])])
                    etas[chastr].append([Exx, float(line[9])])
            #elif (line[2] < line[3]):
            #    chastrTH = '%d-%d' % (1, 1)
            #    chastr = '%d-%d' % (line[2], line[3])
            #    try:
            #        Exx = phases[chastrTH][-1][0]  #+ float(line[1])
            #        phases[chastr].append([Exx, float(line[10])])
            #    except:
            #        phases[chastr] = []
            #        Exx = phases[chastrTH][-1][0]  #+ float(line[1])
            #        phases[chastr].append([Exx, float(line[10])])

    plt.cla()

    fig = plt.figure()

    #if titl != '':
    #    plt.title(titl)

    ax1 = fig.add_axes([0.1, 0.5, 0.8, 0.4], xticklabels=[])
    ax1.xaxis.set_ticks_position('top')

    #leg = ax1.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
    #    ncol=1, mode="expand", borderaxespad=0.)
    ax1.set_ylabel(r'$\delta$ [deg]')

    endiag = []
    for cha in phases:
        en = np.array(phases[cha])[:, 0]
        pha = np.array(phases[cha])[:, 1]

        if cha.split('-')[0] == cha.split('-')[1]:
            stylel = 'solid'
            mark = 3
            linew = 2.5
        else:
            stylel = 'dashdot'
            mark = 1
            linew = 0.5

        ax1.plot(en,
                 pha,
                 label=''.join(cha),
                 linestyle=stylel,
                 marker=mark,
                 linewidth=linew)

    ax1.legend(loc='best', numpoints=1)

    ax2 = fig.add_axes([0.1, 0.1, 0.8, 0.4], ylim=(0, 1))
    ax2.set_xlabel(r'$E_{cm}$ [MeV]        (%s)' % titl)
    ax2.set_ylabel(r'$\eta$')
    for cha in phases:
        en = np.array(phases[cha])[:, 0]
        eta = np.array(etas[cha])[:, 1]

        if cha.split('-')[0] == cha.split('-')[1]:
            stylel = 'solid'
            mark = 3
            linew = 2.5
        else:
            stylel = 'dashdot'
            mark = 1
            linew = 0.5

        ax2.plot(en, eta, linestyle='solid', linewidth=linew)

    plt.savefig(oufi)


def plotphas_old(infi='PHAOUT', oufi='tmp.pdf'):

    # read entire file
    file = [line for line in open(infi)]

    # obtain matrix indices
    ch = [line.split()[2:4] for line in file]
    chu = []
    for ee in ch:
        if ((ee in chu) | (ee[::-1] in chu)) == False:
            chu.append(ee)

    # read phases
    method = '1'
    phs = []
    plt.cla()
    plt.subplot(111)
    #plt.set_title("channel: neutron-neutron")
    #leg = ax1.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
    #    ncol=1, mode="expand", borderaxespad=0.)
    plt.xlabel(r'$E_{cm}$ [MeV]')
    plt.ylabel(r'$\delta$ [deg]')

    endiag = []
    for cha in chu:
        tmp = np.array([
            line.split() for line in file
            if ((line.split()[2:4] == cha) & (line.split()[-1] == method))
        ]).astype(float)
        if cha == ['1', '1']:
            endiag = tmp[:, 0]

        if cha[0] == cha[1]:
            en = endiag
            pha = np.pad(tmp[:, 10], (len(en) - len(tmp[:, 10]), 0),
                         'constant',
                         constant_values=(0, 0))
        else:
            en = tmp[:, 0]
            pha = tmp[:, 10]

        stylel = 'solid' if cha[0] == cha[1] else 'dashdot'
        plt.plot(en, pha, label=''.join(cha), linestyle=stylel)

    plt.legend(loc='best', numpoints=1)
    plt.savefig(oufi)


def c_eta(x):
    #print(np.sqrt(2 * np.pi * x / (np.exp(2 * np.pi * x) - 1)))
    #exit()
    return np.sqrt(2 * np.pi * x / (np.exp(2 * np.pi * x) - 1))


def eta(k, mu, q1=1, q2=1):
    # fine-structure constant in natural units
    alpha = 0.0072973525376
    return q1 * q2 * alpha * mu / k


def hn(k, mu, q1=1, q2=1):
    # H(eta) according to 3NCSBv1.pdf eq.(12)
    hfunc = mpmath.psi(
        0,
        eta(k, mu, q1, q2) * 1j) + 1 / (2 * 1j * eta(k, mu, q1, q2)) - np.log(
            1j * eta(k, mu, q1, q2))
    #print(hfunc)
    #exit()
    return np.real(hfunc)


def appC(d, k, mu, q1=1, q2=1):
    # fine-structure constant in natural units
    alpha = 0.0072973525376
    return -MeVfm / (c_eta(eta(k, mu, q1, q2) + eps)**2 * k *
                     (1 / (np.tan(d) + eps) - 0 * 1j) +
                     q1 * q2 * alpha * mu * hn(k, mu, q1, q2))


def anp(d, k):
    return -MeVfm / k * np.tan(d)


def polyWidths(wmin=10**-2, wmax=10, nbrw=10, npoly=4):

    wds = np.flip([(2. / x)**npoly * (wmin - wmax) / (1 - 2**npoly) + wmax +
                   (wmax - wmin) / (-1. + 1. / 2**npoly)
                   for x in np.linspace(1, 2, nbrw + 2)])
    return wds[1:-1]


def non_zero_couplings(j1, j2, j3):
    # we are interested in viable m1,m3 combinations
    # (L ml,J_deut mJd|J_lit mlit)
    # ECCE: CG takes J *NOT* 2J, i.e., also fractional angular momenta
    m1m3 = []
    m1 = np.arange(-j1, j1 + 1)
    m3 = np.arange(-j3, j3 + 1)

    for mM in np.array(np.meshgrid(m1, m3)).T.reshape(-1, 2):
        clg = CG(j1, mM[0], j2, mM[1] - mM[0], j3, mM[1]).doit()
        cg = 0 if ((clg == 0) | np.iscomplex(clg)) else float(clg.evalf())
        if (cg == 0):
            continue
        m1m3.append(mM)

    return m1m3, m1, m3


def nearest(list1, list2):
    # return list of indices in list2 which correspond to the position of the element with the smallest distance to the element in list1

    indlistn = []
    indlistf = []
    for e1 in list1:
        disti = np.abs(e1 * np.ones(len(list2)) - np.array(list2))
        indlistn.append(np.argmin(disti) + 1)
        indlistf.append(np.argsort(disti) + 1)
    return indlistn, indlistf


def sparse_subset(inset, mindist=0.001):
    """Return a maximal list of elements of points such that no pairs of
    points in the result have distance less than r.

    """
    points = np.sort(inset)[::-1].tolist()

    removing = True
    tmpts = points

    while removing:
        for p, q in itertools.combinations(tmpts, 2):
            if np.linalg.norm(np.array(p) - np.array(q)) < mindist:
                tmpts.remove(q)
                break
        removing = False if len(tmpts) == len(points) else True

    return np.array(tmpts)


def suche_fehler(ifi='OUTPUT'):
    out = [line for line in open(ifi)]
    for nj in range(1, len(out)):
        if (out[nj].find("FEHLERHAFT") >= 0):
            print("TDR2ENDMAT: DIAGONALISATION FEHLERHAFT")
            exit()

    return


def get_n_ev(n=1, ifi='OUTPUT'):
    out = [line for line in open(ifi)]
    E_0 = []
    for nj in range(1, len(out)):
        if (out[nj].strip() == "EIGENWERTE DER NORMMATRIX"):
            for njj in range(len(out)):
                if out[nj + 3 + njj] != '\n':
                    E_0.append(out[nj + 3 + njj].split())
                else:
                    break
            break
    E_0 = np.array(sum(E_0, [])).astype(float)

    if n > 0:
        return E_0[:n]
    else:
        return E_0[0] / E_0[-1]


def get_h_ev(n=1, ifi='OUTPUT'):
    out = [line for line in open(ifi)]

    suche_fehler(ifi)
    for nj in range(1, len(out)):
        if (out[nj].strip() == "EIGENWERTE DES HAMILTONOPERATORS"):
            E_0 = out[nj + 3].split()

    return np.array(E_0[:n]).astype(float)


def get_bind_en(n=1, ifi='OUTPUT'):
    out = [line for line in open(ifi)]
    for nj in range(1, len(out)):
        if (out[nj].strip() == "BINDUNGSENERGIEN IN MEV"):
            E_0 = [float(out[nj + 1][10 * m:10 * (m + 1)]) for m in range(n)]
            break
    return np.array(E_0[:n]).astype(float)


def get_kopplungs_ME(op=4, ifi='OUTPUT'):

    out = [line for line in open(ifi)]

    for nj in range(1, len(out)):
        if (out[nj].strip() == 'KOPPLUNGSMATRIX FUER OPERATOR    %d' % op):
            E_0 = out[nj + 1].split()[0].strip()

    return float(E_0)


def overlap(bipa, chh, Lo=6.0, pair='singel', mpi='137'):

    prep_pot_files_pdp(Lo, 2.0, (-1)**(len(pair) % 2 - 1) * 2.0, 0.0, 0.0, 0.0,
                       0.0, 'pot_' + pair)
    repl_line('INQUA_N', 1, 'pot_' + pair + '\n')

    os.system(bipa + 'QUAFL_' + mpi + '.exe')

    uec = [line for line in open('COEFF')]
    s = ''
    for a in uec:
        s += a

    os.system('cp inen_b INEN')

    repl_line('INEN', 0, ' 10  2 12  9  1  1 -1  0  0 -1\n')
    repl_line('INEN', 2,
              '%12.6f%12.6f%12.6f\n' % (float(1.), float(1.), float(0.)))

    with open('INEN', 'a') as outfile:
        outfile.write(s)

    os.system(bipa + 'DR2END_' + mpi + '.exe')

    os.system('cp OUTPUT end_out_over_' + pair + '_' + chh +
              ' && cp INEN inen_over_' + pair + '_' + chh)

    return get_kopplungs_ME()


def purged_width_set(w0, infil='INEN'):

    lines = [line for line in open(infil)]
    anzch = int(lines[4].split()[1])

    rela = []
    for nc in range(anzch):
        rela.append(np.array(lines[int(6 + 2 * nc)].split()).astype(int))

    indices_to_delete = np.nonzero(np.invert(np.any(rela, axis=0)))
    wp = np.delete(w0, indices_to_delete)
    relwip = np.delete(rela, indices_to_delete, axis=1)

    return wp, relwip


def get_quaf_width_set(inqua='INQUA_N'):

    inq = [line for line in open(inqua)]

    nrw = int(inq[3].split()[1])

    rws = [inq[5 + n].strip() for n in range(int(np.ceil(nrw / 6)))]
    s = ''
    for ln in rws:
        s += ln + '  '

    return s.split()


def get_bsv_rw_idx(inen='INEN', offset=7, int4=1):
    ws = []
    inf = [line for line in open(inen)][offset:]
    for nbv in range(int(inf[0][3 + int(int4):6 + int(2 * int4)])):

        tmp = np.array(
            np.nonzero(np.array(
                inf[int(2 * nbv + 2)].split()).astype(int))) + 1
        ws.append([
            int(inf[int(2 * nbv + 1)][3 + int(int4):6 + int(2 * int4)]),
            tmp[0].tolist()
        ])
    return ws


def get_bas(inen='INEN', offset=7, int4=1):
    ws = []
    inf = [line for line in open(inen)][offset:]
    for nbv in range(int(inf[0][3 + int(int4):6 + int(2 * int4)])):

        tmp = np.array(inf[int(2 * nbv + 2)].split()).astype(int)
        ws.append([
            int(inf[int(2 * nbv + 1)][3 + int(int4):6 + int(2 * int4)]),
            tmp.tolist()
        ])
    return ws


def sparsify(menge, mindist):

    lockereMenge = []
    menge = np.sort(menge)

    if len(menge) == 1:
        return menge

    nref = 0
    npointer = 1

    while (npointer < len(menge)):

        if (np.abs(float(menge[nref]) - float(menge[npointer])) > mindist):
            lockereMenge.append(float(menge[nref]))
            nref = npointer
            npointer = nref + 1

        else:
            npointer += 1

    if lockereMenge == []:
        return menge[0]

    if (np.abs(float(menge[-1]) - float(lockereMenge[-1])) > mindist):
        lockereMenge.append(float(menge[-1]))

    return np.sort(lockereMenge)[::-1]


def sparsifyOnlyOne(menge_to_reduce, menge_fix, mindist):

    #print('Sorting\n', menge_to_reduce, '\n into\n', menge_fix, '...\n\n')

    lockereMenge = []

    for test_w in menge_to_reduce:
        dazu = True
        for ref_w in menge_fix:
            if (np.abs(float(test_w) - float(ref_w)) < mindist):
                dazu = False
        if (dazu):
            lockereMenge = np.concatenate((lockereMenge, [test_w]))

    return np.sort(lockereMenge)[::-1]


def sparsifyPair(menge_a, menge_b, mindist, anzCL):

    #print('Sorting\n', menge_a, '\n into\n', menge_b, '...\n\n')

    seta = menge_a
    setb = menge_b
    dense = True

    while dense:
        widthremoved = False
        for a in seta:
            for b in setb:
                if (np.abs(float(a) - float(b)) < mindist):

                    if (len(seta) <= len(setb)):
                        setb.remove(b)
                    else:
                        seta.remove(a)
                    widthremoved = True
                    break
            if widthremoved:
                break
        if widthremoved == False:
            dense = False

    return np.sort(seta)[::-1], np.sort(setb)[::-1]


def wid_gen(add=10, addtype='top', w0=w120, ths=[1e-5, 8e2, 0.1], sca=1.):
    tmp = []
    # rescale but consider upper and lower bounds
    for ww in w0:
        if ((float(ww) * sca < ths[1]) & (float(ww) * sca > ths[0])):
            tmp.append(float(float(ww) * sca))
        else:
            tmp.append(float(ww))
    tmp.sort()
    w0 = tmp
    w0diff = min(np.diff(tmp))

    # add widths
    n = 0
    addo = 0
    while n < add:

        if addtype == 'top':
            rf = random.uniform(1.2, 1.5)
            addo = float(w0[-1]) * rf

        elif addtype == 'middle':
            rf = random.uniform(0.2, 2.1)
            addo = random.choice(w0) * rf

        elif addtype == 'bottom':
            rf = random.uniform(0.1, 0.9)
            addo = float(w0[0]) * rf

        tmp = np.append(np.array(w0), addo)
        tmp.sort()
        dif = min(abs(np.diff(tmp)))

        if ((addo > ths[0]) & (addo < ths[1]) & (dif >= w0diff)):
            w0.append(addo)
            w0.sort()
            n = n + 1

    w0.reverse()
    w0 = ['%12.6f' % float(float(ww)) for ww in w0]
    return w0


def repl_line(fn, lnr, rstr):
    s = ''
    fil = [line for line in open(fn)]
    for n in range(0, len(fil)):
        if n == lnr:
            s += rstr
        else:
            s += fil[n]
    with open(fn, 'w') as outfile:
        outfile.write(s)


def prep_pot_file_2N(lam, wiC, baC, ps2):
    s = ''
    s += '  1  1  1  1  1  1  1  1  1\n'
    # pdp:       c p2 r2 LS  T Tp
    s += '  0\n%3d  0  0  0  0  0  0\n' % (int((wiC != 0) | (baC != 0)))
    # central LO Cs and Ct and LOp p*p' C_1-4
    if int((wiC != 0) | (baC != 0)):
        s += '%-20.4f%-20.6f%-20.4f%-20.4f%-20.4f\n' % (1.0, float(lam)**2 /
                                                        4.0, wiC, 0.0, baC)

    with open(ps2, 'w') as outfile:
        outfile.write(s)
    return


def prep_pot_file_2N_pp(lam, wiC, baC, ppC, ps2):
    s = ''
    s += '  1  1  1  1  1  1  1  1  1\n'
    # pdp:      pp nn  c
    s += '  0\n  1  0  1  0  0  0  0\n'
    # central LO Cs and Ct and LOp p*p' C_1-4
    s += '%-20.4f%-20.6f%-20.4f%-20.4f%-20.4f%-20.4f%-20.4f%-20.4f\n' % (
        1.0, float(lam)**2 / 4.0, ppC, 0.0, 0.0, 0.0, 0.0, 0.0)
    s += '%-20.4f%-20.6f%-20.4f%-20.4f%-20.4f%-20.4f%-20.4f%-20.4f\n' % (
        1.0, float(lam)**2 / 4.0, wiC, 0.0, baC, 0.0, 0.0, 0.0)

    with open(ps2, 'w') as outfile:
        outfile.write(s)
    return


def prep_pot_file_3N(lam, ps3='', d10=0.0):
    s = ''
    s += '  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1\n  1  1\n'
    # pure central, no (iso)spin dependence
    s += '%-20.4f%-20.4f%-20.4f%-20.4f\n' % (
        d10, float(lam)**2 / 4.0, float(lam)**2 / 4.0, float(lam)**2 / 4.0)
    # central, (s_j s_k)(t_j t_k) to project, set INEN factors +/- 4
    s += '%-20.4f%-20.4f%-20.4f%-20.4f' % (
        d10, float(lam)**2 / 4.0, float(lam)**2 / 4.0, float(lam)**2 / 4.0)

    with open(ps3, 'w') as outfile:
        outfile.write(s)

    return


def parse_ev_coeffs(mult=0, infil='OUTPUT', outf='COEFF', bvnr=1):
    os.system('cp ' + infil + ' tmp')
    out = [line2 for line2 in open(infil)]
    #for n in range(1,len(out)):
    #    if(out[n].strip()=="EIGENWERTE DES HAMILTONOPERATORS"):
    #        print(float(out[n+3].split()[0]))
    coef = ''
    coeffp = []
    coeff_mult = []
    bvc = 0
    for line in range(0, len(out) - 1):
        if re.search('ENTWICKLUNG DES%3d TEN EIGENVEKTORS' % bvnr, out[line]):
            for bvl in range(line + 2, len(out)):
                if ((out[bvl][:3] == ' KO') | (out[bvl][:3] == '\n') |
                    (out[bvl][:3] == '0 D')):

                    bvc = int(out[bvl -
                                  1].strip().split('/')[-1].split(')')[0])
                    break
                coeffp += [
                    float(coo.split('/')[0])
                    for coo in out[bvl].strip().split(')')[:-1]
                ]
                coef += out[bvl]
            break
    s = ''
    for n in range(len(coeffp)):
        if mult:
            for m in range(len(coeffp) - n):
                if m == 0:
                    s += '%18.10g' % (coeffp[n] * coeffp[n + m]) + '\n'
                # for identical fragments, c1*c2|BV1>|BV2> appears twice and can be summed up => faktor 2
                # see coef_mul_id.exe
                else:
                    s += '%18.10g' % (coeffp[n] * coeffp[n + m] * 2) + '\n'
        else:
            s += '%E' % (coeffp[n]) + '\n'
            #s += '%18.10g' % (coeffp[n]) + '\n'

    ss = s.replace('e', 'E')
    if bvc == 0:
        print("No coefficients found in %s" % infil)
    with open(outf, 'w') as outfile:
        outfile.write(ss)

    return ss.split()


def parse_ev_coeffs_2(infil1='OUTPUT', infil2='OUTPUT', outf='COEFF', bvnr=1):

    os.system('cp ' + infil1 + ' tmp')
    out1 = [line2 for line2 in open(infil1)]
    os.system('cp ' + infil2 + ' tmp')
    out2 = [line2 for line2 in open(infil2)]

    coef = ''
    coeffp1 = []
    coeffp2 = []
    coeff_mult = []

    bvc1 = 0
    for line in range(0, len(out1) - 1):
        if re.search('ENTWICKLUNG DES%3d TEN EIGENVEKTORS' % bvnr, out1[line]):
            for bvl in range(line + 2, len(out1)):
                if ((out1[bvl][:3] == ' KO') | (out1[bvl][:3] == '\n') |
                    (out1[bvl][:3] == '0 D')):

                    bvc1 = int(out1[bvl -
                                    1].strip().split('/')[-1].split(')')[0])
                    break
                coeffp1 += [
                    float(coo.split('/')[0])
                    for coo in out1[bvl].strip().split(')')[:-1]
                ]
                coef += out1[bvl]
            break
    bvc2 = 0
    for line in range(0, len(out2) - 1):
        if re.search('ENTWICKLUNG DES%3d TEN EIGENVEKTORS' % bvnr, out2[line]):
            for bvl in range(line + 2, len(out2)):
                if ((out2[bvl][:3] == ' KO') | (out2[bvl][:3] == '\n') |
                    (out2[bvl][:3] == '0 D')):

                    bvc2 = int(out2[bvl -
                                    1].strip().split('/')[-1].split(')')[0])
                    break
                coeffp2 += [
                    float(coo.split('/')[0])
                    for coo in out2[bvl].strip().split(')')[:-1]
                ]
                coef += out2[bvl]
            break

    s = ''

    for n in range(len(coeffp1)):
        for m in range(len(coeffp2)):
            s += '%18.10g' % (coeffp1[n] * coeffp2[m]) + '\n'

    ss = s.replace('e', 'E')
    if ((bvc1 == 0) | (bvc2 == 0)):
        print("No coefficients found in %s or %s" % (infil1, infil2))
    with open(outf, 'w') as outfile:
        outfile.write(ss)

    return ss.split()


def parse_ev_coeffs_normiert(mult=0,
                             infil='OUTPUT',
                             outf='COEFF_NORMAL',
                             nbv=1):
    os.system('cp ' + infil + ' tmp')
    out = [line2 for line2 in open(infil)]

    coef = ''
    coeffp = []
    coeff_mult = []
    bvc = 0
    for line in range(0, len(out) - 1):
        if re.search('ENTWICKLUNG DES  %d TEN EIGENVEKTORS,AUS' % nbv,
                     out[line]):
            for bvl in range(line + 2, len(out)):
                if '0UE' in out[bvl][:4]:
                    break
                else:
                    coeffp += [
                        float(coo)
                        for coo in out[bvl].strip().split(')')[-1].split()
                    ]
            break
    coeffp = np.where(np.abs(coeffp) > 10**3, 10, coeffp)
    s = ''
    bvc = len(coeffp)
    for n in range(len(coeffp)):
        s += '%E' % (coeffp[n]) + '\n'
        #s += '%18.10g' % (coeffp[n]) + '\n'
    ss = s.replace('e', 'E')
    if bvc == 0:
        print("No coefficients found in %s" % infil)
    with open(outf, 'w') as outfile:
        outfile.write(ss)

    return coeffp


def read_phase(phaout='PHAOUT', ch=[1, 1], meth=1, th_shift=''):
    lines = [line for line in open(phaout)]

    th = {'': 0.0}
    phase = []
    phc = []
    ech = [0]

    try:
        for ln in range(0, len(lines)):
            if (lines[ln].split()[2] != lines[ln].split()[3]):
                th[lines[ln].split()[2] + '-' + lines[ln].split()[3]] = abs(
                    float(lines[ln].split()[1]) - float(lines[ln].split()[0]))
        ths = th[th_shift]
        #print(th)

        for ln in range(0, len(lines)):
            if ((int(lines[ln].split()[2]) == ch[0]) &
                (int(lines[ln].split()[3]) == ch[1]) &
                (int(lines[ln].split()[11]) == meth)):
                # energy tot -- energy th -- phase
                phase.append([
                    float(lines[ln].split()[0]),
                    float(lines[ln].split()[1]) + ths,
                    float(lines[ln].split()[10])
                ])
    except:
        return []

    return phase


def write_phases(ph_array,
                 filename='tmp.dat',
                 append=0,
                 comment='',
                 mu=mn['137']):

    outs = ''

    if append == 1:
        oldfile = [line for line in open(filename)]
        for n in range(len(oldfile)):
            if oldfile[n].strip()[0] != '#':
                outs += oldfile[n].strip() + ' %12.8f' % float(
                    ph_array[n - 1][2]) + '\n'
            else:
                outs += oldfile[n]

    elif append < 0:
        oldfile = [line for line in open(filename)][:append]
        for n in range(len(oldfile)):
            if oldfile[n].strip()[0] != '#':
                for entry in oldfile[n].strip().split()[:append]:
                    outs += ' %12.8f' % float(entry)
                outs += ' %12.8f' % float(ph_array[n - 1][2]) + '\n'
            else:
                outs += oldfile[n]

    elif append == 0:
        outs = '#% -10s  %12s %12s %12s' % ('E_tot', 'E_tot-Eth', 'Phase(s)',
                                            'scatt. length\n')
        for line in range(len(ph_array)):
            atmp = MeVfm * (-1) * np.tan(
                ph_array[line][2] * np.pi / 180.) / np.sqrt(
                    2 * mu * ph_array[line][0])
            outs += '%12.8f %12.8f %12.8f %12.8f' % (float(
                ph_array[line][0]), float(
                    ph_array[line][1]), float(ph_array[line][2]), atmp)
            outs += '\n'

    if comment != '': outs += comment + '\n'

    with open(filename, 'w') as outfile:
        outfile.write(outs)
    return


def identicalt_stru(dir, spli):
    out = [line for line in open(dir + 'OUTPUT')]
    for n in range(1, len(out)):
        if (out[n].strip(
        ) == "ENTWICKLUNG DES  1 TEN EIGENVEKTORS,AUSGEDRUECKT DURCH NORMIERTE BASISVEKTOREN"
            ):
            sp = [t.strip() for t in out[n + 2:n + 2 + spli]]
            sp2 = [t.strip() for t in out[n + 2 + spli:n + 2 + 2 * spli]]
    le = 0
    for tt in sp:
        le += len(tt.split()) - 1
    le2 = 0
    for tt in sp2:
        le2 += len(tt.split()) - 1
    return le, le2


def determine_struct(inqua='INQUA_N'):
    """
    the number of basis vectors for given Spin- and Spatial coupling
    schemes is deduced from INQUA_N, where the width tuples associated
    with the scheme are put in a decadent ordered;
    'count #BV until the first width in the tuple is larger!'
    """

    lines_inqua = [line for line in open(inqua)]
    lines_inqua = lines_inqua[2:]

    stru = []

    bv_in_scheme = 0
    block_head = 0
    wr1_oldblock = 1e12

    while block_head < len(lines_inqua):

        try:
            bvinz = int(lines_inqua[block_head][:4])
        except:
            break

        rel = int(lines_inqua[block_head + 1][4:7])
        nl = int(rel / 6)
        if rel % 6 != 0:
            nl += 1

        wr1_newblock = float(lines_inqua[block_head + 2].strip().split()[0])
        if wr1_newblock > wr1_oldblock:
            stru.append(bv_in_scheme)
            bv_in_scheme = 0

        #print(wr1_oldblock, wr1_newblock, block_head)

        bv_in_scheme += bvinz

        wr1_oldblock = float(lines_inqua[block_head + 1 +
                                         bvinz].strip().split()[0])

        if bvinz < 7:
            block_head = block_head + 2 + bvinz + nl + 2 * bvinz
        else:
            block_head = block_head + 2 + bvinz + nl + 3 * bvinz

    stru.append(bv_in_scheme)

    return [np.sum(stru[:n]) for n in range(1, len(stru) + 1)]


def plotrelativewave(infi='OUTPUTSPOLE',
                     oufi='tmp.pdf',
                     col=1,
                     chan=[1],
                     titl='',
                     nbrE=1):

    data = [line for line in open(infi)]

    plt.cla()
    plt.subplot(111, label=r'relWFKT')

    if titl != '':
        plt.title(titl)
    #leg = ax1.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
    #    ncol=1, mode="expand", borderaxespad=0.)
    plt.xlabel(r'$R_{rel}$ [fm]')
    plt.ylabel(r'$h_+(kR)$ [?]')

    start = 0
    end = 1
    nE = 0
    ch = 1

    for nj in range(1, len(data)):

        cmap = plt.get_cmap('winter')

        if (-1 not in [
                data[nj + (ch - 1) * 4].find('DER%3d TE KANAL IST OFFEN' % ch)
                for ch in chan
        ]):
            ch = 1
            for njj in range(start, len(data)):
                #if (data[njj].find(
                #        'Q--                   H+                     H-') >=
                #        0):
                if (data[njj].find(
                        'DARSTELLUNG DER STREUFUNKTIONEN IM%3d TEN.KANAL' % ch)
                        >= 0):
                    #if (data[njj].find(
                    #        'R(FM) W-FUNKTION, N**1/2*W-FUNK., D*Gauss, -I+S*O, -I+S*O +WF IM WECHSELWIRKUNGBEREICH' % ch)
                    #        >= 0):
                    start = njj + 6
                    for njjj in range(start + 1, len(data)):
                        if (len(data[njjj].split()) != 10):
                            end = njjj
                            break
                    nE += 1

                    if nE == nbrE:

                        wfdata = np.array([
                            line.split() for line in data[start + 1:end]
                        ]).astype(float)

                        rr = np.array(wfdata)[:, 0][0::2]
                        rrAPP = np.array(wfdata)[:, 0][1::2]
                        wfkt = np.array(wfdata)[:, int(1 + 2 * col)][0::2]
                        wfktAPP = np.array(wfdata)[:, int(1 + 2 * col)][1::2]

                        colo = c = cmap(ch / len(chan))

                        #plt.ylim(-2, 2)
                        plt.plot(rr,
                                 wfkt,
                                 label='F_L(r,ch=%d)' % ch,
                                 linestyle='solid',
                                 color=colo)
                        plt.plot(rrAPP,
                                 wfktAPP,
                                 label='F_L(r,ch=%d) APP' % ch,
                                 linestyle='dashdot',
                                 color=colo)

                        if ch == chan[-1]:
                            plt.legend(loc='best', numpoints=1)
                            plt.savefig(oufi)
                            return

                        ch += 1
                        nE -= 1


def plotapproxwave(infi='OUTPUTSPOLE',
                   oufi='tmp.pdf',
                   col=0,
                   chan=[1],
                   titl='',
                   nbrE=1):

    data = [line for line in open(infi)]

    plt.cla()
    ax = plt.subplot(111, label=r'approxWFKT')

    if titl != '':
        plt.title(titl)
    #leg = ax1.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
    #    ncol=1, mode="expand", borderaxespad=0.)
    plt.xlabel(r'$R_{rel}$ [fm]')

    start = 0
    end = 1
    nE = 0
    ch = 1

    for nj in range(1, len(data)):

        cmap = plt.get_cmap('winter')

        if (-1 not in [
                data[nj + (ch - 1) * 4].find('DER%3d TE KANAL IST OFFEN' % ch)
                for ch in chan
        ]):

            ch = 1

            if nE == nbrE:

                for njj in range(nj, len(data)):

                    if (data[njj].find(
                            'R(FM)   WELLENFUNKTION IM KANAL%3d' % ch) >= 0):

                        ylab = data[njj + 1].split(',')[col].strip()

                        for njjj in range(njj + 1, len(data)):

                            if (data[njjj].split() == [str(ch), str(ch)]):

                                for m in range(njjj, len(data)):

                                    if len(data[m + 2].split()) != 11:
                                        break
                                wfdata = np.array([
                                    line.split()
                                    for line in data[njjj + 2:m + 2]
                                ]).astype(float)

                                rr = np.array(wfdata)[:, 0]
                                wfkt = np.array(wfdata)[:, int(1 + 2 * col)]
                                colo = c = cmap(ch / len(chan))

                                #ym = np.median(wfkt)
                                ym = 2
                                #plt.ylim(-ym, ym)

                                plt.plot(rr,
                                         wfkt,
                                         label='F_L(r,ch=%d)' % ch,
                                         linestyle='solid',
                                         color=colo)

                                ch += 1
                                break

                plt.ylabel(r'%s' % ylab)
                plt.legend(loc='best', numpoints=1)
                plt.savefig(oufi)
                return

            else:
                nE += 1


def plotDcoeff(infi='OUTPUTSPOLE',
               oufi='tmp.pdf',
               col=0,
               chan=[1],
               titl='',
               nbrE=1):

    data = [line for line in open(infi)]

    start = 0
    end = 1
    nE = 0
    ch = 1

    Dcoeffs = []
    Dcoeffs2 = []

    for nj in range(1, len(data)):

        cmap = plt.get_cmap('winter')

        if (-1 not in [
                data[nj + (ch - 1) * 4].find('DER%3d TE KANAL IST OFFEN' % ch)
                for ch in chan
        ]):

            ch = 1

            if nE == nbrE:

                for njj in range(nj, len(data)):

                    if (data[njj].find('D-KOEFFIZIENTEN') >= 0):

                        for njjj in range(njj + 1, len(data)):

                            if (data[njjj].find('D-KOEFFIZIENTEN') >= 0):
                                break

                            Dcoeffs += data[njjj].split()
#                        print(np.array(Dcoeffs).astype(float))
                        for n2 in range(njjj + 1, 2 * njjj - njj):
                            Dcoeffs2 += data[n2].split()
#                        print(np.array(Dcoeffs2).astype(float))
                        break


#                        exit()

#                            for m in range(njjj, len(data)):
#                                if len(data[m + 2].split()) != 11:
#                                    break
#                            wfdata = np.array([
#                                line.split() for line in data[njjj + 2:m + 2]
#                            ]).astype(float)
#                            rr = np.array(wfdata)[:, 0]
#                            wfkt = np.array(wfdata)[:, int(1 + 2 * col)]
#                            colo = c = cmap(ch / len(chan))
#                            ym = np.median(wfkt)
#                            ym = 2
#                            plt.ylim(-ym, ym)
#                            plt.plot(rr,
#                                     wfkt,
#                                     label='F_L(r,ch=%d)' % ch,
#                                     linestyle='solid',
#                                     color=colo)
#                            ch += 1
#                            break

                n_bins = 40

                fig, ax = plt.subplots()

                # We can set the number of bins with the *bins* keyword argument.
                Dcoeffs = np.array(Dcoeffs).astype(float)
                Dcoeffs2 = np.array(Dcoeffs2).astype(float)
                varm = max([np.var(Dcoeffs), np.var(Dcoeffs2)])

                n, bins, patches = ax.hist(Dcoeffs,
                                           bins=n_bins,
                                           density=True,
                                           stacked=True,
                                           cumulative=False,
                                           alpha=0.5)
                n2, bins2, patches2 = ax.hist(Dcoeffs2,
                                              bins=n_bins,
                                              density=True,
                                              stacked=True,
                                              cumulative=False,
                                              alpha=0.5)

                #print("patches: ", patches)
                #for i in range(10):
                #    print(patches[i])

                ub = float(min(Dcoeffs))
                ob = float(max(Dcoeffs))

                tiklocs = [
                    bins[bb] for bb in range(0, len(n), int(len(n) / 5))
                ]

                plt.xticks(tiklocs, ['%4.2f' % nn for nn in tiklocs],
                           rotation=0)

                plt.title(r'max(variance) = %f' % varm)
                #plt.legend(loc='best', numpoints=1)
                plt.savefig(oufi)
                return

            else:
                nE += 1


def readDcoeff(infi='OUTPUTSPOLE', chan=[1], nbrE=1):

    data = [line for line in open(infi)]

    start = 0
    end = 1
    nE = 0
    ch = 1

    Dcoeffs = []
    Dcoeffs2 = []

    for nj in range(1, len(data)):

        if (-1 not in [
                data[nj + (ch - 1) * 4].find('DER%3d TE KANAL IST OFFEN' % ch)
                for ch in chan
        ]):

            ch = 1

            if nE == nbrE:

                for njj in range(nj, len(data)):

                    if (data[njj].find('D-KOEFFIZIENTEN') >= 0):

                        for njjj in range(njj + 1, len(data)):

                            if (data[njjj].find('D-KOEFFIZIENTEN') >= 0):
                                break

                            Dcoeffs += data[njjj].split()
#                        print(np.array(Dcoeffs).astype(float))
                        for n2 in range(njjj + 1, 2 * njjj - njj):
                            Dcoeffs2 += data[n2].split()


#                        print(np.array(Dcoeffs2).astype(float))
                        break
            else:
                nE += 1

    return np.array(Dcoeffs).astype(float), np.array(Dcoeffs2).astype(float)


def shuffle_distchanns(fin='INEN', fout='INEN'):

    # 1) read the sequentially-ordered file

    inen = [line for line in open(fin)]
    for ll in range(len(inen)):
        if ((inen[ll][-3:-1] == '-1') & (len(inen[ll].strip()) == 10)):
            anzDist = int((len(inen) - ll) / 4)
            lineoffirstDist = ll
            break

    # remove dist-chan marker
    inen[lineoffirstDist] = '  ' + inen[lineoffirstDist].strip()[:-3] + '\n'

    # 2) redistribute the ordered sequence of channels

    nbrCh = int((len(inen) - lineoffirstDist) / 4)
    rndorder = np.arange(nbrCh)
    np.random.shuffle(rndorder)

    outstr = ''.join(inen[:lineoffirstDist])

    nc = 0

    for nch in rndorder:
        if nc == 0:
            inen[lineoffirstDist + 4 * rndorder[nch]] = '  ' + inen[
                lineoffirstDist + 4 * rndorder[nch]].strip() + ' -1\n'

        outstr += ''.join(
            inen[lineoffirstDist + 4 * rndorder[nch]:lineoffirstDist +
                 4 * rndorder[nch] + 4])

        nc += 1

    with open(fout, 'w') as outfile:
        outfile.write(outstr)
