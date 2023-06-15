import subprocess
import os, fnmatch, copy, struct
import numpy as np
import sympy as sy
# CG(j1, m1, j2, m2, j3, m3)
from sympy.physics.quantum.cg import CG
from scipy.linalg import eigh

from three_particle_functions import *

from four_particle_functions import *

from PSI_parallel_M import *

from rrgm_functions import *

from genetic_width_growth import *
from plot_dist import *

from parameters_and_constants import *

import multiprocessing
from multiprocessing.pool import ThreadPool

# prepare spin/orbital matrices for parallel computation
einzel4 = 0
findstablebas = 0

newCal = 0

evalChans = [[1, 1], [2, 2], [3, 3], [4, 4], [5, 5]]  #[[3, 3]]
noDistortion = True

# col = 0 :  wave function (real part)
#       1 :  normalized wave function (real part)
#       2   :  D*Gauss
#       3 :  -I+S*O
#       4 :  -I+S*O +WF IM WECHSELWIRKUNGBEREICH
waveToPlot = 2

# the wave function is ploted for the n-th energy above the channel threshold
energyToPlot = 1

nMatch = 0

if newCal == 2:
    import bridgeA2_plus
    import bridgeA3_plus

J0 = 0

# convention: bound-state-expanding BVs: (1-8), i.e., 8 states per rw set => nzf0*8
J1J2SC = []

if os.path.isdir(sysdir4) == False:
    subprocess.check_call(['mkdir', '-p', sysdir4])
    einzel4 = True

os.chdir(sysdir4)
subprocess.call('cp %s .' % nnpot, shell=True)
subprocess.call('cp %s .' % nnnpot, shell=True)

costr = ''
zop = nOperators if tnni == 11 else 14
for nn in range(1, zop):
    if (nn == 1):
        cf = int(withCoul)
    elif (nn == 2):
        cf = twofac
    elif (nn == 14):
        cf = tnifac
    else:
        cf = 1.0

    costr += '%12.7f' % cf if (nn % 7 != 0) else '%12.7f\n' % cf

if einzel4:
    prepare_einzel4(sysdir4, BINBDGpath, channels_4_scatt)

os.chdir(sysdir4)

twodirs = []
for ch in channels_2:
    twodirs.append(sysdir2base + '/' + ch)

fragment_energies = []
cofli = []
strus = []
zstrus = []
qua_str = []
sbas = []
ph2d = []

subprocess.call('rm -rf INQUA_N', shell=True)

if len(widthSet_relative) != len(channels_4_scatt):
    print(
        'ECCE: not enough relative width sets available to expand all %d asymptotic channels.'
        % len(channels_4_scatt))
    exit()

chnbr = 0

for chan in channels_4_scatt:

    # check if 2-2 channel
    if len(chan[1][0].split('_')) == 2:
        continue

    sysdir21 = sysdir2base + '/' + chan[1][0].split('_')[0]
    sysdir22 = sysdir2base + '/' + chan[1][0].split('_')[1]

    if ((os.path.isfile(sysdir21 + '/bndg_out_%s' % lam) == False) |
        (os.path.isfile(sysdir22 + '/bndg_out_%s' % lam) == False)):
        print("2-body bound state unavailable for L = %s" % lam)
        exit()

    ths1 = get_h_ev(n=1, ifi=sysdir21 + '/bndg_out_%s' % lam)
    ths2 = get_h_ev(n=1, ifi=sysdir22 + '/bndg_out_%s' % lam)
    fragment_energies.append(ths1[0] + ths2[0])

    if (sysdir21 == sysdir22):
        zstrus_tmp, outs = from2to4(zwei_inq=sysdir21 + '/INQUA_N_%s' % lam,
                                    vier_dir=sysdir4,
                                    fn=nnpotstring,
                                    relw=widthSet_relative[chnbr],
                                    app='True')
    else:
        zstrus_tmp, outs = from22to4(zwei_inq_1=sysdir21 + '/INQUA_N_%s' % lam,
                                     zwei_inq_2=sysdir22 + '/INQUA_N_%s' % lam,
                                     vier_dir=sysdir4,
                                     fn=nnpotstring,
                                     relw=widthSet_relative[chnbr],
                                     app='True')

    zstrus.append(zstrus_tmp)
    qua_str.append(outs)

    strus.append(len(zstrus[-1]) * [chan[:2]])
    J1J2SC.append(chan[2])

    sbas.append(get_bsv_rw_idx(inen=sysdir21 + '/INEN_BDG', offset=4, int4=0))
    sbas.append(get_bsv_rw_idx(inen=sysdir22 + '/INEN_BDG', offset=4, int4=0))

    if (sysdir21 == sysdir22):
        ddCoff = parse_ev_coeffs(mult=1,
                                 infil=sysdir21 + '/bndg_out_%s' % lam,
                                 outf='COEFF',
                                 bvnr=1)
    else:
        ddCoff = parse_ev_coeffs_2(infil1=sysdir21 + '/bndg_out_%s' % lam,
                                   infil2=sysdir22 + '/bndg_out_%s' % lam,
                                   outf='COEFF',
                                   bvnr=1)

    ddCoff = np.array(ddCoff).astype(float)
    cofli.append(ddCoff.tolist())

    ph2d.append(
        read_phase(phaout=sysdir21 + '/phaout_%s' % (lam),
                   ch=[1, 1],
                   meth=1,
                   th_shift=''))
    #print(sysdir21, '\n', sysdir22)
    chnbr += 1

# the order matters as we conventionally include physical channels
# in ascending threshold energy. E.g. dd then he3n then tp;
threedirs = []
for ch in channels_3:
    threedirs.append(sysdir3base + '/' + ch)

for sysdir3 in threedirs:

    for line in open(sysdir3 + '/obstru_%s' % lam):
        gogo = True
        for ch in channels_4_scatt:
            for cfg in ch[1]:
                if sysdir3.split('/')[-1] in cfg:
                    J1J2SC.append(ch[2])
                    gogo = False
                    break
            if gogo == False:
                break
        if gogo == False:
            break

    sbas.append(
        get_bsv_rw_idx(inen=sysdir3 + '/INEN', offset=inenOffset, int4=1))
    strus.append([
        dict_3to4[line.strip()] for line in open(sysdir3 + '/obstru_%s' % lam)
    ])
    zstrus.append(
        [int(line.strip()) for line in open(sysdir3 + '/drei_stru_%s' % lam)])

    fragment_energies.append(
        get_h_ev(n=1, ifi=sysdir3 + '/bndg_out_%s' % lam)[0])
    threeCoff = parse_ev_coeffs(mult=0,
                                infil=sysdir3 + '/bndg_out_%s' % lam,
                                outf='COEFF',
                                bvnr=1)

    threeCoff = np.array(threeCoff).astype(float)
    cofli.append(threeCoff.tolist())

    qua_str.append(''.join(
        replace_wrel('%s/inq_3to4_%s' % (sysdir3, lam),
                     widthSet_relative[chnbr])))

    #qua_str.append(''.join(
    #    [line for line in open('%s/inq_3to4_%s' % (sysdir3, lam))]))
    #subprocess.call('cat %s/inq_3to4_%s >> INQUA_N' % (sysdir3, lam),
    #                shell=True)
    chnbr += 1

# if the 2 trimers (t, he) and the 4 dimers (d, dq, nn, pp) which can be formed
# with 4-flavour fermions are degenerate wrt. to their ground-state energy,
# the ordering of physical channels as defined by "channels_4_scatt" is adopted
if deg_channs == 1:
    idx = list(range(len(fragment_energies)))
# otherwise, the 5 asympttotic 2-fragment channels are ordered according to
# descending binding energies
else:
    idx = np.array(fragment_energies).argsort()[::-1]

asyChanLabels = sum([asy[0][1] for asy in [strus[id] for id in idx]], [])[::-1]

strus = sum([strus[id] for id in idx], [])
zstrus = sum([zstrus[id] for id in idx], [])
sbas = sum([sbas[id] for id in idx], [])
cofli = [cofli[id] for id in idx]
J1J2SC = [J1J2SC[id] for id in idx]

qua_str = [qua_str[id] for id in idx]
outs = ' 10  8  9  3 00  0  0  0\n%s\n' % nnpotstring
for qua_part in qua_str:
    outs += qua_part
with open('INQUA_N', 'w') as outfile:
    outfile.write(outs)

sb = []
bv = 1
varspacedim = sum([len(rset[1]) for rset in sbas])

anzch = len(evalChans) if noDistortion == True else int(
    np.max([1, len(sum(cofli, [])) - 5 * len(cofli)]))

print('\n Commencing 4-body calculation with %d channels.' % anzch)
if noDistortion:
    print('NO DISTORTION CHANNELS.')
print('>>> working directory: ', sysdir4, '\n')

for nbv in range(1, varspacedim):
    relws = [
        [1, 0] for n in range(int(0.5 * len(widthSet_relative)))
    ] if nbv % 2 == 0 else [[0, 1]
                            for n in range(int(0.95 * len(widthSet_relative)))]
    sb.append([nbv, sum(relws, [])])

if newCal > 0:

    # include only width parameters > 0.5 (reasonable choice for cutoffs of about 4fm^-1)
    # to guarantee that distortion channels only extend the variational space in the
    # interaction region and that they do NOT interfere with physical, asymptotic states
    maxDistRelW = np.min(
        [len([ww for ww in wsr if ww > 0.5]) for wsr in widthSet_relative])

    relwDistCH = [(n + 1) % 2 for n in range(maxDistRelW)] + [0, 0]

    ma = blunt_ev4(cfgs=strus,
                   bas=sb,
                   dmaa=relwDistCH,
                   j1j2sc=J1J2SC,
                   funcPath=sysdir4,
                   nzopt=zop,
                   frgCoff=cofli,
                   costring=costr,
                   bin_path=BINBDGpath,
                   mpipath=MPIRUN,
                   potNN='%s' % nnpotstring,
                   potNNN='%s' % nnnpotstring,
                   parall=parall,
                   anzcores=max(2, min(len(strus), MaxProc)),
                   tnnii=tnni,
                   jay=J0,
                   nchtot=anzch)

    smartEV, basCond, smartRAT = smart_ev(ma, threshold=10**-7)
    gs = smartEV[-1]
    print(
        'jj-coupled hamiltonian yields:\n E_0 = %f MeV\ncondition number = %E\n|coeff_max/coeff_min| = %E'
        % (gs, basCond, smartRAT))

spole_2(nzen=nzEN,
        e0=E0,
        d0=D0,
        eps=epsM,
        bet=Bet,
        nzrw=anzStuez,
        frr=StuezAbs,
        rhg=rgh,
        rhf=StuezBrei,
        pw=0,
        adaptweightUP=SPOLE_adaptweightUP,
        adaptweightLOW=SPOLE_adaptweightLOW,
        adaptweightL=SPOLE_adaptweightL,
        GEW=SPOLE_GEW,
        QD=SPOLE_QD,
        QS=SPOLE_QS)

# if S-POLE terminates with a ``STOP 7'' message,
# the channel ordering might not be in the correct, i.e.,
# ascending-in-energy order; at present, this order needs
# to be entered manually, according to the B(d) and B(dq)
# results obtained in the employed subspaces
# => the order in the <phys_chan> argument of ``inen_str_4'' (called in PSI_parallel_M)

if findstablebas:
    inen = [line for line in open('INEN')]
    for ll in range(len(inen)):
        if ((inen[ll][-3:-1] == '-1') & (len(inen[ll]) == 13)):
            anzDist = int((len(inen) - ll) / 4)
            break

    anzCh = 0
    for DistCh in range(anzDist):

        line_offset = 7 if nOperators == 31 else 6
        inenLine = inen[line_offset][:4] + '%4d' % (
            len(channels_4_scatt) + anzCh) + inen[line_offset][8:]
        #print(inenLine)
        repl_line('INEN', line_offset, inenLine)
        subprocess.run([BINBDGpath + spectralEXE_mpi])
        lastline = [ll for ll in open('OUTPUT')][-1]
        tmp = get_h_ev()
        print('E_0(#Dch:%d)=%4.4f MeV' % (len(channels_4_scatt) + anzCh, tmp))
        if 'NOT CO' in lastline:

            if anzCh == 0:
                print(
                    'Basis numerically unstable. Re-optimze the 2- and/or 3-body bases and/or select a different set of widths to expand the relative motion.'
                )
                exit()

            inen = [line for line in open('INEN')]
            for ll in range(len(inen)):
                if ((inen[ll][-3:-1] == '-1') & (len(inen[ll]) == 13)):
                    anzDist = int((len(inen) - ll) / 4)
                    break

            outs = ''
            for line in range(len(inen)):
                if ((line < ll + (anzCh - 1) * 4) | (line >=
                                                     (ll + anzCh * 4))):
                    outs += inen[line]
                else:
                    print(inen[line])

            with open('tmp', 'w') as outfile:
                outfile.write(outs)
            subprocess.call('cp tmp INEN', shell=True)

        else:
            anzCh += 1

if newCal >= 0:
    subprocess.run([BINBDGpath + spectralEXE_mpi])

    suche_fehler()

a_aa = ''
for channel in channels_2:
    J0 = channels_2[channel][1]

    sysdir2 = sysdir2base + '/' + channel
    phafile = sysdir2 + '/phaout_%s' % lam
    if os.path.isfile(phafile) == False:
        print("2-body phase shifts unavailable for L = %s" % lam)
        exit()
    phaa = read_phase(phaout=phafile, ch=[1, 1], meth=1, th_shift='')
    a_aa += '  %.4g' % ([
        -MeVfm * np.tan(phaa[n][2] * np.pi / 180.) /
        np.sqrt(mn['137'] * phaa[n][0]) for n in range(len(phaa))
    ][0])

channel_thresholds = get_bind_en(n=len(evalChans))
channel_thresholds = ' '.join(['%4.4g' % ee for ee in channel_thresholds])

groundstate_4 = get_h_ev()[0]

redmass = (4. / 4.) * mn['137']
a_of_epsi = []

neps = 0

a_of_epsi = {}
a_of_Ematch = {}
for key in asyChanLabels:
    #a_of_epsi[key] = []
    a_of_Ematch[key] = []

for chToRead in evalChans:
    a_of_epsi['%s--%s' % (chToRead[0], chToRead[1])] = []

for epsi in np.linspace(eps0, eps1, epsNBR):
    spole_2(nzen=nzEN,
            e0=E0,
            d0=D0,
            eps=epsi,
            bet=Bet,
            nzrw=anzStuez,
            frr=StuezAbs,
            rhg=rgh,
            rhf=StuezBrei,
            pw=0,
            adaptweightUP=SPOLE_adaptweightUP,
            adaptweightLOW=SPOLE_adaptweightLOW,
            adaptweightL=SPOLE_adaptweightL,
            GEW=SPOLE_GEW,
            QD=SPOLE_QD,
            QS=SPOLE_QS)

    subprocess.run([BINBDGpath + smatrixEXE_multichann])

    # col = 0 :  wave function (real part)
    #       1 :  normalized wave function (real part)
    #       2 :  D*Gauss
    #       3 :  -I+S*O
    #       4 :  -I+S*O +WF IM WECHSELWIRKUNGBEREICH

    chans = list(range(1, 1 + len(evalChans)))

    plotapproxwave(infi='OUTPUTSPOLE',
                   oufi='expandedWFKT_%d.pdf' % neps,
                   col=waveToPlot,
                   chan=chans,
                   titl='$\epsilon=[ $%s$ ]$fm$^{-2}$' %
                   (' , '.join(['%.4g' % float(ep) for ep in epsi])),
                   nbrE=energyToPlot)

    plotrelativewave(infi='OUTPUTSPOLE',
                     oufi='relWFKT_%d.pdf' % neps,
                     col=waveToPlot,
                     chan=chans,
                     titl='$\epsilon=[ $%s$ ]$fm$^{-2}$' %
                     (' , '.join(['%.4g' % float(ep) for ep in epsi])),
                     nbrE=energyToPlot)

    subprocess.call('cp OUTPUTSPOLE outps_%d' % neps, shell=True)
    subprocess.call('grep "FUNCTIONAL BERUECKSICHTIGT" OUTPUTSPOLE',
                    shell=True)

    plotphas(oufi='4_ph_%d_%s_%s.pdf' % (neps, lam, lecstring),
             diag=True,
             titl='$\epsilon=[ $%s$ ]$fm$^{-2}$' %
             (' , '.join(['%.4g' % float(ep) for ep in epsi])))

    head_str = '# lambda                       channel   a(ch)     eps                        a(2)     B(4)   B(thresh)\n'
    print(head_str)
    for chToRead in evalChans:
        try:
            phdd = read_phase(phaout='PHAOUT',
                              ch=chToRead,
                              meth=1,
                              th_shift='')

            if ((chToRead == [2, 2]) | (chToRead == [3, 3])) & cib:
                a_dd = [
                    appC(phdd[n][2] * np.pi / 180.,
                         np.sqrt(2 * redmass * phdd[n][0]),
                         redmass,
                         q1=1,
                         q2=1) for n in range(len(phdd))
                ]
            else:
                a_dd = [
                    -MeVfm * np.tan(phdd[n][2] * np.pi / 180.) /
                    np.sqrt(2 * redmass * phdd[n][0]) for n in range(len(phdd))
                ]

            chanstr = asyChanLabels[int(chToRead[0]) -
                                    1] + '--' + asyChanLabels[int(chToRead[1])
                                                              - 1]

            results_bare = '%.3f   %30s   %.4g   %.4g%s   %.4g   %s' % (
                float(lam), chanstr, a_dd[0].real, epsi[0], a_aa,
                groundstate_4, channel_thresholds)
            print(results_bare)
            rafile = 'a_result_%s.dat' % chanstr
            with open(rafile, 'w') as outfile:
                outfile.write(head_str + results_bare)

            #a_of_epsi[asyChanLabels[int(chToRead[0]) - 1]].append(
            #    [epsi, a_dd[0].real, a_dd[-1].real])
            a_of_epsi['%s--%s' % (chToRead[0], chToRead[1])].append(
                [epsi, a_dd[0].real, a_dd[-1].real])
            a_of_Ematch[asyChanLabels[int(chToRead[0]) - 1]].append(
                [np.array(phdd)[:, 0], a_dd])

        except:
            print('no phase shifts in <PHAOUT> for channel:  ', chToRead)
            continue

    neps += 1

xx = []
yy = []
leg = []
for ch in a_of_epsi.keys():
    if a_of_epsi[ch] != []:
        xx.append([an[0][0] for an in a_of_epsi[ch]])
        yy.append([an[1] for an in a_of_epsi[ch]])
        leg.append(ch)

plotarray2(outfi='a_of_eps_%s_%s.pdf' % (lam, lecstring),
           infix=[xx],
           infiy=[yy],
           title=['$a_{dd}$ dependence on $\epsilon$'],
           xlab=['$\epsilon$ [fm$^{-2}$]'],
           ylab=['$a_{dd}$ [fm]'],
           leg=[leg],
           plotrange=[''])

xx = []
yy = []
leg = []
for ch in a_of_Ematch.keys():
    if a_of_Ematch[ch] != []:
        epsset = 0
        for eps_add_set in a_of_Ematch[ch]:
            xx.append(eps_add_set[0])
            yy.append(eps_add_set[1])
            leg.append(ch + '-$\epsilon_%d$' % epsset)
            epsset += 1

plotarray2(outfi='a_of_Ematch_%s_%s.pdf' % (lam, lecstring),
           infix=[xx],
           infiy=[yy],
           title=['$a_{dd}$ dependence on $E_{0}$'],
           xlab=['$E_0$ [MeV]'],
           ylab=['$a_{dd}$ [fm]'],
           leg=[leg],
           plotrange=[''])

exit()

phtp = read_phase(phaout='PHAOUT', ch=[1, 1], meth=1, th_shift='')

phhen = read_phase(phaout='PHAOUT', ch=[2, 2], meth=1, th_shift='1-2')
phtphen = read_phase(phaout='PHAOUT', ch=[1, 2], meth=1, th_shift='1-2')

phdd = read_phase(phaout='PHAOUT', ch=[3, 3], meth=1, th_shift='1-3')
phtpdd = read_phase(phaout='PHAOUT', ch=[1, 3], meth=1, th_shift='1-3')
phhendd = read_phase(phaout='PHAOUT', ch=[2, 3], meth=1, th_shift='2-3')

exit()

# this ordering must match the threshold order, e.g., if B(t)>B(3He)>B(d)>B(dq),
# phtp -> evalChans[0]
# phhen -> evalChans[1]
# phdd -> evalChans[2]
# phdqdq -> evalChans[3]
phtp = read_phase(phaout='PHAOUT', ch=evalChans[0], meth=1, th_shift='')
phhen = read_phase(phaout='PHAOUT', ch=evalChans[1], meth=1, th_shift='1-2')
phdqdq = read_phase(phaout='PHAOUT', ch=evalChans[3], meth=1, th_shift='1-4')
#phmix = read_phase(phaout='PHAOUT', ch=evalChans[3], meth=1, th_shift='1-2')

write_phases(ph2d[0],
             filename='np3s_phases_%s.dat' % lam,
             append=0,
             comment='',
             mu=0.5 * mn['137'])
write_phases(ph2d[1],
             filename='np1s_phases_%s.dat' % lam,
             append=0,
             comment='',
             mu=0.5 * mn['137'])
write_phases(phdd,
             filename='d-d_phases_%s.dat' % lam,
             append=0,
             comment='',
             mu=mn['137'])
write_phases(phdqdq,
             filename='dq-dq_phases_%s.dat' % lam,
             append=0,
             comment='',
             mu=mn['137'])
write_phases(phtp,
             filename='t-p_phases_%s.dat' % lam,
             append=0,
             comment='',
             mu=(2. / 3.) * mn['137'])
write_phases(phhen,
             filename='he3-n_phases_%s.dat' % lam,
             append=0,
             comment='',
             mu=(2. / 3.) * mn['137'])

a_np3s = [
    anp(ph2d[0][n][2] * np.pi / 180., np.sqrt(mn['137'] * ph2d[0][n][0]))
    for n in range(len(ph2d[0]))
]

a_np1s = [
    anp(ph2d[1][n][2] * np.pi / 180., np.sqrt(mn['137'] * ph2d[1][n][0]))
    for n in range(len(ph2d[1]))
]

a_dd = [
    appC(phdd[n][2] * np.pi / 180.,
         np.sqrt(2 * mn['137'] * phdd[n][0]),
         mn['137'],
         q1=1,
         q2=1) for n in range(len(phdd))
] if withCoul == True else [
    -MeVfm * np.tan(phdd[n][2] * np.pi / 180.) /
    np.sqrt(2 * mn['137'] * phdd[n][0]) for n in range(len(phdd))
]

a_dqdq = [
    appC(phdqdq[n][2] * np.pi / 180.,
         np.sqrt(2 * mn['137'] * phdqdq[n][0]),
         mn['137'],
         q1=1,
         q2=1) for n in range(len(phdqdq))
] if withCoul == True else [
    -MeVfm * np.tan(phdqdq[n][2] * np.pi / 180.) /
    np.sqrt(2 * mn['137'] * phdqdq[n][0]) for n in range(len(phdqdq))
]

a_tp = [
    appC(phtp[n][2] * np.pi / 180.,
         np.sqrt((3. / 2.) * mn['137'] * phtp[n][0]), (3. / 4.) * mn['137'],
         q1=1,
         q2=1) for n in range(len(phtp))
] if withCoul == True else [
    -MeVfm * np.tan(phtp[n][2] * np.pi / 180.) / np.sqrt(
        (2. / 3.) * mn['137'] * phtp[n][0]) for n in range(len(phtp))
]

a_hen = [
    -MeVfm * np.tan(phhen[n][2] * np.pi / 180.) / np.sqrt(
        (3. / 2.) * mn['137'] * phhen[n][0]) for n in range(len(phhen))
]

outs = '#    E_match       a_np3s       a_np1s         a_dd       a_dqdq         a_tp        a_hen\n'
for n in range(len(a_dqdq)):
    outs += '%12.4f %12.4f %12.4f %12.4f %12.4f %12.4f %12.4f\n' % (
        ph2d[0][n][0], a_np3s[n].real, a_np1s[n].real, a_dd[n].real,
        a_dqdq[n].real, a_tp[n].real, a_hen[n])

rafile = 'a_ratio_%s.dat' % lam
with open(rafile, 'w') as outfile:
    outfile.write(outs)

tmp = '\"ratfile=\'%s\' ; lambda=\'%s\'\"' % (rafile, lam)

os.system('gnuplot -e %s ' % tmp + pathbase + '/src_python/phases_np.gnu')
os.system('mv 0p_phases.pdf 0p_phases_%s.pdf' % lam)

#subprocess.call('rm -rf TQUAOUT.*', shell=True)
#subprocess.call('rm -rf TDQUAOUT.*', shell=True)
subprocess.call('rm -rf DMOUT.*', shell=True)
subprocess.call('rm -rf DRDMOUT.*', shell=True)
subprocess.call('rm -rf matout_*.*', shell=True)