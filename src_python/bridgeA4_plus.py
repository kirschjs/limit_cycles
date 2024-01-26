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
findstablebas = 1
smallestAllowedDistortionW = 0.1
indexOfLargestAllowedDistW = 2
normStabilityThreshold = 10**-18
maxCofDev = 1000.1
newCal = -1

# ECCE: variable whose consistency with evalChans must be given:
# nbr_of_threebody_boundstates ,
evalChans = eDict[lecstring.split('-')[-1]][4]
pltChans = evalChans  #+ [[1, 2], [1, 3], [2, 3]]

chDict = {'[1, 1]': [], '[2, 2]': [], '[3, 3]': []}


def redmass(f1=1, f2=1, mpi='137'):
    return (f1 * f2 / (f1 + f2)) * mn[mpi]


noDistortion = False

# col = 0 :  wave function (real part)
#       1 :  normalized wave function (real part)
#       2 :  D*Gauss
#       3 :  -I+S*O
#       4 :  -I+S*O +WF IM WECHSELWIRKUNGBEREICH
waveToPlot = 0

# col = 0 : H+
#       1 : H-
#       2 : H'+
#       3 : H'-
#       4 : T-1
relwaveToPlot = 1

nMatch2 = 0
nMatch = 0

# the wave function is ploted for the n-th energy above the channel threshold
energyToPlot = 2

if newCal == 2:
    import bridgeA2_plus
    import bridgeA3_plus
    # optimizes a set of distortion channels which should ensure that the exited tetramers are
    # expanded accurately;
    import bridgeA4_opt
    print(
        'ECCE: all fragment optimization codes are expected to have been run separately.'
    )

J0 = 0

# convention: bound-state-expanding BVs: (1-8), i.e., 8 states per rw set => nzf0*8
J1J2SC = []

if os.path.isdir(sysdir4) == False:
    subprocess.check_call(['mkdir', '-p', sysdir4])
    prepare_einzel4(sysdir4, BINBDGpath, channels_4_scatt)

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

os.chdir(sysdir4)

twodirs = []
for ch in channels_2:
    twodirs.append(sysdir2base + '/' + ch)

chFRG = []
fragment_energies = []
fragment_bvrange = []
cofli = []
strus = []
zstrus = []
qua_str = []
ph2d = []

subprocess.call('rm -rf INQUA_N', shell=True)

if len(widthSet_relative) != len(channels_4_scatt):
    print(
        'ECCE: not enough relative width sets available to expand all %d asymptotic channels.'
        % len(channels_4_scatt))
    exit()

chnbr = 0

bvcount = 0
fragment_energies_tmp = []
fragment_bvrange_tmp = []
J1J2SC_tmp = []
cofli_tmp = []
chFRG_tmp = []

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
    fragment_energies_tmp.append(ths1[0] + ths2[0])

    # treat nn=pp if SU(4)
    if ((sysdir21 == sysdir22) | SU4):
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
    fragment_bvrange_tmp.append([bvcount, bvcount + sum(zstrus_tmp)])
    qua_str.append(outs)

    strus.append(len(zstrus[-1]) * [chan[:2]])
    J1J2SC_tmp.append(chan[2])
    chFRG_tmp.append([2, 2])

    if (sysdir21 == sysdir22) | SU4:
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
    cofli_tmp.append(ddCoff.tolist())

    ph2d.append(
        read_phase(phaout=sysdir21 + '/phaout_%s' % (lam),
                   ch=[1, 1],
                   meth=phasCalcMethod,
                   th_shift=''))
    #print(sysdir21, '\n', sysdir22)
    chnbr += 1
    bvcount += sum(zstrus_tmp)

cofli.append(cofli_tmp)
J1J2SC.append(J1J2SC_tmp)
chFRG.append(chFRG_tmp)

fragment_energies.append(fragment_energies_tmp)
fragment_bvrange.append(fragment_bvrange_tmp)

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
                    J1J2SC_tmp = []
                    for nn in range(nbr_of_threebody_boundstates.count(1)):
                        J1J2SC_tmp.append(ch[2])
                    J1J2SC.append(J1J2SC_tmp)
                    gogo = False
                    break
            if gogo == False:
                break
        if gogo == False:
            break

    strus.append([
        dict_3to4[line.strip()] for line in open(sysdir3 + '/obstru_%s' % lam)
    ])
    zstrus.append(
        [int(line.strip()) for line in open(sysdir3 + '/drei_stru_%s' % lam)])

    fragment_energies_tmp = []
    fragment_bvrange_tmp = []
    cofli_tmp = []
    for nn in range(len(nbr_of_threebody_boundstates)):
        if nbr_of_threebody_boundstates[nn] == 0:
            continue
        ew = get_h_ev(n=1 + nn, ifi=sysdir3 + '/bndg_out_%s' % lam)[nn]
        fragment_energies_tmp.append(ew)
        fragment_bvrange_tmp.append([bvcount, bvcount + sum(zstrus[-1])])

        threeCoff = parse_ev_coeffs(mult=0,
                                    infil=sysdir3 + '/bndg_out_%s' % lam,
                                    outf='COEFF',
                                    bvnr=1 + nn)

        threeCoff = np.array(threeCoff).astype(float)
        cofli_tmp.append(threeCoff.tolist())
        chFRG_tmp.append([3, 1])

    chFRG.append(chFRG_tmp)
    cofli.append(cofli_tmp[::-1])
    fragment_energies.append(fragment_energies_tmp[::-1])
    fragment_bvrange.append(fragment_bvrange_tmp)

    qua_str.append(''.join(
        replace_wrel('%s/inq_3to4_%s' % (sysdir3, lam),
                     widthSet_relative[chnbr])))

    #qua_str.append(''.join(
    #    [line for line in open('%s/inq_3to4_%s' % (sysdir3, lam))]))
    #subprocess.call('cat %s/inq_3to4_%s >> INQUA_N' % (sysdir3, lam),
    #                shell=True)
    chnbr += 1
    bvcount += sum(zstrus[-1])

idx = np.array(list(more_itertools.collapse([fragment_energies]))).argsort()

chFRG = [list(more_itertools.collapse([chFRG], levels=2))[id] for id in idx]
nt = 0

for nn in chDict:
    if nt >= len(chFRG):
        break
    chDict[nn] = chFRG[nt]
    nt += 1

print('ECCE >>> fragment masses in the asymptotic channels: ',
      chDict,
      end='\n\n')

# create a threshold-ordered list of asymptotic channels
#
asymptCH = [
    [list(more_itertools.collapse([J1J2SC], levels=2))[id] for id in idx],
    [
        list(more_itertools.collapse([fragment_bvrange], levels=2))[id]
        for id in idx
    ], [list(more_itertools.collapse([cofli], levels=2))[id] for id in idx]
]

idx = np.array([ee[0] for ee in fragment_energies]).argsort()[::-1]
fragment_energies = np.array(fragment_energies, dtype=object)[idx]

asyChanLabels = sum([1 * asy[0][1] for asy in [strus[id] for id in idx]],
                    [])[::-1]

asyChanLabels = list(
    more_itertools.collapse([
        len(J1J2SC[::-1][nn]) * [asyChanLabels[nn]]
        for nn in range(len(asyChanLabels))
    ]))

#strus = sum([strus[id] for id in idx], [])
#zstrus = sum([zstrus[id] for id in idx], [])
strus = sum([strus[id] for id in range(len(strus))], [])
zstrus = sum([zstrus[id] for id in range(len(zstrus))], [])

# this awkward line should take care of multiple fragments corresponding to the
# same strucutre, i.e., an excited trimer
cofli = [cofli[id] for id in idx]
J1J2SC = [J1J2SC[id] for id in idx]
#qua_str = [qua_str[id] for id in idx]

# include only width parameters > 0.5 (reasonable choice for cutoffs of about 4fm^-1)
# to guarantee that distortion channels only extend the variational space in the
# interaction region and that they do NOT interfere with physical, asymptotic states
maxDistRelW = np.min([
    len([ww for ww in wsr if ww > smallestAllowedDistortionW])
    for wsr in widthSet_relative
] + [anzRelw4opt])

relwDistCH = [n % 2 for n in range(np.min([maxDistRelW, 12]))] + [0, 0]
"""
in the `alpha' directory, an basis is expected which was optimized for the
4-body bound-state problem; this basis is added to the variational basis in
the form of distortion channels in order to improve the description/expansion
of the wave function in the non-asymptotic region;
(see, e.g., K. Wildermuth and Y.C. Tang, A Unified Theory of the Nucleus [ch.7, eq.(7.45ff)] )
"""
if os.path.isdir(sysdir4 + '/alpha') == True:
    print(
        'adding additional distortion channels based on optimized bound-state space: \'alpha\''
    )
    qua_alpha = [
        line for line in open(sysdir4 + '/alpha/INQUA_N') if line[0] != 'e'
    ][2:]
    qua_str = qua_str + qua_alpha

    lu_alpha = [
        line for line in open(sysdir4 + '/alpha/lustru_alpha_%s' % lam)
    ]
    ob_alpha = [
        line for line in open(sysdir4 + '/alpha/obstru_alpha_%s' % lam)
    ]
    alpha_strus = [[[lu_alpha[nn].strip()], [ob_alpha[nn].strip()]]
                   for nn in range(len(lu_alpha))]

    strus = strus + alpha_strus

    zstrus_alpha = [
        int(line.strip())
        for line in open(sysdir4 + '/alpha/vier_stru_alpha_%s' % lam)
    ]
    nbr_phy_bv = sum(zstrus)
    zstrus = zstrus + zstrus_alpha

    coflist = list(more_itertools.collapse(cofli))
    distuec = [
        nc + 1 for nc in range(len(coflist))
        if 10**2 > np.abs(coflist[nc]) > 0.1
    ]

    chStr = ''
    bv = 1
    for nn in range(len(zstrus_alpha)):
        for ach in channels_4:
            cfgs = -1
            for ncfg in range(len(channels_4[ach][1])):
                if channels_4[ach][1][ncfg] == alpha_strus[nn][1][0]:
                    cfgs = channels_4[ach][2][ncfg]
                    break
            if cfgs != -1:
                for nd in range(zstrus_alpha[nn]):
                    chStr += '%3d%3d%3d\n' % (cfgs[0], cfgs[1], cfgs[2])
                    chStr += '%4d%4d\n' % (1, nbr_phy_bv + bv)
                    chStr += '%-4d\n' % (np.random.choice(distuec))
                    #s += relwoffset
                    for relw in relwDistCH:
                        chStr += '%3d' % relw
                    chStr += '\n'
                    bv += 1
                break

outs = ' 10  8  9  3 00  0  0  0\n%s\n' % nnpotstring
for qua_part in qua_str:
    outs += qua_part
with open('INQUA_N', 'w') as outfile:
    outfile.write(outs)

sb = []
bv = 1

anzch = len(
    evalChans) if noDistortion == True else sum(zstrus) - 5 * len(cofli)

print('\n Commencing 4-body calculation with %d channels.' % anzch)
if noDistortion:
    print('NO DISTORTION CHANNELS.')
print('>>> working directory: ', sysdir4, '\n')

for nbv in range(1, anzch):
    relws = [[1, 1] for n in range(int(0.5 * anzRelw))
             ] if nbv % 2 == 0 else [[1, 1] for n in range(int(0.5 * anzRelw))]
    sb.append([nbv, sum(relws, [])])

if newCal > 0:

    ma = blunt_ev4(cfgs=strus,
                   bas=sb,
                   dmaa=relwDistCH,
                   j1j2sc=asymptCH,
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
                   nchtot=anzch,
                   distchannels=chStr)

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

    subprocess.call('cp INEN INEN_bkp', shell=True)

    shuffle_distchanns(fin='INEN_bkp', fout='INEN')

    inen = [line for line in open('INEN')]
    for ll in range(len(inen)):

        if ((inen[ll][-3:-1] == '-1') & (len(inen[ll].strip()) == 10)):
            anzDist = int((len(inen) - ll) / 4)
            lineoffirstDist = ll
            break

    anzCh = 1
    var0 = 0.0
    nbrRemoved = 0

    #indexOfLargestAllowedDistW = 3
    #maxDistRelW = 4

    for DistCh in range(anzDist - 1):
        inen = [line for line in open('INEN')]
        # FIRST: select the structure of the distortion

        line_offset = 7 if nOperators == 31 else 6
        inenLine = inen[line_offset][:4] + '%4d' % (
            len(asymptCH[0]) + anzCh) + inen[line_offset][8:]
        repl_line('INEN', line_offset, inenLine)
        # SECOND: for the selected structure, i.e., spin-/orbital-angular momentum
        #         coupling scheme of a particular asymptotic channel, cycle through
        #         the relative widths to be included: |Dch>=|struct,relW>
        dist_line = np.zeros(maxDistRelW + 1)

        # at which line is this string to be inserted?
        linenbrDistCh = int(lineoffirstDist + 4 * (DistCh - nbrRemoved) + 3)
        print(
            'Adding rel. widths for basis-vector structure %3d as distortions.'
            % int(inen[linenbrDistCh - 2].split()[1]))
        for NrelW in range(indexOfLargestAllowedDistW, maxDistRelW, 1):

            dist_line[NrelW] = 1
            dist_line_str = ''.join(['%3d' % ww for ww in dist_line]) + '\n'

            repl_line('INEN', linenbrDistCh, dist_line_str)

            subprocess.run([BINBDGpath + spectralEXE_mpi])
            subprocess.run([BINBDGpath + smatrixEXE_multichann])

            #
            tmp = get_n_ev(n=-1, ifi='OUTPUT')

            if tmp < normStabilityThreshold:
                dist_line[NrelW] = 0
                dist_line_str = ''.join(['%3d' % ww
                                         for ww in dist_line]) + '\n'
                repl_line('INEN', linenbrDistCh, dist_line_str)

#             dc, dc2 = readDcoeff()  # OUTPUTSPOLE
#
#            maxVar = max(np.var(dc), np.var(dc2))
#            varDev = np.abs(maxVar - var0)
#
#            lastline = [ll for ll in open('OUTPUT')][-1]
#            tmp = get_h_ev()
#            print('E_0(#Dch:%d) = %4.4f MeV  D-coeff variance = %4.4f' %
#                  (len(channels_4_scatt) + anzCh, tmp, maxVar))
#            if (('NOT CO' in lastline) | (varDev > maxCofDev)):
#                dist_line[NrelW] = 0
#            maxCofDev = varDev

        if np.any(dist_line) == False:

            print('configuration unstable for all rel. widths.')

            nbrRemoved += 1
            if anzCh == 1:
                print(
                    'Basis numerically unstable. Re-optimze the 2- and/or 3-body bases and/or select a different set of widths to expand the relative motion.'
                )
                break

            outs = ''
            ttmp = 0
            for line in range(len(inen)):
                if ((line < lineoffirstDist + (anzCh - 1) * 4) |
                    (line >= (lineoffirstDist + anzCh * 4))):
                    outs += inen[line]
                else:
                    #print(inen[line])
                    ttmp = line if ttmp == 0 else ttmp

                with open('tmp', 'w') as outfile:
                    outfile.write(outs)

                # we must reduce the number of channels by one in case of the
                # unstable channel coincides with the last
                if DistCh == (anzDist - 2):
                    inenLine = inen[line_offset][:4] + '%4d' % (
                        len(asymptCH[0]) + anzCh - 1) + inen[line_offset][8:]
                    repl_line('tmp', line_offset, inenLine)

                subprocess.call('cp tmp INEN', shell=True)

            if ttmp != 0:
                print('BV structure %d unstable for all relative widths.' %
                      int(inen[ttmp + 1][4:]))

        else:
            anzCh += 1
#            var0 = maxVar

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
    phaa = read_phase(phaout=phafile,
                      ch=[1, 1],
                      meth=phasCalcMethod,
                      th_shift='')
    a_aa += '  %.4g' % ([
        -MeVfm * np.tan(phaa[n][2] * np.pi / 180.) /
        np.sqrt(mn['137'] * phaa[n][0]) for n in range(len(phaa))
    ][nMatch2])

channel_thresholds = get_bind_en(n=len(evalChans))
channel_thresholds = ' '.join(['%4.4g' % ee for ee in channel_thresholds])

groundstate_4 = get_h_ev()[0]

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

    try:
        plotapproxwave(infi='OUTPUTSPOLE',
                       oufi='expandedWFKT_%d.pdf' % neps,
                       col=waveToPlot,
                       chan=chans,
                       titl='$\epsilon=[ $%s$ ]$fm$^{-2}$' %
                       (' , '.join(['%.4g' % float(ep) for ep in epsi])),
                       nbrE=energyToPlot)

        plotrelativewave(infi='OUTPUTSPOLE',
                         oufi='relWFKT_%d.pdf' % neps,
                         col=relwaveToPlot,
                         chan=chans,
                         titl='$\epsilon=[ $%s$ ]$fm$^{-2}$' %
                         (' , '.join(['%.4g' % float(ep) for ep in epsi])),
                         nbrE=energyToPlot)
        plotDcoeff(infi='OUTPUTSPOLE',
                   oufi='DcoffHist_%d.pdf' % neps,
                   col=0,
                   chan=chans,
                   titl='$\epsilon=[ $%s$ ]$fm$^{-2}$' %
                   (' , '.join(['%.4g' % float(ep) for ep in epsi])),
                   nbrE=energyToPlot)
    except:
        print("Wave-function plotting failed!")

    subprocess.call('cp OUTPUTSPOLE outps_%d' % neps, shell=True)
    subprocess.call('grep "FUNCTIONAL BERUECKSICHTIGT" OUTPUTSPOLE',
                    shell=True)

    plotphas(oufi='4_ph_%d_%s_%s.pdf' % (neps, lam, lecstring),
             chs=pltChans,
             titl='$\epsilon=[ $%s$ ]$fm$^{-2}$' %
             (' , '.join(['%.4g' % float(ep) for ep in epsi])))

    head_str = '# lambda                       channel   a(ch)     eps                        a(2)     B(4)   B(thresh)\n'
    print(head_str)

    chCounter = 0
    for chToRead in evalChans:
        phdd = read_phase(phaout='PHAOUT',
                          ch=chToRead,
                          meth=phasCalcMethod,
                          th_shift='')
        if ((chToRead == [2, 2]) | (chToRead == [3, 3])) & cib:
            a_dd = [
                appC(phdd[n][2] * np.pi / 180.,
                     np.sqrt(2 * redmass(chDict[str(chToRead)][0],
                                         chDict[str(chToRead)][1]) *
                             phdd[n][0]),
                     redmass(chDict[str(chToRead)][0],
                             chDict[str(chToRead)][1]),
                     q1=1,
                     q2=1) for n in range(len(phdd))
            ]
        else:
            a_dd = [
                -MeVfm * np.tan(phdd[n][2] * np.pi / 180.) /
                np.sqrt(2 * redmass(chDict[str(chToRead)][0],
                                    chDict[str(chToRead)][1]) * phdd[n][0])
                for n in range(len(phdd))
            ]

        chanstr = asyChanLabels[int(chToRead[0]) -
                                1] + '--' + asyChanLabels[int(chToRead[1]) - 1]
        chanstr = '(%d)-(%d)' % (int(
            chDict[str(chToRead)][0]), int(chDict[str(chToRead)][1]))
        try:

            results_bare = '%.3f   %30s   %.4g   %.4g%s   %.4g   %s' % (
                float(lam), chanstr, a_dd[nMatch].real, epsi[0], a_aa,
                groundstate_4, channel_thresholds)
            print(results_bare)
            rafile = 'a_result_%s.dat' % chanstr
            with open(rafile, 'w') as outfile:
                outfile.write(head_str + results_bare)

            #a_of_epsi[asyChanLabels[int(chToRead[0]) - 1]].append(
            #    [epsi, a_dd[0].real, a_dd[-1].real])
            a_of_epsi['%s--%s' % (chToRead[0], chToRead[1])].append(
                [epsi, a_dd[nMatch].real, a_dd[-1].real])
            a_of_Ematch[asyChanLabels[int(chToRead[0]) - 1]].append(
                [np.array(phdd)[:, 0], a_dd])

        except:
            print('no phase shifts in <PHAOUT> for channel:  ', chToRead)
            continue

    neps += 1

os.system('rm *mosaic*.pdf')

os.system('pdfjam --quiet --outfile %s --nup 2x%d 4_ph_*.pdf' %
          ('4bdy_phase-mosaic_%s_%s.pdf' %
           (lam, lecstring), int(np.ceil(epsNBR / 2))))
os.system('pdfjam --quiet --outfile %s --nup 2x%d expandedWFKT_*.pdf' %
          ('4bdy_expandedWFKT-mosaic_%s_%s.pdf' %
           (lam, lecstring), int(np.ceil(epsNBR / 2))))
os.system('pdfjam --quiet --outfile %s --nup 2x%d relWFKT_*.pdf' %
          ('4bdy_relWFKT-mosaic_%s_%s.pdf' %
           (lam, lecstring), int(np.ceil(epsNBR / 2))))
os.system('pdfjam --quiet --outfile %s --nup 2x%d DcoffHist_*.pdf' %
          ('4bdy_Dcoff-mosaic_%s_%s.pdf' %
           (lam, lecstring), int(np.ceil(epsNBR / 2))))

os.system('rm 4_ph_*.pdf')
os.system('rm relWFKT_*.pdf')
os.system('rm expandedWFKT_*.pdf')
os.system('rm DcoffHist_*.pdf')

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

phtp = read_phase(phaout='PHAOUT', ch=[1, 1], meth=phasCalcMethod, th_shift='')
phhen = read_phase(phaout='PHAOUT',
                   ch=[2, 2],
                   meth=phasCalcMethod,
                   th_shift='1-2')
phdd = read_phase(phaout='PHAOUT',
                  ch=[3, 3],
                  meth=phasCalcMethod,
                  th_shift='1-3')
phdqdq = read_phase(phaout='PHAOUT',
                    ch=[4, 4],
                    meth=phasCalcMethod,
                    th_shift='1-3')
phnnpp = read_phase(phaout='PHAOUT',
                    ch=[5, 5],
                    meth=phasCalcMethod,
                    th_shift='1-3')

phtphen = read_phase(phaout='PHAOUT',
                     ch=[1, 2],
                     meth=phasCalcMethod,
                     th_shift='1-2')

phtpdd = read_phase(phaout='PHAOUT',
                    ch=[1, 3],
                    meth=phasCalcMethod,
                    th_shift='1-3')
phhendd = read_phase(phaout='PHAOUT',
                     ch=[2, 3],
                     meth=phasCalcMethod,
                     th_shift='2-3')

# this ordering must match the threshold order, e.g., if B(t)>B(3He)>B(d)>B(dq),
# phtp -> evalChans[0]
# phhen -> evalChans[1]
# phdd -> evalChans[2]
# phdqdq -> evalChans[3]
phtp = read_phase(phaout='PHAOUT',
                  ch=evalChans[0],
                  meth=phasCalcMethod,
                  th_shift='')
phhen = read_phase(phaout='PHAOUT',
                   ch=evalChans[1],
                   meth=phasCalcMethod,
                   th_shift='1-2')
#phdqdq = read_phase(phaout='PHAOUT', ch=evalChans[3], meth=phasCalcMethod, th_shift='1-4')
#phmix = read_phase(phaout='PHAOUT', ch=evalChans[3], meth=phasCalcMethod, th_shift='1-2')

exit()

write_phases(phnnpp,
             filename='nn-pp_phases_%s.dat' % lam,
             append=0,
             comment='',
             mu=mn['137'])
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
             mu=(3. / 4.) * mn['137'])
write_phases(phhen,
             filename='he3-n_phases_%s.dat' % lam,
             append=0,
             comment='',
             mu=(3. / 4.) * mn['137'])

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
        (3. / 2.) * mn['137'] * phtp[n][0]) for n in range(len(phtp))
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