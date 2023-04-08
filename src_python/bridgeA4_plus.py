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
einzel4 = 1
findstablebas = 0
newCal = 1

if newCal == 1:
    import bridgeA2_plus
    import bridgeA3_plus

J0 = 0

# convention: bound-state-expanding BVs: (1-8), i.e., 8 states per rw set => nzf0*8
J1J2SC = []
channels = [
    #[['000-0'], ['nn1s_nn1s_S0'], [0, 0, 0]],  # no DSI
    [['000-0'], ['pp1s_nn1s_S0'], [0, 0, 0]],  # DSI
    [['000-0'], ['np1s_np1s_S0'], [0, 0, 0]],  # DSI
    [['000-0'], ['np3s_np3s_S0'], [2, 2, 0]],  # DSI
    [['000-0'], ['tp_1s0', 'tp_6s0'], [1, 1, 0]],
    [['000-0'], ['hen_1s0', 'hen_6s0'], [1, 1, 0]],
]

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
    prepare_einzel4(sysdir4, BINBDGpath, channels)

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

for chan in channels:

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

    if sysdir21 == sysdir22:
        zstrus_tmp, outs = from2to4(zwei_inq=sysdir21 + '/INQUA_N_%s' % lam,
                                    vier_dir=sysdir4,
                                    fn=nnpotstring,
                                    relw=widthSet_relative,
                                    app='True')
    else:
        zstrus_tmp, outs = from22to4(zwei_inq_1=sysdir21 + '/INQUA_N_%s' % lam,
                                     zwei_inq_2=sysdir22 + '/INQUA_N_%s' % lam,
                                     vier_dir=sysdir4,
                                     fn=nnpotstring,
                                     relw=widthSet_relative,
                                     app='True')

    zstrus.append(zstrus_tmp)
    qua_str.append(outs)

    strus.append(len(zstrus[-1]) * [chan[:2]])
    J1J2SC.append(chan[2])

    sbas.append(get_bsv_rw_idx(inen=sysdir21 + '/INEN_BDG', offset=4, int4=0))
    sbas.append(get_bsv_rw_idx(inen=sysdir22 + '/INEN_BDG', offset=4, int4=0))

    if sysdir21 == sysdir22:
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
    print(sysdir21, '\n', sysdir22)

# the order matters as we conventionally include physical channels
# in ascending threshold energy. E.g. dd then he3n then tp;
threedirs = []
for ch in channels_3:
    threedirs.append(sysdir3base + '/' + ch)

for sysdir3 in threedirs:

    for line in open(sysdir3 + '/obstru_%s' % lam):
        gogo = True
        for ch in channels:
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

    fragment_energies.append([float(ee) for ee in open(sysdir3 + '/E0')][0])
    threeCoff = parse_ev_coeffs(mult=0,
                                infil=sysdir3 + '/bndg_out_%s' % lam,
                                outf='COEFF',
                                bvnr=1)

    threeCoff = np.array(threeCoff).astype(float)
    cofli.append(threeCoff.tolist())

    qua_str.append(''.join(
        [line for line in open('%s/inq_3to4_%s' % (sysdir3, lam))]))
    #subprocess.call('cat %s/inq_3to4_%s >> INQUA_N' % (sysdir3, lam),
    #                shell=True)

idx = np.array(fragment_energies).argsort()[::-1]

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

anzch = int(np.max([1, len(sum(cofli, [])) - 5 * len(cofli)]))

anzch = 99

print(
    '\n Commencing 4-body calculation with %d channels (physical + distortion).'
    % anzch)
print('>>> working directory: ', sysdir4)

for nbv in range(1, varspacedim):
    relws = [
        [1, 0] for n in range(int(0.5 * len(widthSet_relative)))
    ] if nbv % 2 == 0 else [[0, 1]
                            for n in range(int(0.5 * len(widthSet_relative)))]
    sb.append([nbv, sum(relws, [])])

if newCal:

    ma = blunt_ev4(cfgs=strus,
                   bas=sb,
                   dmaa=[1, 1, 1, 1, 1, 1, 1, 0],
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
        pw=0)

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
            len(channels) + anzCh) + inen[line_offset][8:]
        #print(inenLine)
        repl_line('INEN', line_offset, inenLine)
        subprocess.run([BINBDGpath + spectralEXE_mpi])
        lastline = [ll for ll in open('OUTPUT')][-1]
        tmp = get_h_ev()
        print('E_0(#Dch:%d)=%4.4f MeV' % (len(channels) + anzCh, tmp))
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

subprocess.run([BINBDGpath + spectralEXE_mpi])

chToRead = [3, 3]

redmass = (4. / 4.) * mn['137']
a_of_epsi = []

for epsi in np.linspace(eps0, eps1, 20):
    spole_2(nzen=nzEN,
            e0=E0,
            d0=D0,
            eps=epsi,
            bet=Bet,
            nzrw=anzStuez,
            frr=StuezAbs,
            rhg=rgh,
            rhf=StuezBrei,
            pw=0)
    subprocess.run([BINBDGpath + smatrixEXE_multichann])
    phdd = read_phase(phaout='PHAOUT', ch=chToRead, meth=1, th_shift='')

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
    a_of_epsi.append([epsi, a_dd[0].real, a_dd[-1].real])

spole_2(nzen=nzEN,
        e0=E0,
        d0=D0,
        eps=epsM,
        bet=Bet,
        nzrw=anzStuez,
        frr=StuezAbs,
        rhg=rgh,
        rhf=StuezBrei,
        pw=0)

subprocess.run([BINBDGpath + smatrixEXE_multichann])

phdd = read_phase(phaout='PHAOUT', ch=chToRead, meth=1, th_shift='')

redmass = (4. / 4.) * mn['137']
if ((chToRead == [2, 2]) | (chToRead == [3, 3])) & cib:
    print('charged-fragment channel:\n')
    a_dd = [
        appC(phdd[n][2] * np.pi / 180.,
             np.sqrt(2 * redmass * phdd[n][0]),
             redmass,
             q1=1,
             q2=1) for n in range(len(phdd))
    ]
else:
    print('uncharged-fragment channel:\n')
    a_dd = [
        -MeVfm * np.tan(phdd[n][2] * np.pi / 180.) /
        np.sqrt(2 * redmass * phdd[n][0]) for n in range(len(phdd))
    ]
print(
    'l = %s fm^-1\n scattering lengths (lower/upper end of energy matching interval):\na_dd(E_min) = %4.4f fm   a_dd(E_max) = %4.4f fm'
    % (lam, a_dd[0].real, a_dd[-1].real))

plotphas(oufi='4_body_phases.pdf', diag=True)

plotarray(infix=[float(a[0]) for a in a_of_epsi],
          infiy=[n[1] for n in a_of_epsi],
          ylab='$a_{dd}$ [fm]',
          xlab='$\epsilon$ [fm$^{-1}$]',
          outfi='a_of_epsilon.pdf',
          plotrange='med')

exit()

plotarray([float(a.real) for a in a_dd],
          [phdd[n][0] for n in range(len(phdd))],
          ylab='$a_{dd}$ [fm]',
          xlab='$E_0$ [MeV]',
          outfi='a_dimer-dimer.pdf')

phtp = read_phase(phaout='PHAOUT', ch=[1, 1], meth=1, th_shift='')

phhen = read_phase(phaout='PHAOUT', ch=[2, 2], meth=1, th_shift='1-2')
phtphen = read_phase(phaout='PHAOUT', ch=[1, 2], meth=1, th_shift='1-2')

phdd = read_phase(phaout='PHAOUT', ch=[3, 3], meth=1, th_shift='1-3')
phtpdd = read_phase(phaout='PHAOUT', ch=[1, 3], meth=1, th_shift='1-3')
phhendd = read_phase(phaout='PHAOUT', ch=[2, 3], meth=1, th_shift='2-3')

plotarray2(outfi='tmp.pdf',
           infix=[[phdd[n][0] for n in range(len(phdd))],
                  [float(a[0]) for a in a_of_epsi],
                  [[phtp[n][0] for n in range(len(phtp))],
                   [phhen[n][0] for n in range(len(phhen))],
                   [phdd[n][1] for n in range(len(phdd))],
                   [phtpdd[n][1] for n in range(len(phtpdd))],
                   [phhendd[n][1] for n in range(len(phhendd))]]],
           infiy=[
               [float(a.real) for a in a_dd],
               [n[1] for n in a_of_epsi],
               [[phtp[n][2] for n in range(len(phtp))],
                [phhen[n][2] for n in range(len(phhen))],
                [phdd[n][2] for n in range(len(phdd))],
                [phtpdd[n][2] for n in range(len(phtpdd))],
                [phhendd[n][2] for n in range(len(phhendd))]],
           ],
           title=[
               '$a_{dd}$ dependence on $k_0$',
               '$a_{dd}$ dependence on $\epsilon$', '2-fragment 4-body phases'
           ],
           xlab=['$E_{cm}$ [MeV]', '$\epsilon$ [fm$^{-2}$]', '$E_{cm}$ [MeV]'],
           ylab=['$a_{dd}$ [fm]', '$a_{dd}$ [fm]', '$\delta_{f_1-f_2}$ [Deg]'],
           leg=[[], [], [
               't-p',
               '${}^3$He-n',
               '$d-d$',
               '$tp-d$',
               '$hen-d$',
           ]],
           plotrange=['', 'max', ''])

for channel in channels_2:
    J0 = channels_2[channel][1]

    sysdir2 = sysdir2base + '/' + channel
    phafile = sysdir2 + '/phaout_%s' % lam
    if os.path.isfile(phafile) == False:
        print("2-body phase shifts unavailable for L = %s" % lam)
        exit()
    phaa = read_phase(phaout=phafile, ch=[1, 1], meth=1, th_shift='')
    a_aa = [
        -MeVfm * np.tan(phaa[n][2] * np.pi / 180.) /
        np.sqrt(mn['137'] * phaa[n][0]) for n in range(len(phaa))
    ]
    print(
        'a_aa(E_min) = %4.4f fm   a_aa(E_max) = %4.4f fm\na_dd/a_aa(E_min) = %4.4f   a_dd/a_aa(E_max) = %4.4f'
        %
        (a_aa[0], a_aa[-1], a_dd[0].real / a_aa[0], a_dd[-1].real / a_aa[-1]))

exit()

chans = [[1, 1], [2, 2], [3, 3], [4, 4]]

# this ordering must match the threshold order, e.g., if B(t)>B(3He)>B(d)>B(dq),
# phtp -> chans[0]
# phhen -> chans[1]
# phdd -> chans[2]
# phdqdq -> chans[3]
phtp = read_phase(phaout='PHAOUT', ch=chans[0], meth=1, th_shift='')
phhen = read_phase(phaout='PHAOUT', ch=chans[1], meth=1, th_shift='1-2')
phdqdq = read_phase(phaout='PHAOUT', ch=chans[3], meth=1, th_shift='1-4')
#phmix = read_phase(phaout='PHAOUT', ch=chans[3], meth=1, th_shift='1-2')

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