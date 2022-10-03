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

#import bridgeA2
#import bridgeA3_plus

newCal = 0

os.chdir(sysdir4)

# numerical stability
nBV = 6
nREL = 8
mindisti = [0.02, 0.02]
width_bnds = [0.001, 1.5, 0.001, 1.5]
minCond = 10**-11

# genetic parameters
anzNewBV = 4
muta_initial = 0.05
anzGen = 50
civ_size = 30
target_pop_size = 20

J0 = 0

# convention: bound-state-expanding BVs: (1-8), i.e., 8 states per rw set => nzf0*8
J1J2SC = [[2, 2, 0], [0, 0, 0], [1, 1, 0], [1, 1, 0]]
channels = [
    #['000-0', 'nnnnS0t'],  # no DSI
    ['000-0', 'np3s_np3s_S0'],  # DSI
    ['000-0', 'np1s_np1s_S0'],  # DSI
    ['000-0', 'tp_1s0'],
    ['000-0', 'tp_6s0'],
    ['000-0', 'hen_1s0'],
    ['000-0', 'hen_6s0']
]

costr = ''
zop = 31 if tnni == 11 else 14
for nn in range(1, zop):
    cf = 1.0 if ((nn == 1) | (nn == 2) | (nn == 14)) else 0.0
    #cf = 1.0 if ((nn == 2) | (nn == 14)) else 0.0
    costr += '%12.7f' % cf if (nn % 7 != 0) else '%12.7f\n' % cf

einzel4 = False
if einzel4:
    prepare_einzel4(sysdir4, BINBDGpath, channels)

os.chdir(sysdir4)

twodirs = []
if 'np3s' in [ch[1].split('_')[0] for ch in channels]:
    twodirs.append(sysdir2np3s)
if 'np1s' in [ch[1].split('_')[0] for ch in channels]:
    twodirs.append(sysdir2np1s)

fragment_energies = []
cofli = []
strus = []
zstrus = []
qua_str = []
sbas = []
ph2d = []
n2 = 0

subprocess.call('rm -rf INQUA_N', shell=True)

for sysdir2 in twodirs:

    if os.path.isfile(sysdir2 + '/bndg_out_%s' % lam) == False:
        print("2-body bound state unavailable for L = %s" % lam)
        exit()
    ths = get_h_ev(n=1, ifi=sysdir2 + '/bndg_out_%s' % lam)
    fragment_energies.append(ths[0])

    zstrus_tmp, outs = from2to4(zwei_inq=sysdir2 + '/INQUA_N_%s' % lam,
                                vier_dir=sysdir4,
                                fn=nnpot,
                                relw=widthSet_relative,
                                app='True')
    zstrus.append(zstrus_tmp)
    qua_str.append(outs)

    strus.append(len(zstrus[-1]) * channels[n2])
    n2 += 1

    sbas.append(get_bsv_rw_idx(inen=sysdir2 + '/INEN_BDG', offset=4, int4=0))

    ddCoff = parse_ev_coeffs(mult=1,
                             infil=sysdir2 + '/bndg_out_%s' % lam,
                             outf='COEFF',
                             bvnr=1)
    ddCoff = np.array(ddCoff).astype(float)
    cofli.append(ddCoff.tolist())

    ph2d.append(
        read_phase(phaout=sysdir2 + '/phaout_%s' % (lam),
                   ch=[1, 1],
                   meth=1,
                   th_shift=''))

# the order matters as we conventionally include physical channels
# in ascending threshold energy. E.g. dd then he3n then tp;
threedirs = []
if 'tp' in [ch[1].split('_')[0] for ch in channels]:
    threedirs.append(sysdir3t)
if 'hen' in [ch[1].split('_')[0] for ch in channels]:
    threedirs.append(sysdir3he)

for sysdir3 in threedirs:

    sbas.append(get_bsv_rw_idx(inen=sysdir3 + '/INEN', offset=7, int4=1))

    strus.append(
        sum([
            dict_3to4[line.strip()]
            for line in open(sysdir3 + '/obstru_%s' % lam)
        ], []))
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
outs = ' 10  8  9  3 00  0  0  0\n%s\n' % nnpot
for qua_part in qua_str:
    outs += qua_part
with open('INQUA_N', 'w') as outfile:
    outfile.write(outs)

sb = []
bv = 1
for nbv in range(len(sbas)):
    sb.append([bv, np.arange(1, 1 + len(sbas[nbv][1])).tolist()])
    bv += 1

varspacedim = sum([len(rset[1]) for rset in sbas])

if newCal:
    ma = blunt_ev4(cfgs=strus,
                   bas=sb,
                   dmaa=[0, 1, 0, 1, 0, 0, 0, 0],
                   j1j2sc=J1J2SC,
                   funcPath=sysdir4,
                   nzopt=zop,
                   frgCoff=cofli,
                   costring=costr,
                   bin_path=BINBDGpath,
                   mpipath=MPIRUN,
                   potNN='%s' % nnpot,
                   potNNN='%s' % nnnpot,
                   parall=parall,
                   anzcores=max(2, min(len(strus), MaxProc)),
                   tnnii=tnni,
                   jay=J0,
                   nchtot=int(varspacedim - (2 + 1) * len(cofli)))

    dim = int(np.sqrt(len(ma) * 0.5))
    # read Norm and Hamilton matrices
    normat = np.reshape(np.array(ma[:dim**2]).astype(float), (dim, dim))
    hammat = np.reshape(np.array(ma[dim**2:]).astype(float), (dim, dim))
    # diagonalize normalized norm (using "eigh(ermitian)" to speed-up the computation)
    ewN, evN = eigh(normat)
    ewH, evH = eigh(hammat, normat)
    qual, gs, basCond = basQ(ewN, ewH, minCond)
    print(
        '(d-d)+(dq-dq)+(t-p)+(he3-n) structured hamiltonian yields:\n E_0 = %f MeV\ncondition number = %E'
        % (gs, basCond))

    spole_2(nzen=nzEN,
            e0=E0,
            d0=D0,
            eps=Eps,
            bet=Bet,
            nzrw=100,
            frr=0.06,
            rhg=8.0,
            rhf=1.0,
            pw=0)

    # if S-POLE terminates with a ``STOP 7'' message,
    # the channel ordering might not be in the correct, i.e.,
    # ascending-in-energy order; at present, this order needs
    # to be entered manually, according to the B(d) and B(dq)
    # results obtained in the employed subspaces
    # => the order in the <phys_chan> argument of ``inen_str_4'' (called in PSI_parallel_M)

    subprocess.run([BINBDGpath + 'S-POLE_PdP.exe'])

chans = [[1, 1], [2, 2], [3, 3], [4, 4]]

if os.path.isfile(sysdir2 + '/phaout_%s' % lam) == False:
    print("2-body phase shifts unavailable for L = %s" % lam)
    exit()

# this ordering must match the threshold order, e.g., if B(t)>B(3He)>B(d)>B(dq),
# phtp -> chans[0]
# phhen -> chans[1]
# phdd -> chans[2]
# phdqdq -> chans[3]
phtp = read_phase(phaout='PHAOUT', ch=chans[0], meth=1, th_shift='')
phhen = read_phase(phaout='PHAOUT', ch=chans[1], meth=1, th_shift='1-2')
phdd = read_phase(phaout='PHAOUT', ch=chans[2], meth=1, th_shift='1-3')
phdqdq = read_phase(phaout='PHAOUT', ch=chans[3], meth=1, th_shift='1-4')
#phmix = read_phase(phaout='PHAOUT', ch=chans[3], meth=1, th_shift='1-2')

write_phases(ph2d[0],
             filename='np3s_phases.dat',
             append=0,
             comment='',
             mu=0.5 * mn['137'])
write_phases(ph2d[1],
             filename='np1s_phases.dat',
             append=0,
             comment='',
             mu=0.5 * mn['137'])
write_phases(phdd,
             filename='d-d_phases.dat',
             append=0,
             comment='',
             mu=mn['137'])
write_phases(phdqdq,
             filename='dq-dq_phases.dat',
             append=0,
             comment='',
             mu=mn['137'])
write_phases(phtp,
             filename='t-p_phases.dat',
             append=0,
             comment='',
             mu=(2. / 3.) * mn['137'])
write_phases(phhen,
             filename='he3-n_phases.dat',
             append=0,
             comment='',
             mu=(2. / 3.) * mn['137'])

a_np3s = [
    #appC(ph2[n][2] * np.pi / 180.,
    #     np.sqrt(mn['137'] * ph2[n][0]),
    #     mn['137'] / 2.,
    #     q1=1,
    #     q2=1)
    anp(ph2d[0][n][2] * np.pi / 180., np.sqrt(mn['137'] * ph2d[0][n][0]))
    for n in range(len(ph2d[0]))
]
a_np1s = [
    #appC(ph2[n][2] * np.pi / 180.,
    #     np.sqrt(mn['137'] * ph2[n][0]),
    #     mn['137'] / 2.,
    #     q1=1,
    #     q2=1)
    anp(ph2d[1][n][2] * np.pi / 180., np.sqrt(mn['137'] * ph2d[1][n][0]))
    for n in range(len(ph2d[1]))
]
a_dd = [
    appC(phdd[n][2] * np.pi / 180.,
         np.sqrt(2 * mn['137'] * phdd[n][0]),
         mn['137'],
         q1=1,
         q2=1)
    #-MeVfm * np.tan(phdd[n][2] * np.pi / 180.) /    np.sqrt(2 * mn['137'] * phdd[n][0])
    for n in range(len(phdd))
]
a_dqdq = [
    appC(phdqdq[n][2] * np.pi / 180.,
         np.sqrt(2 * mn['137'] * phdqdq[n][0]),
         mn['137'],
         q1=1,
         q2=1)
    #-MeVfm * np.tan(phdd[n][2] * np.pi / 180.) /    np.sqrt(2 * mn['137'] * phdd[n][0])
    for n in range(len(phdqdq))
]
a_tp = [
    appC(phtp[n][2] * np.pi / 180.,
         np.sqrt((3. / 2.) * mn['137'] * phtp[n][0]), (3. / 4.) * mn['137'],
         q1=1,
         q2=1)
    #-MeVfm * np.tan(phtp[n][2] * np.pi / 180.) / np.sqrt( (2. / 3.) * mn['137'] * phtp[n][0])
    for n in range(len(phtp))
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

with open('a_ratio_%s.dat' % lam, 'w') as outfile:
    outfile.write(outs)

os.system('gnuplot ' + pathbase + '/src_python/phases_np.gnu')

#subprocess.call('rm -rf TQUAOUT.*', shell=True)
#subprocess.call('rm -rf TDQUAOUT.*', shell=True)
subprocess.call('rm -rf DMOUT.*', shell=True)
subprocess.call('rm -rf DRDMOUT.*', shell=True)
subprocess.call('rm -rf matout_*.*', shell=True)