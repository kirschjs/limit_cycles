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

newCal = 1

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
channels = [
    #['000-0', 'nnnnS0t'],  # no DSI
    #['000-0', 'dqdqS0'],  # DSI
    ['000-0', 'ddS0'],  # DSI
    ['000-0', 'tp_1s0'],
    ['000-0', 'tp_6s0'],
    ['000-0', 'hen_1s0'],
    ['000-0', 'hen_6s0']
]

costr = ''
zop = 31 if tnni == 11 else 14
for nn in range(1, zop):
    cf = 1.0 if ((nn == 1) | (nn == 2) | (nn == 14)) else 0.0
    costr += '%12.7f' % cf if (nn % 7 != 0) else '%12.7f\n' % cf
prepare_einzel4(sysdir4, BINBDGpath, channels)
os.chdir(sysdir4)

strus_22 = from2to4(
    zwei_inq=sysdir2 + '/INQUA_N_%s' % lam,
    vier_dir=sysdir4,
    fn=nnpot,
    relw=widthSet_relative,
)
zstrus_22 = strus_22
strus_22 = len(strus_22) * channels[0]

strus_31 = sum(
    [dict_3to4[line.strip()] for line in open(sysdir3 + '/obstru_%s' % lam)],
    [])

zstrus_31 = [
    int(line.strip()) for line in open(sysdir3 + '/drei_stru_%s' % lam)
]

zstrus = zstrus_22 + zstrus_31
strus = strus_22 + strus_31

if os.path.isfile(sysdir2 + '/bndg_out_%s' % lam) == False:
    print("2-body bound state unavailable for L = %s" % lam)
    exit()

ddCoff = parse_ev_coeffs(mult=1,
                         infil=sysdir2 + '/bndg_out_%s' % lam,
                         outf='COEFF',
                         bvnr=1)

ddCoff = np.array(ddCoff).astype(float)

subprocess.call('cat %s/inq_3to4_%s >> INQUA_N' % (sysdir3, lam), shell=True)

tCoff = parse_ev_coeffs(mult=0,
                        infil=sysdir3 + '/bndg_out_%s' % lam,
                        outf='COEFF',
                        bvnr=1)

tCoff = np.array(tCoff).astype(float)

cofli = [ddCoff.tolist(), tCoff.tolist()]

sbas = []
bv = 1

for n in range(len(zstrus)):
    for m in range(zstrus[n]):
        sbas += [[bv, [x for x in range(1, 1 + len(widthSet_relative))]]]
        bv += 1

#ddC = ''
#for n in range(len(ddCoff)):
#    for m in range(len(ddCoff) - n):
#        ddC += '%12.4e\n' % (ddCoff[n] * ddCoff[m])
#for n in range(len(ddCoff)):
#    ddC += '%12.4e\n' % (ddCoff[n])

if newCal:
    ma = blunt_ev4(cfgs=strus,
                   bas=sbas,
                   dmaa=[0, 1, 0, 1, 0, 0, 0, 0],
                   j1j2sc=[[2, 2, 0], [1, 1, 0]],
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
                   nchtot=int(len(sbas) - (2 + 1) * len(cofli)))

    dim = int(np.sqrt(len(ma) * 0.5))
    # read Norm and Hamilton matrices
    normat = np.reshape(np.array(ma[:dim**2]).astype(float), (dim, dim))
    hammat = np.reshape(np.array(ma[dim**2:]).astype(float), (dim, dim))
    # diagonalize normalized norm (using "eigh(ermitian)" to speed-up the computation)
    ewN, evN = eigh(normat)
    ewH, evH = eigh(hammat, normat)
    qual, gs, basCond = basQ(ewN, ewH, minCond)
    print(
        '(t-p)+(d-d) structured hamiltonian yields:\n E_0 = %f MeV\ncondition number = %E'
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

    subprocess.run([BINBDGpath + 'S-POLE_PdP.exe'])

chans = [[1, 1], [2, 2], [2, 1]]

if os.path.isfile(sysdir2 + '/phaout_%s' % lam) == False:
    print("2-body phase shifts unavailable for L = %s" % lam)
    exit()

ph2 = read_phase(phaout=sysdir2 + '/phaout_%s' % lam,
                 ch=[1, 1],
                 meth=1,
                 th_shift='')

phtp = read_phase(phaout='PHAOUT', ch=chans[0], meth=1, th_shift='')
phdd = read_phase(phaout='PHAOUT', ch=chans[1], meth=1, th_shift='1-2')
phmix = read_phase(phaout='PHAOUT', ch=chans[2], meth=1, th_shift='1-2')

write_phases(ph2,
             filename='atom-atom_phases.dat',
             append=0,
             comment='',
             mu=0.5 * mn['137'])
write_phases(phdd,
             filename='dimer-dimer_phases.dat',
             append=0,
             comment='',
             mu=mn['137'])
write_phases(phtp,
             filename='trimer-atom_phases.dat',
             append=0,
             comment='',
             mu=mn['137'])
write_phases(phmix,
             filename='tp-dd-mixing_phases.dat',
             append=0,
             comment='',
             mu=mn['137'])

a_aa = [
    -MeVfm * np.tan(ph2[n][2] * np.pi / 180.) / np.sqrt(mn['137'] * ph2[n][1])
    for n in range(len(ph2))
]

a_dd = [
    -MeVfm * np.tan(phdd[n][2] * np.pi / 180.) /
    np.sqrt(2 * mn['137'] * phdd[n][0]) for n in range(len(phdd))
]

a_tp = [
    -MeVfm * np.tan(phtp[n][2] * np.pi / 180.) /
    np.sqrt(1.5 * mn['137'] * phtp[n][0]) for n in range(len(phtp))
]

a_mix = [
    -MeVfm * np.tan(phmix[n][2] * np.pi / 180.) /
    np.sqrt(2 * mn['137'] * phmix[n][0]) for n in range(len(phmix))
]

outs = '#    E_cm-11         a_11      E_ch-22         a_22      E_ch-31         a_31      E_ch-22         a_mix\n'
for n in range(len(a_dd)):
    outs += '%12.8f %12.8f %12.8f %12.8f %12.8f %12.8f %12.8f %12.8f\n' % (
        ph2[n][0], a_aa[n], phdd[n][0], a_dd[n], phtp[n][0], a_tp[n],
        phmix[n][0], a_mix[n])
with open('a_ratio_%s.dat' % lam, 'w') as outfile:
    outfile.write(outs)

os.system('gnuplot ' + pathbase + '/src_python/phases.gnu')