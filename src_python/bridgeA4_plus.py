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

import multiprocessing
from multiprocessing.pool import ThreadPool

home = os.getenv("HOME")

pathbase = home + '/kette_repo/limit_cycles'
sysdir2 = pathbase + '/systems/2'
sysdir3 = pathbase + '/systems/3'
sysdir = pathbase + '/systems/4'
os.chdir(sysdir)

BINBDGpath = pathbase + '/src_nucl/'

parall = -1
anzproc = 6  #int(len(os.sched_getaffinity(0)) / 1)

# numerical stability
nBV = 6
nREL = 8
mindisti = [0.02, 0.02]
width_bnds = [0.001, 1.5, 0.001, 1.5]
minCond = 10**-11

# NN: tnni=10   NN+NNN: tnni=11
tnni = 11

# genetic parameters
anzNewBV = 4
muta_initial = 0.05
anzGen = 50
civ_size = 30
target_pop_size = 20

nnpot = 'nn_pot'
nnnpot = 'nnn_pot'

J0 = 0

lam = 2.00
la = ('%-4.2f' % lam)[:4]
if la in lec_list_def.keys():
    pass
else:
    print('LECs unavailable for chosen cutoff! Available cutoffs:\n',
          lec_list_def.keys())
    exit()

# B2 = 1 MeV and B3 = 8.48 MeV
cloW = lec_list_def[la][0]
cloB = 0.0
d0 = 20 * lec_list_def[la][1]

prep_pot_file_2N(lam=lam, wiC=cloW, baC=0.0, ps2=nnpot)
prep_pot_file_3N(lam=la, d10=d0, ps3=nnnpot)

# convention: bound-state-expanding BVs: (1-8), i.e., 8 states per rw set => nzf0*8
channels = [
    ['000-0', 'nnnnS0t'],  # no DSI
    #['000-0', 'dqdqS0'],  # DSI
]

costr = ''
zop = 31 if tnni == 11 else 14
for nn in range(1, zop):
    cf = 1.0 if ((nn == 1) | (nn == 2) | (nn == 14)) else 0.0
    costr += '%12.7f' % cf if (nn % 7 != 0) else '%12.7f\n' % cf
prepare_einzel4(sysdir, BINBDGpath, channels)
os.chdir(sysdir)

widthSet_relative = w120

strus = fromNNto4(
    zwei_dir=sysdir2,
    vier_dir=sysdir,
    fn=nnpot,
    relw=widthSet_relative,
)

sbas = []
bv = 1
for n in range(len(strus)):
    for m in range(strus[n]):
        sbas += [[bv, [x for x in range(1, 1 + len(widthSet_relative))]]]
        bv += 1

ddCoff = parse_ev_coeffs(mult=1,
                         infil=sysdir2 + '/bndg_out',
                         outf='COEFF',
                         bvnr=1)

ddCoff = np.array(ddCoff).astype(float)
#ddC = ''
#for n in range(len(ddCoff)):
#    for m in range(len(ddCoff) - n):
#        ddC += '%12.4e\n' % (ddCoff[n] * ddCoff[m])
#for n in range(len(ddCoff)):
#    ddC += '%12.4e\n' % (ddCoff[n])

ma = blunt_ev4(cfgs=len(strus) * channels,
               bas=sbas,
               dmaa=[0, 1, 0, 1, 0, 1, 0, 1],
               j1j2sc=[0, 0, 0],
               funcPath=sysdir,
               nzopt=zop,
               frgCoff=ddCoff,
               costring=costr,
               bin_path=BINBDGpath,
               mpipath=MPIRUN,
               potNN='./%s' % nnpot,
               potNNN='./%s' % nnnpot,
               parall=parall,
               anzcores=max(2, min(len(strus), MaxProc)),
               tnnii=tnni,
               jay=J0)

spole_2(nzen=200,
        e0=0.01,
        d0=0.01,
        eps=0.01,
        bet=1.1,
        nzrw=100,
        frr=0.06,
        rhg=8.0,
        rhf=1.0,
        pw=0)
subprocess.run([BINBDGpath + 'S-POLE_PdP.exe'])

chans = [[1, 1]]
ph2 = read_phase(phaout=sysdir2 + '/PHAOUT', ch=[1, 1], meth=1, th_shift='')

for chan in chans:
    ph4 = read_phase(phaout='PHAOUT', ch=chan, meth=1, th_shift='')
    write_phases(ph2,
                 filename='atom-atom_phases.dat',
                 append=0,
                 comment='',
                 mu=0.5 * mn['137'])
    write_phases(ph4,
                 filename='dimer-dimer_phases.dat',
                 append=0,
                 comment='',
                 mu=mn['137'])
    a_ratio = [
        np.tan(ph4[n][2] * np.pi / 180.) /
        (np.sqrt(2) * np.tan(ph2[n][2] * np.pi / 180.))
        for n in range(len(ph4))
    ]

    a_aa = [
        -MeVfm * np.tan(ph2[n][2] * np.pi / 180.) /
        np.sqrt(mn['137'] * ph2[n][1]) for n in range(len(ph2))
    ]
    a_dd = [
        -MeVfm * np.tan(ph4[n][2] * np.pi / 180.) /
        np.sqrt(2 * mn['137'] * ph4[n][1]) for n in range(len(ph4))
    ]

dim = int(np.sqrt(len(ma) * 0.5))
# read Norm and Hamilton matrices
normat = np.reshape(np.array(ma[:dim**2]).astype(float), (dim, dim))
hammat = np.reshape(np.array(ma[dim**2:]).astype(float), (dim, dim))
# diagonalize normalized norm (using "eigh(ermitian)" to speed-up the computation)
ewN, evN = eigh(normat)
ewH, evH = eigh(hammat, normat)
qual, gs, basCond = basQ(ewN, ewH, minCond)
print(
    'd-d structured hamiltonian yields:\n E_0 = %f MeV\ncondition number = %E'
    % (gs, basCond))

outs = ''
for n in range(len(a_ratio)):
    outs += '%12.8f %12.8f %12.8f %12.8f\n' % (ph2[n][0], a_ratio[n], a_aa[n],
                                               a_dd[n])
with open('a_ratio.dat', 'w') as outfile:
    outfile.write(outs)

os.system('gnuplot ' + pathbase + '/src_python/phases.gnu')