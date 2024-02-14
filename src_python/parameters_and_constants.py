import os
import numpy as np
import more_itertools
from plot_array import *

# -- LEC lists
# see lec_sets.py

# Gaussian width parameters optimized to scattering calculations with
# V18/UIX interaction potentials

w120 = [
    129.5665, 51.3467, 29.47287, 13.42339, 8.2144556, 4.447413, 2.939,
    1.6901745, 1.185236, 0.84300, 0.50011, 0.257369, 0.13852, 0.071429,
    0.038519, 0.018573, 0.0097261, 0.00561943, 0.002765, 0.00101
]

wNew = [
    19.5665, 11.3467, 9.47287, 3.42339, 2.2144556, 1.447413, 0.939, 0.6901745,
    0.185236, 0.084300, 0.050011, 0.0257369, 0.013852, 0.0071429, 0.0038519,
    0.0018573, 0.00097261, 0.000561943, 0.0002765, 0.000101
]

w12 = [
    12.95665, 5.13467, 2.947287, 1.342339, .82144556, .4447413, 2.939,
    1.6901745, 1.185236, 0.84300, 0.50011, 0.257369, 0.13852, 0.071429,
    0.038519, 0.018573, 0.0097261, 0.00561943, 0.002765, 0.00101
]

# nuclear masses for various pion masses

mn = {
    '137': 938.91852,
    '300': 1053.0,
    '450': 1226.0,
    '510': 1320.0,
    '805': 1634.0
}

two_body_channels = {
    # r7 c2:    J              S  L              S_c
    'np1s': 1,  #  1   :   0  0  1S0         0
    'np3s': 2,  #  2   :   1  0  3S1         2
    'nn1s': 3,  #  4   :   0  0  1S0         0
    'nn3p': 4,  #  5   :   1  1  3P0,3P1,3P2 2
    'nn1d': 5,  #  6   :   0  2  1D2         2
    'nn3f': 6,  #  7   :   0  1  3F2,3F3,3F4 0
    'np1p': 7,  #  8   :   1  1  1P1         2
    'np3p': 8,  #  9   :   1  2  3P0,3P1,3P2 2
    'np3d': 9,  #  9   :   1  2  3D1,3D2,3D3 2
    'pp1s': 10,  #  9   :   1  2  1S0         2
    'pp3p': 11,  #  9   :   1  2  3P0,3P1,3P2 2
    'pp1d': 12,  #  9   :   1  2  1D0         2
    'pp3f': 13,  #  9   :   1  2  3F2,3F3,3F4 2
}

dict_3to4 = {
    '123': [['000-0'], ['tp_123-4']],
    't_no1': [['000-0'], ['tp_1s0']],
    't_no6': [['000-0'], ['tp_6s0']],
    'he_no1': [['000-0'], ['hen_1s0']],
    'he_no6': [['000-0'], ['hen_6s0']],
}

dict_4to5 = {
    'tp_1s0': [['000-0-0'], ['tpn_1s0h']],
    'tp_6s0': [['000-0-0'], ['tpn_6s0h']],
    'hen_1s0': [['000-0-0'], ['henn_1s0h']],
    'hen_6s0': [['000-0-0'], ['henn_6s0h']],
}

home = os.getcwd()

pathbase = home + '/..'
BINBDGpath = pathbase + '/src_nucl/'

# NN: tnni=10   NN+NNN: tnni=11
tnni = 11
tnifac = 1.65
twofac = 1.0
parall = -1

# limits the number of parallel processes in a single process pool
# if running on my laptop, I need to set this number in order to avoid
# too many files too be opened simulataneously
maxParLen = 120

cib = 0  # if set, EFTnoPi with charge independence broken by Coulomb and an acompanying
# contact-term correction is employed (leading order)

lam = 6.00  # 4,6,8,10 (for presentation)
b3 = 13.0
la = ('%-4.2f' % lam)[:4]
tb = ('%-4.2f' % b3)[:4]

lecstring = 'B2-05_B3-' + tb
"""
B(2) = 0.43(1) MeV
B(3) -- 0.46 (10)  0.66 (30)  0.71 (34)  0.84     0.9      1.1     1.9       8.4
D    -- 1730.4873 672.4964 572.2156 317.1505 238.5372 -4.1843 -532.5492 -1703.9865
"""

# LEC dictionary [cutoff][B(3)] [2-body LEC, 3-body LEC] for B(2)=0.5MeV
lec_set = {
    '6.00': {
        '10.0': [-702.16, -787.8583],
        '13.0': [-702.16, -845.1583],
        '20.0': [-702.16, -946.4991],
    },
    '4.00': {
        '13.0': [-473.27, -601.7702],
    },
}

SU4 = True

#for mm in lec_lists.keys():
#    print(mm)

# list of suffices:
# _m1       : hbar/(2m) = 1  -> modified kinetic-energy operators (available for _v18-uix operator set, only, at present)
# _v18-uix  :
# _eft      :
# _eft-cib  :
# if there is a 'b' attached it refers to a symmetrization of the total wave function

bin_suffix = '_eft-cib' if cib else '_v18-uix'

if bin_suffix == '_v18-uix':
    inenOffset = 7
    nOperators = 31
    withCoul = False
elif bin_suffix == '_eft-cib':
    inenOffset = 6
    nOperators = 28
    withCoul = True
else:
    print('unrecognized operator structure.')
    exit()

# 2-body iso-spin and spin matrix elements
# <<< INOB
NNspinEXE = 'KOBER%s.exe' % bin_suffix

# 3-body iso-spin and spin matrix elements
# <<< INOB
NNNspinEXE = 'DROBER%s.exe' % bin_suffix

# 2-body orbital-angular-momentum structure of the spatial matrix elements
# <<< INLUCN
NNorbitalEXE = 'LUDW%s.exe' % bin_suffix

# 3-body orbital-angular-momentum structure of the spatial matrix elements
# <<< INLU
NNNorbitalEXE = 'DRLUD%s.exe' % bin_suffix

# calculation of Hamilton and Norm matrix combining (iso)spin, orbital, and radial wave-function components
# for 2-body operators
# <<< INQUA_N (2nd line = 2N potential)
NNhamilEXE_serial = 'QUAFL_N%s.exe' % bin_suffix
NNhamilEXE_pool = 'QUAFL_N%s_pop.exe' % bin_suffix
NNhamilEXE_mpi = 'V18_PAR/mpi_quaf_n%s' % bin_suffix
NNcollect_mpi = 'V18_PAR/sammel%s' % bin_suffix

# calculation of Hamilton and Norm matrix combining (iso)spin, orbital, and radial wave-function components
# for 3-body operators
# <<< INQUA_N (2nd line = 3N potential, otherwise identical to 2N INQUA_N)
NNNhamilEXE_serial = 'DRQUA_N%s.exe' % bin_suffix
NNNhamilEXE_pool = 'DRQUA_N%s_pop.exe' % bin_suffix
NNNhamilEXE_mpi = 'UIX_PAR/mpi_drqua_n%s' % bin_suffix
NNNcollect_mpi = 'UIX_PAR/drsammel%s' % bin_suffix

# diagonalization of the Norm- and Hamilton matrices
spectralEXE_serial = 'DR2END%s.exe' % bin_suffix
spectralEXE_mpi = 'TDR2END%s.exe' % bin_suffix
spectralEXE_serial_pool = 'DR2END%s_pop.exe' % bin_suffix
spectralEXE_mpi_pool = 'TDR2END%s_pop.exe' % bin_suffix

# calculation of scattering matrices/amplitudes as solutions to the Kohn variational functional
smatrixEXE = 'S-POLE_zget.exe'  #'S-POLE_PdP.exe'
smatrixEXE_multichann = 'S-POLE_zget.exe'  #'S-POLE_PdP.exe'  #

pas = False
if la in lec_set.keys():
    if tb in lec_set[la].keys():
        pas = True
if pas == False:
    print('LECs unavailable for chosen cutoff! Available cutoffs:\n',
          lec_set.keys())
    exit()

# deg_channs: if true, the spatial wave functions are identical for t and he3, and d,dq,nn,pp
#                      only the first 2- and 3-body channel - canonical choice t and d - are optimized
deg_channs = 1
id_chan = 0

channels_2 = {
    # L    J
    'np3s': ['0', '1'],
    #'np1s': ['0', '0'],
    #'nn1s': ['0', '0'],
    #'pp1s': ['0', '0'],
}

channels_3 = {
    '123': [['000', ['123', '123']]],
    #'t': [['000', ['t_no1', 't_no6']]],
    #'he': [['000', ['he_no1', 'he_no6']]],
    #'boltz': ['000', ['dist_3', 'dist_3']],
}

channels_4 = {
    'alpha': [
        ['000-0'],
        [
            'np3s_np3s_12-34', 'tp_123-4'
            #'dist_4',
            #'tp_1s0',
            #'tp_6s0',
            #'hen_1s0',
            #'hen_6s0',
            #'np3s_np3s_S0',
            #'np1s_np1s_S0'
        ],
        [[2, 2, 0], [1, 1, 0]]
    ],
}

channels_4_scatt = [
    #[['000-0'], ['nn1s_nn1s_S0'], [0, 0, 0]],  # DSI
    #[['000-0'], ['np1s_np1s_S0'], [0, 0, 0]],  # DSI
    #[['000-0'], ['np3s_np3s_S0'], [2, 2, 0]],  # DSI
    #[['000-0'], ['hen_1s0', 'hen_6s0'], [1, 1, 0]],
    #[['000-0'], ['tp_1s0', 'tp_6s0'], [1, 1, 0]],
    [['000-0'], ['np3s_np3s_12-34'], [2, 2, 0]],
    [['000-0'], ['tp_123-4'], [1, 1, 0]],
]

sysdir2base = pathbase + '/systems/2_%s/%s' % (lecstring, la)
sysdir3base = pathbase + '/systems/3_%s/%s' % (lecstring, la)
sysdir4 = pathbase + '/systems/4_%s/%s' % (lecstring, la)
sysdir5 = pathbase + '/systems/5-dist_%s/%s' % (lecstring, la)

nnpotstring = 'nn_pot'
nnnpotstring = 'nnn_pot'

nnpot = sysdir2base + '/' + nnpotstring
nnnpot = sysdir2base + '/' + nnnpotstring

if len(lec_set[la][tb]) >= 4:
    cloW = 0.5 * (lec_set[la][tb][0] + lec_set[la][tb][1])
    cloB = 0.5 * (lec_set[la][tb][0] - lec_set[la][tb][1])
    d0 = lec_set[la][tb][2]
    cpp = lec_set[la][tb][3]
elif len(lec_set[la][tb]) == 3:
    cloW = 0.5 * (lec_set[la][tb][0] + lec_set[la][tb][1])
    cloB = 0.5 * (lec_set[la][tb][0] - lec_set[la][tb][1])
    d0 = lec_set[la][tb][2]
elif len(lec_set[la][tb]) == 2:
    cloW = lec_set[la][tb][0]
    cloB = 0.0
    d0 = lec_set[la][tb][1]

evWindow = [-211.5, -1.70]
nbrStatesOpti2 = 1
nbrStatesOpti3 = [-3]
nbrStatesOpti4 = [-4]

eDict = {
    #    [#energies, E0, dE, [3bdy GS, 3bdy ES1, 3bdy ES2, ...]]
    '10.0': [100, 0.0, 0.03, [0, 1], [[1, 1], [2, 2]]],
    '13.0': [100, 0.0, 0.2, [0, 1, 1], [[1, 1], [2, 2], [3, 3]]],
}

# include the n-th 3-body bounstate of the 3-body spectrum as asymptotic fragments
# in the 3-1 partition of the 4-body calculation; e.g. [0,1] includes an asymptotic
# 4-body channel where 3 atoms are bound in the 1st excited state
nbr_of_threebody_boundstates = eDict[tb][3]

nzEN = eDict[tb][0]
E0 = eDict[tb][1]
D0 = eDict[tb][2]

epL = 0.0001
epU = 0.2
eps0 = [epL * 1.0, epL, epL, epL, epL]
eps1 = [epU * 1.0, epU, epU, epU, epU]
epsM = (np.array(eps1) + np.array(eps0)) / 2
epsNBR = 40

phasCalcMethod = 1
# parameters for the expansion of the fragment-relative function
# (i.e., both fragments charged: Coulomb function, else sperical Bessel)
# in Gaussians
SPOLE_adaptweightUP = 0.15
SPOLE_adaptweightLOW = 0.0
SPOLE_adaptweightL = 0.5
SPOLE_GEW = 0.8  # smaller values decrease the maximal radius up to which values enter the fit
SPOLE_QD = 1.0  # this shifts the interval smaller values try to optimize the behavior closer to zero
SPOLE_QS = 3.2

beta0 = 2.1
Bet = [beta0, beta0, beta0, beta0, beta0]
rgh = 8.0
anzStuez = 400
StuezAbs = 1.250
StuezBrei = 1.50

MeVfm = 197.3161329

# generate Gaussian-width sets for each physical channel
#widthSet_relative_geom = [
#    np.abs(
#        np.geomspace(start=19.5 + (np.random.random() - 0.5),
#                     stop=0.001 * np.max([np.random.random(), 0.1]),
#                     num=38,
#                     endpoint=True,
#                     dtype=None)) for nn in range(1, 1 + len(channels_4_scatt))
#]

# number of relative widths used for the refinement of the 4-body state
# in the interaction region (see bridgeA4_opt.py)
anzRelw4opt = 10

# number of Gaussian basis functions/widths used to expand the fragment-relative wave function
anzRelw = 20  # 10, 12, 14, 20, ....
maxRelW = 10.1
widthSet_relative = [
    np.append(
        np.sort(
            np.abs(
                np.concatenate([
                    np.array([
                        ww
                        for ww in np.logspace(-3.2 + 0.0 * np.random.random(),
                                              1.2 + 0.0 * np.random.random(),
                                              num=int(anzRelw / 2),
                                              endpoint=True,
                                              dtype=None)[::-1] if ww < maxRelW
                    ]),
                    np.array([
                        ww
                        for ww in np.logspace(-1.2 + 0.0 * np.random.random(),
                                              1.1 + 0.0 * np.random.random(),
                                              num=int(anzRelw / 2),
                                              endpoint=True,
                                              dtype=None)[::-1] if ww < maxRelW
                    ])
                ])))[::-1], []) for nn in range(1, 1 + len(channels_4_scatt))
]

#np.linspace(start=11.5, stop=0.005, num=30, endpoint=True, dtype=None))

eps = np.finfo(float).eps