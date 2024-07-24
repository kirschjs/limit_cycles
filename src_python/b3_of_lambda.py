import subprocess
import os, fnmatch, copy, struct
import numpy as np
from scipy.linalg import eigh

from rrgm_functions import *
from genetic_width_growth import *
from parameters_and_constants import *

os.chdir('/home/johannesk/kette_repo/limit_cycles/systems/3he')

for lam in np.linspace(-130, -296, 152):

    lec = lam
    prep_pot_file_2N(lam=2.0, wiC=lec, baC=0.0, ps2='nnpot')
    subprocess.call(BINBDGpath + 'QUAFL_N_pop.exe %s %s %s' %
                    ('INQUA_N', 'output_qua', 'qua2end'),
                    shell=True)
    subprocess.call(BINBDGpath + 'DR2END_AK_pop.exe %s %s %s %s %s' %
                    ('qua2end', 'drquaf_to_end', 'INEN_2', 'output_end',
                     'matout_%s' % str(lam)),
                    shell=True)
    NormHam = np.core.records.fromfile('matout_%s' % str(lam),
                                       formats='f8',
                                       offset=4)

    dim = int(np.sqrt(len(NormHam) * 0.5))

    ## read Norm and Hamilton matrices
    normat = np.reshape(np.array(NormHam[:dim**2]).astype(float), (dim, dim))
    hammat = np.reshape(np.array(NormHam[dim**2:]).astype(float), (dim, dim))

    ## diagonalize normalized norm (using "eigh(ermitian)" to speed-up the computation)
    ewN, evN = eigh(normat)
    idx = ewN.argsort()[::-1]
    ewN = [eww for eww in ewN[idx]]
    evN = evN[:, idx]

    ewH, evH = eigh(hammat, normat)
    idx = ewH.argsort()[::-1]
    ewH = [eww for eww in ewH[idx]]
    evH = evH[:, idx]

    print(lec, ':  ', ewH[-3:])

exit()
