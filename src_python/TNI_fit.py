import subprocess
from parameters_and_constants import *
from rrgm_functions import *
from scipy import optimize


def btt(tni, fit_bdg, orda=0):

    coeff = '%12.6f%12.6f%12.6f%12.6f%12.6f%12.6f%12.6f\n' % (
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, tni)

    repl_line('INEN', 3, coeff)

    subprocess.run([BINBDGpath + 'DR2END_AK.exe'])

    lines_output = [line for line in open('OUTPUT')]

    for lnr in range(0, len(lines_output)):
        if lines_output[lnr].find('EIGENWERTE DES HAMILTONOPERATORS') >= 0:
            mdt = float(lines_output[lnr + 3].split()[orda])

    return mdt + fit_bdg


dir3 = '/home/johannesk/kette_repo/limit_cycles/systems/3/3.00/he'

os.chdir(dir3)

fit_ene = 8.482
ordn = 0
fac = 1.01

ft = optimize.newton(btt, fac, args=(fit_ene, ordn), maxiter=100)

print('D0 initial       = %4.4f MeV\nD0(B3=%4.4fMeV) = %12.8f MeV' %
      (d0, fit_ene, ft * d0))

btt(ft, fit_ene, orda=ordn)