import subprocess
import multiprocessing
import os, fnmatch, copy, struct, time, sys
import numpy as np
import shutil


def du(path):
    """disk usage in human readable format (e.g. '2,1GB')"""
    tmp = subprocess.check_output(
        'du -s %s 2>&1 | grep -v "Permission denied"' % path,
        shell=True  #'2>&1' | grep -v "Permission denied"
    ).split()[0].decode('utf-8')

    return tmp


def disk_avail(path):

    (total, used, free) = shutil.disk_usage(path)

    return used / free


def polynomial_sum_weight(nbr, order=1):
    if order == 1:
        nor = int(0.5 * nbr * (nbr + 1))
        p = [n / nor for n in range(nbr + 1)]
    elif order == 2:
        nor = int(nbr * (nbr + 1) * (2 * nbr + 1) / 6)
        p = [n**2 / nor for n in range(nbr + 1)]
    elif order == 3:
        nor = int(0.25 * nbr**2 * (nbr + 1)**2)
        p = [n**3 / nor for n in range(nbr + 1)]
    elif order == 4:
        nor = int(nbr * (nbr + 1) * (2 * nbr + 1) * (3 * nbr**2 + 3 * nbr - 1))
        p = [n**4 / nor for n in range(nbr + 1)]
    else:
        print("(polynomial sum weight) no implementation for this order!")
        exit()

    return p


def sortprint(civi, pr=False):

    civi.sort(key=lambda tup: tup[2])
    civi = civi[::-1]
    if pr:
        print(
            '\n        pulchritude       E_groundstate cond. number (norm)\n-----------------------------------------------------------'
        )

        for civ in civi:
            print('%19.5f %19.5f %19.5e' % (civ[2], civ[3], civ[4]))
        print('-----------------------------------------------------------')
    return civi


def write_indiv(indi, outf):

    if os.path.exists(outf): os.remove(outf)
    sout = '%12.8f\n' % indi[3]
    for n in range(len(indi[1])):
        sout += '%12.4e %12.4e\n' % (float(indi[-1][n]), float(indi[1][n]))

    with open(outf, 'w') as oof:
        oof.write(sout)
    oof.close()