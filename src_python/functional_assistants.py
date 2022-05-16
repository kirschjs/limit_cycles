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

def polynomial_sum_weight(nbr,order=1):
    switch(order){
    case 1:
        nor = int(0.5 * nbr * (nbr + 1))
        p= [ n/ nor for n in range(nbr+1)  ]
    case 2:
        nor = int(nbr * (nbr + 1)* (2*nbr + 1)/6)
        p=[ n**2/ nor for n in range(nbr+1)  ]
    case 3:
        nor = int(0.25 * nbr**2 * (nbr + 1)**2)
        p=[ n**3/ nor for n in range(nbr+1)  ]
    case 4:
        nor = int(nbr * (nbr + 1) * (2*nbr + 1) * (3*nbr**2+3*nbr - 1))
        p=[ n**4/ nor for n in range(nbr+1)  ]
    }
    return p