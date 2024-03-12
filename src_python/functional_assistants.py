import subprocess
import multiprocessing
import os, fnmatch, copy, struct, time, sys
import numpy as np
import shlex
from scipy.linalg import eigh
import shutil
"""
obtain a list of numbers in a specified interval
which are expontentially distributed
(1/s)*exp(-x/s) 

num       = how many?
start/stop = bounds of the interval
scale     = s (smaller s => higher value density at lower end(start))
"""


def expspace(start=0.0, stop=1.0, scal=1.0, num=10):
    ws = []
    ite = 0
    while len(ws) < num:
        tmp = np.array([
            start + ll * (stop - start)
            for ll in np.sort(np.random.exponential(scale=scal, size=num))
        ])
        condlist = [(tmp < stop) & (start < tmp)]
        choicelist = [tmp]
        ws += [w for w in np.select(condlist, choicelist) if w > 0]
        ite += 1
        if ite > num**2:
            print(
                'exponential grid could not be set up with the given constraints!'
            )
            exit()

    return np.sort(ws[:num])


def log_with_density_enhancement(wi, wf, lbase, numb, num_additional_points,
                                 std_dev):
    sta = np.log(wi) / np.log(lbase)
    sto = np.log(wf) / np.log(lbase)
    init_points = np.abs(
        np.logspace(start=sta,
                    stop=sto,
                    base=lbase,
                    num=numb,
                    endpoint=True,
                    dtype=None))

    # Calculate density of points in the logspace grid
    # density = np.diff(np.log(lit_w_tmp))

    # Identify points with lower density
    # low_density_points = lit_w_tmp[:-1][density < np.median(density)]

    # Generate additional points around low-density regions using Gaussian distribution
    out_grid = []
    for point in init_points:
        additional_point = np.random.normal(loc=point, scale=std_dev, size=1)
        out_grid.append(additional_point)

    # Sort the final array
    #lit_w_tmp = [ww for ww in np.sort(out_grid) if ww > 0.0]
    lit_w_tmp = np.sort(np.abs(out_grid))

    return np.array(lit_w_tmp)


def expspaceS(start=0.0, stop=1.0, scal=1.0, num=10, deltam=20):
    ws = []
    ite = 0
    while len(ws) < num:
        tmp = np.random.exponential(scale=scal)

        if ws == []:
            ws += [tmp]
        else:

            adw = False

            if ((tmp < stop) & (start < tmp)):
                adw = True
                for wi in ws:
                    diff = abs(tmp - wi)
                    maxPercentageDiff = np.max([wi / diff, tmp / diff])
                    if maxPercentageDiff > deltam:
                        #print(wi, tmp, diff, maxPercentageDiff)
                        adw = False
                        break
                if adw == True:
                    ws += [tmp]
        ite += 1
        if ite > num**2:
            #print(
            #    'exponential grid could not be set up with the given constraints!\n(ECCE) continuing with decreased minimally allowed distance.'
            #)
            ite = 0
            ws = []
            scal = 0.001 + 0.9 * np.random.random()
            if scal < 10**-5:
                print('(functional assistants)  expS failed!')
                exit()

    return np.sort(ws[:num]), deltam


"""
False: The distance between two array elements is larger than "minDist"% of the
       two elements: max(|ai/(ai-aj)|,|aj/(ai-aj)|)< "minDist"
True : there is at least one pair in the array whose distance is smaller than
       "minDist"% of either element
"""


def check_dist(width_array1=[], width_array2=[], minDist=10.0):
    tooClose = False
    # the number of decimals should be consistent with the precission
    # used to write these widths in the fortran input file INQUA_N

    closeSet = []

    rounded_array1 = np.round(width_array1, decimals=7)
    rounded_array2 = np.round(width_array2, decimals=7)

    if list(rounded_array2) == []:
        for m in range(len(rounded_array1) - 1):
            for n in range(m + 1, len(rounded_array1) - 1):
                delt = np.linalg.norm(rounded_array1[m] - rounded_array1[n])
                if delt == 0.0:
                    #print(
                    #    'identical widths (%d,%d)=(%f,%f) in ws!\n' %
                    #    (m, n, rounded_array1[m], rounded_array1[n]),
                    #    rounded_array1)
                    #print('id close ', end='')
                    tooClose = True
                    closeSet.append([m, n])
                    continue

                nm = np.max([
                    np.linalg.norm(rounded_array1[m]) / delt,
                    np.linalg.norm(rounded_array1[n]) / delt
                ])

                if (nm > minDist):
                    tooClose = True
                    closeSet.append([m, n])
                    continue
    else:
        for m in range(1, len(rounded_array1)):
            for n in range(1, len(rounded_array2)):
                delt = np.linalg.norm(rounded_array1[m] - rounded_array2[n])
                if delt == 0:
                    tooClose = True
                    return tooClose
                nm = np.max([
                    np.linalg.norm(rounded_array1[m]) / delt,
                    +np.linalg.norm(rounded_array2[n]) / delt
                ])
                if (nm > minDist):
                    tooClose = True
                    return tooClose

    return tooClose  #, closeSet


def smart_ev(matout, threshold=10**-7):

    dim = int(np.sqrt(len(matout) * 0.5))

    # read Norm and Hamilton matrices
    normat = np.reshape(np.array(matout[:dim**2]).astype(float), (dim, dim))
    hammat = np.reshape(np.array(matout[dim**2:]).astype(float), (dim, dim))

    # obtain naively the ratio between the smallest and largest superposition
    # coefficient in the expansion of the ground state; use this as an additional
    # quality measure for the basis
    gsCoeffRatio = 42.1
    #    try:

    ewn, evn = eigh(normat)
    idxn = ewn.argsort()[::-1]
    ewn = [eww for eww in ewn[idxn]]
    normCond = np.abs(ewn[-1] /
                      ewn[0]) if np.any(np.array(ewn) < 0) == False else -1.0
    ewt, evt = eigh(hammat, normat)
    idxt = ewt.argsort()[::-1]
    ewt = [eww for eww in ewt[idxt]]
    evt = evt[:, idxt]
    gsC = np.abs(evt[:, -1])
    gsCoeffRatio = np.max(gsC) / np.min(gsC)

    #    except:
    #        gsCoeffRatio = 10**8
    #        normCond = -1.0

    # normalize the matrices with the Norm's diagonal
    normdiag = [normat[n, n] for n in range(dim)]
    umnorm = np.diag(1. / np.sqrt(normdiag))
    nm = np.dot(np.dot(np.transpose(umnorm), normat), umnorm)
    hm = np.dot(np.dot(np.transpose(umnorm), hammat), umnorm)

    # diagonalize normalized norm (using "eigh(ermitian)" to speed-up the computation)
    ew, ev = eigh(nm)
    #ew, ev = LA.eigh(nm)
    idx = ew.argsort()[::-1]
    ew = [eww for eww in ew[idx]]

    # project onto subspace with ev > threshold
    ew = [eww for eww in ew if np.real(eww) > threshold]
    dimRed = len(ew)
    ev = ev[:, idx][:, :dimRed]

    # transormation matric for (H-E*N)PSI=0 such that N->id
    Omat = np.dot(ev, np.diag(1. / np.sqrt(ew)))

    # diagonalize the projected Hamiltonian (using "eigh(ermitian)" to speed-up the computation)
    Hgood = np.dot(np.dot(np.transpose(Omat), hm), Omat)
    #ewGood, evGood = LA.eigh(Hgood)
    ewGood, evGood = eigh(Hgood)

    idx = ewGood.argsort()[::-1]
    ewGood = [eww for eww in ewGood[idx]]
    evGood = evGood[:, idx]

    #ewt, evt = eigh(hammat, normat)
    #idxt = ewt.argsort()[::-1]
    #ewt = [eww for eww in ewt[idxt]]
    #evt = evt[:, idxt]
    #print('(stable) Eigenbasisdim = %d(%d)' % (dimRed, dim))
    #return the ordered eigenvalues
    return ewGood, normCond, gsCoeffRatio


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
        nor = int(nbr * (nbr + 1) * (2 * nbr + 1) *
                  (3 * nbr**2 + 3 * nbr - 1) / 30)
        p = [n**4 / nor for n in range(nbr + 1)]
    else:
        print("(polynomial sum weight) no implementation for this order!")
        exit()

    #print('weights: ', p, sum(p))
    return p


def sortprint(civi, pr=False, ordn=2):

    civi.sort(key=lambda tup: tup[ordn])
    civi = civi[::-1]
    if pr:
        print(
            '\n        pulchritude       E_groundstate cond. number (norm)\n-----------------------------------------------------------'
        )

        for civ in civi:
            print('%19.5e %19.5f %19.5e' % (civ[2], civ[3], civ[4]))
        print('-----------------------------------------------------------')
    return civi


def write_indiv(indi, outf):

    # indi = channel, relw, qualREF, gsREF, basCond, gsvREF
    if os.path.exists(outf): os.remove(outf)
    sout = '%12.8f\n' % indi[3]

    for n in range(len(indi[1])):
        sout += '%12.4e %12.4e\n' % (float(indi[-1][n]), float(indi[1][n]))

    with open(outf, 'w') as oof:
        oof.write(sout)
    oof.close()


def write_indiv3(indi, outf):

    if os.path.exists(outf): os.remove(outf)
    sout = '%12.8f\n' % indi[3]

    sbas = []
    bv = 1
    for n in range(len(indi[0])):
        sbas.append([])
        for m in range(len(indi[1][0][n])):
            sbas[-1].append([
                bv, [x for x in range(1 + bv % 2, 1 + len(indi[1][1][n]), 2)]
            ])
            bv += 1

    w1 = sum(indi[1][0], [])

    nbv = 0
    for ncfg in range(len(indi[0])):
        for bv in sbas[ncfg]:
            for rel in bv[1]:
                sout += '%12.4e %12.4e %12.4e\n' % (float(
                    indi[5][nbv]), float(
                        w1[bv[0] - 1]), float(indi[1][1][ncfg][rel - 1]))
                nbv += 1

    with open(outf, 'w') as oof:
        oof.write(sout)
    indi[6].tofile('n_' + outf)
    oof.close()