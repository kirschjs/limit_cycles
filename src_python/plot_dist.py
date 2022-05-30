import os
import math
import numpy as np
import matplotlib.pyplot as plt


def plotwidths(wrels, coeffs):

    assert len(wrels) == len(coeffs)

    dime = len(wrels)

    rSpace = np.linspace(0, 20, 200)

    gswfkt = [
        np.sum([coeffs[n] * np.exp(-wrels[n] * rr**2) for n in range(dime)])
        for rr in rSpace
    ]

    fig = plt.figure(figsize=(10, 12), dpi=95)
    fig.subplots_adjust(hspace=0.4, wspace=0.4)

    ax1 = plt.subplot(111)
    ax1.set_title(r'dimer $\rho$')
    ax1.set_xlabel(r'$\rho$ [fm]')
    ax1.set_ylabel(r'$\phi(\rho)$')
    ax1.plot(rSpace, gswfkt)

    #leg = ax1.legend(loc='best')
    fig.savefig('bases.pdf')
    #plt.show()
    plt.clf()
    plt.close()


wrkdir = '/home/kirscher/kette_repo/limit_cycles/systems/2'
cow = np.array([line.split()
                for line in open(wrkdir + '/opt_deut.dat')]).astype(float)
rwids = cow[:, 1]
uecof = cow[:, 0]

plotwidths(rwids, uecof)
