import os, re
import math
import numpy as np
import matplotlib.pyplot as plt


def sGauss(rr, co, wi):

    assert len(co) == len(wi)

    return np.sum(
        np.array([co[n] * np.exp(-wi[n] * rr**2) for n in range(len(co))]))


def plotwidths(sysdir):

    co_wi = []
    for root, dirs, files in os.walk(sysdir):
        for f in files:
            if re.search('civ_', f):
                idx = int(f.split('_')[1].split('.')[0])
                lines = [line for line in open(sysdir + '/' + f)]
                co_wi.append([
                    np.array([line.split()
                              for line in lines[1:]]).astype(float),
                    float(lines[0]), idx
                ])

    co_wi.sort(key=lambda tup: tup[2])
    dime = len(co_wi[0][0][0])

    rSpace = np.linspace(0, 2.1, 100)

    wfkts = []
    widths = []
    coeffs = []
    norm = []
    for n in range(len(co_wi)):
        coeffs.append(co_wi[n][0][:, 0])
        widths.append(co_wi[n][0][:, 1])
        tmp = 0.0
        for i in range(dime):
            for j in range(dime):
                tmp += coeffs[-1][i] * coeffs[-1][i] * (
                    np.pi / (widths[-1][i] + widths[-1][j]))**(1.5)
        norm.append(tmp**(-1))

    for m in range(len(co_wi)):
        wfkts.append([[sGauss(rr, coeffs[m], widths[m]) for rr in rSpace],
                      co_wi[m][1], co_wi[m][2]])

    fig = plt.figure(figsize=(10, 12), dpi=95)
    fig.subplots_adjust(hspace=0.4, wspace=0.4)

    ax1 = plt.subplot(111)
    ax1.set_title(r'dimer $\rho$')
    ax1.set_xlabel(r'$\rho$ [fm]')
    ax1.set_ylabel(r'$\phi(\rho)$')
    [
        ax1.plot(rSpace, gswfkt[0], label='%d) %4.6f' % (gswfkt[2], gswfkt[1]))
        for gswfkt in wfkts
    ]

    leg = ax1.legend(loc='best')
    fig.savefig('bases.pdf')

    plt.clf()
    plt.close()