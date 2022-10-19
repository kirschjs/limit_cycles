import os
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import shutil


def plotphas(infi='PHAOUT'):

    # read entire file
    file = [line for line in open(infi)]

    # obtain matrix indices
    ch = [line.split()[2:4] for line in file]
    chu = []
    for ee in ch:
        if ((ee in chu) | (ee[::-1] in chu)) == False:
            chu.append(ee)

    # read phases
    method = '1'
    phs = []
    plt.subplot(111)
    #plt.set_title("channel: neutron-neutron")
    #leg = ax1.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
    #    ncol=1, mode="expand", borderaxespad=0.)
    plt.xlabel(r'$E_{cm}$ [MeV]')
    plt.ylabel(r'$\delta$ [deg]')

    for cha in chu:

        tmp = np.array([
            line.split() for line in file
            if ((line.split()[2:4] == cha) & (line.split()[-1] == method))
        ]).astype(float)
        stylel = 'solid' if cha[0] == cha[1] else 'dashdot'
        plt.plot(tmp[:, 0], tmp[:, 10], label=''.join(cha), linestyle=stylel)

    plt.legend(loc='best', numpoints=1)
    plt.show()
    exit()


wrkdir = '/home/johannesk/kette_repo/limit_cycles/systems/4/'
os.chdir(wrkdir)

plotphas()