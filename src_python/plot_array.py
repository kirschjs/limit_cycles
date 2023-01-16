import os
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import shutil


def plotarray(infiy,
              infix,
              outfi,
              xlab='$E_{cm}$ [MeV]',
              ylab='$\delta$ [deg]',
              lab=''):

    # read entire file
    plt.cla()
    plt.subplot(111)
    #plt.set_title("channel: neutron-neutron")
    #leg = ax1.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
    #    ncol=1, mode="expand", borderaxespad=0.)
    plt.xlabel(r'%s' % xlab)
    plt.ylabel(r'%s' % ylab)
    ym = np.median(infiy)
    if ym < 0:
        plt.ylim(bottom=2 * ym)
    else:
        plt.ylim(top=2 * ym)

    stylel = 'solid' if len(infiy) > 100 else 'dashdot'
    plt.plot(infix, infiy, label='%s' % lab, linestyle=stylel)

    plt.legend(loc='best', numpoints=1)
    plt.savefig(outfi)