import os
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import shutil


def plotarray(infiy,
              infix,
              outfi,
              plotrange='max',
              xlab='$E_{cm}$ [MeV]',
              ylab='$\delta$ [deg]',
              lab='plotarray function (<plot_array.py>)'):

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

    if plotrange == 'max':
        plt.ylim(np.min(infiy), np.max(infiy))

    stylel = 'solid' if len(infiy) > 100 else 'dashdot'
    plt.plot(infix, infiy, label='%s' % lab, linestyle=stylel)

    plt.legend(loc='best', numpoints=1)
    plt.savefig(outfi)


def plotarray2(outfi,
               infix=[],
               infiy=[],
               title=[],
               plotrange=['max'],
               xlab=['$E_{cm}$ [MeV]'],
               ylab=['$\delta$ [deg]'],
               leg=[]):

    nbr_panels = len(infiy)

    nbr_plt_rows = int(np.ceil(nbr_panels / 2))

    fig, axs = plt.subplots(nbr_plt_rows, 2)
    if nbr_panels % 2 == 1: axs[nbr_plt_rows - 1, 1].set_visible(False)

    nSet = 0

    for nR in range(nbr_plt_rows):
        for nC in range(2):

            try:
                for nset in range(len(infiy[nSet])):
                    axs[nR, nC].plot(infix[nSet][nset],
                                     infiy[nSet][nset],
                                     linestyle='solid',
                                     label='%s' % leg[nSet][nset])
            except:
                axs[nR, nC].plot(infix[nSet], infiy[nSet])

            try:
                axs[nR, nC].set_title(title[nSet])
            except:
                axs[nR, nC].set_title("")

            try:
                axs[nR, nC].set_xlabel(r'%s' % xlab[nSet])
                axs[nR, nC].set_ylabel(r'%s' % ylab[nSet])
            except:
                axs[nR, nC].set_xlabel(r'X')
                axs[nR, nC].set_ylabel(r'Y')

            ym = np.median(np.array(infiy[nSet], dtype=object))

            if plotrange[nSet] != '':
                if ym < 0:
                    axs[nR, nC].set_ylim(bottom=2 * ym)
                else:
                    axs[nR, nC].set_ylim(top=2 * ym)

                    if plotrange[nSet] == 'max':
                        axs[nR, nC].set_ylim(
                            np.min(np.array(infiy[nSet], dtype=object)),
                            np.max(np.array(infiy[nSet], dtype=object)))
            #exit()

            #stylel = 'solid' if len(infiy) > 100 else 'dashdot'
            #plt.plot(infix, infiy, label='%s' % lab, linestyle=stylel)

            if leg[nSet] != []:
                axs[nR, nC].legend(loc='best', numpoints=1)

            nSet += 1
            if nSet >= nbr_panels: break

        if nSet >= nbr_panels: break

    #leg = ax1.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
    #    ncol=1, mode="expand", borderaxespad=0.)
    fig.tight_layout()
    plt.savefig(outfi)


#dom1 = np.linspace(0.01, 100, 100)
#ran1 = [np.log(x) for x in dom1]
#ran2 = [np.log(x**2) for x in dom1]
#plotarray2(outfi='tmp.pdf', infix=[dom1, dom1], infiy=[ran1, ran2],title=[a])