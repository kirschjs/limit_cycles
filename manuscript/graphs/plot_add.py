import sys, os, re
import matplotlib.pyplot as plt
import numpy as np

wrkdir = '/home/kirscher/kette_repo/limit_cycles/manuscript/graphs'
os.chdir(wrkdir)

# read data

infile = 'add_data.dat'
data = np.array([line.split() for line in open(infile) if line[0] != '#'])

labt = [r'ABAB', r'ABCC', r'ABCD']
labb = ['1', '2.2', '5', '7']
colo = ['r', 'g', 'b']

fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.set_ylim([1.0, 5.5])
ax1.set_xlabel(r'$\lambda\;\;[fm^{-1}]$', fontsize=15)
ax1.set_ylabel(r'$a_{dd}\;\;[fm]$', fontsize=15)

for b2plot in [0, 1, 2, 3]:

    for add2plot in [0, 1, 2]:
        pdata = []

        for nn in range(len(data)):
            if ((data[nn][1] == labb[b2plot]) &
                (data[nn][2 + add2plot] != 'nan')):
                pdata.append([data[nn][0], data[nn][2 + add2plot]])

        pdata = np.array(pdata)
        pdat = []
        for lam in np.unique(pdata[:, 0]):
            lset = np.array([pda[1] for pda in pdata
                             if pda[0] == lam]).astype(float)
            avg = np.average(lset)
            std = np.std(lset)
            pdat.append([lam, avg, std])
        xdata = np.array(pdat).astype(float)[:, 0]
        ydata = np.array(pdat).astype(float)[:, 1]
        errdata = np.array(pdat).astype(float)[:, 2]

        label = labt[add2plot] if b2plot == 0 else r''
        ax1.plot(xdata,
                 ydata,
                 color=colo[add2plot],
                 linestyle='dashed',
                 linewidth=1.,
                 alpha=0.5)
        plt.fill_between(xdata,
                         ydata - errdata,
                         ydata + errdata,
                         label=label,
                         color=colo[add2plot],
                         alpha=0.2)

#ax1.axhline(y=r, xmin=0, xmax=1)
plt.legend(loc='best', fontsize=12)

fig.savefig("addtmp.pdf", bbox_inches='tight')