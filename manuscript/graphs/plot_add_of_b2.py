import sys, os, re
import matplotlib.pyplot as plt
import numpy as np

MeVfm = 197.3161329
mu = 938.91852

wrkdir = '/home/kirscher/kette_repo/limit_cycles/manuscript/graphs'
os.chdir(wrkdir)

# read data

infile = 'add_data.dat'
data = np.array([line.split() for line in open(infile) if line[0] != '#'])

labt = [r'ABAB', r'ABCC', r'ABCD']
labb = ['1', '2.2', '5', '7']
colo = ['orange', 'g', 'black']

fig = plt.figure()
ax1 = fig.add_subplot(111)
#ax1.set_ylim([1.0, 5.5])
ax1.set_xlabel(r'B(dimer) [MeV]', fontsize=15)
ax1.set_ylabel(r'$a_{dd}/a_{aa}\;\;[fm]$', fontsize=15)

for add2plot in [0, 1, 2]:
    pdata = []
    for nn in range(len(data)):
        if (data[nn][2 + add2plot] != 'nan'):
            pdata.append([data[nn][1], data[nn][2 + add2plot]])
    pdata = np.array(pdata)
    pdat = []
    for b2 in np.unique(pdata[:, 0]):
        aaa = MeVfm / np.sqrt(mu * float(b2))
        bset = np.array([float(pda[1]) for pda in pdata
                         if pda[0] == b2]).astype(float)
        avg = np.average(bset)
        std = np.std(bset)
        pdat.append([b2, avg, std])
    xdata = np.array(pdat).astype(float)[:, 0]
    ydata = np.array(pdat).astype(float)[:, 1]
    errdata = np.array(pdat).astype(float)[:, 2]
    label = labt[add2plot]
    ax1.plot(xdata,
             ydata,
             color=colo[add2plot],
             linestyle='dashed',
             linewidth=1.,
             alpha=1.)
    plt.fill_between(xdata,
                     ydata - errdata,
                     ydata + errdata,
                     label=label,
                     color=colo[add2plot],
                     alpha=0.1)

#ax1.axhline(y=r, xmin=0, xmax=1)
plt.legend(loc='best', fontsize=12)

fig.savefig("addofB.pdf", bbox_inches='tight')