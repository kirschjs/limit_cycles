import sys, os, re
import matplotlib.pyplot as plt
import numpy as np

#plt.style.use('bmh')

offs = 3
infile = '/home/kirscher/kette_repo/limit_cycles/systems/2_B2-05_B3-8.00/4.00/np3s/lec_l42.00_b2.22.dat'

data1 = np.array(
    [float(line.split()[1]) for line in open(infile) if line[0] != '#'])
lamda1 = np.array(
    [float(line.split()[0]) for line in open(infile) if line[0] != '#'])

infile = '/home/kirscher/kette_repo/limit_cycles/systems/2_B2-05_B3-8.00/4.00/np3s/lec_ex0_b2.22.dat'

data2 = np.array(
    [float(line.split()[1]) for line in open(infile) if line[0] != '#'])
lamda2 = np.array(
    [float(line.split()[0]) for line in open(infile) if line[0] != '#'])

xval = [lamda1, lamda2]
yval = [data1, data2]

fig = plt.figure()
#num=None, figsize=(14, 24), dpi=60, facecolor='w', edgecolor='k')
#fig.suptitle(
#    r'$r_e\leq 2\left(R-\frac{R^2}{a}+\frac{R^3}{3a^2}\right)$', fontsize=16)
ax1 = fig.add_subplot(111)

#[
#    ax1.plot(
#        xval[nn],
#        np.abs(yval[nn]),
#        label=r'$B_{2}^{(%d)}$' % (nn),
#        #linestyle='dashed',
#        #linewidth=1,
#        #alpha=0.5
#    ) for nn in range(len(xval))
#]

ax1.plot(
    xval[0],
    np.abs(yval[0]) / np.abs(yval[1]),
    #label=r'$B_{2}^{(%d)}$' % (nn),
    #linestyle='dashed',
    #linewidth=1,
    #alpha=0.5
)

ax1.set_xlabel(r'$\lambda\;\;[fm^{-1}]$', fontsize=15)
ax1.set_ylabel(r'$C\;\;[MeV]$', fontsize=15)

#ax1.axhline(y=r, xmin=0, xmax=1)
#plt.ylim(-0.5 * r, 2 * r)
#plt.legend(loc='best', fontsize=22)
#plt.show()

fig.savefig("lec_of_l.pdf", bbox_inches='tight')