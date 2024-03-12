import sys, os, re
import matplotlib.pyplot as plt
import numpy as np

#plt.style.use('bmh')

offs = 3
infile = '/home/kirscher/kette_repo/limit_cycles/systems/3_B2-05_B3-8.00/4.00/123/spect/Spect3_of_D.dat'

data = np.array(
    [np.array(line.split()).astype(float) for line in open(infile)])
#evs = []
#d0 = []
#for n in range(0, len(data) - 1):
#    evs.append(data[n].split(' '))
#    d0.append(float(data[n].split(' ')[0]))

lecd = data[:, 0]
numEV = 4

lecs = []
evs = []
for nn in range(1, numEV):
    tlecs = []
    tevs = []
    for mm in range(len(lecd)):
        if data[mm, -nn] < -2.22:
            tlecs.append(lecd[mm])
            tevs.append(np.log(np.abs(data[mm, -nn])))
    lecs.append(tlecs)
    evs.append(tevs)

fig = plt.figure()
#num=None, figsize=(14, 24), dpi=60, facecolor='w', edgecolor='k')
#fig.suptitle(
#    r'$r_e\leq 2\left(R-\frac{R^2}{a}+\frac{R^3}{3a^2}\right)$', fontsize=16)
ax1 = fig.add_subplot(111)
ax1.plot(lecd, [np.log(2.22) for i in lecd],
         color='r',
         linestyle='dashed',
         linewidth=0.5,
         label=r'$E_2=2.22\;\;MeV$')
#ax1.set_yscale('log')
[
    ax1.plot(
        lecs[nn],
        evs[nn],
        label=r'$E_{3}^{(%d)}$' % (nn),
        #linestyle='dashed',
        #linewidth=1,
        #alpha=0.5
    ) for nn in range(numEV - 1)
]

ax1.set_xlabel(r'$D\;\;[MeV]$', fontsize=15)
ax1.set_ylabel(r'$\log(E)\;\;[\log(MeV)]$', fontsize=15)

#ax1.axhline(y=r, xmin=0, xmax=1)
#plt.ylim(-0.5 * r, 2 * r)
plt.legend(loc='best', fontsize=22)
#plt.show()

fig.savefig("A3_spectrum.pdf", bbox_inches='tight')