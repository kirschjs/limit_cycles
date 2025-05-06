import sys, os, re
import matplotlib.pyplot as plt
import numpy as np

#plt.style.use('bmh')
wrkdir = '/home/kirscher/kette_repo/limit_cycles/manuscript/graphs'
os.chdir(wrkdir)

offs = 3

# read 3-body spectrum as a function of LEC-D

infile3 = '/home/kirscher/kette_repo/limit_cycles/systems/3_B2-05_B3-8.00/4.00/123/spect/Spect3_of_D.dat'
data3 = [line.split() for line in open(infile3)]

lecd3 = [float(da[0]) for da in data3]
ene3 = [np.abs(np.array(da[1:]).astype(float)) for da in data3]
uncert3 = 0.5

# the ground-state energy as f(LEC-D) sets the threshold only below which
# a 4-body EV corresponds to a state, stable wrt. to a 3-1 decay
ths3 = [np.log(np.abs(gs[-1])) for gs in ene3]

# 2-1 breakup threshold set by the dimer binding energy
th2 = 2.22

# read 4-body spectrum as a function of LEC-D

infile4 = '/home/kirscher/kette_repo/limit_cycles/systems/4_B2-05_B3-8.00/4.00/alpha/Spect4_of_D.dat'
data4 = [line.split() for line in open(infile4)]

lecd4 = [float(da[0]) for da in data4]
ene4 = [np.abs(np.array(da[1:]).astype(float)) for da in data4]
uncert4 = 4.5

assert lecd4 == lecd3

numEV = len(ene3[0])

# plot data
lecs3 = []
evs3 = []
lecs4 = []
evs4 = []
for nn in range(1, numEV):
    lectemp3 = []
    evstemp3 = []
    lectemp4 = []
    evstemp4 = []
    for mm in range(len(lecd4)):
        if ene3[mm][-nn] > th2:
            lectemp3.append(lecd4[mm])
            evstemp3.append(ene3[mm][-nn])

        if len(ene4[mm]) >= nn:
            if ene4[mm][-nn] > ene3[mm][-1]:
                lectemp4.append(lecd4[mm])
                evstemp4.append(ene4[mm][-nn])

    lecs3.append(lectemp3)
    evs3.append(evstemp3)
    lecs4.append(lectemp4)
    evs4.append(evstemp4)

fig = plt.figure()
#num=None, figsize=(14, 24), dpi=60, facecolor='w', edgecolor='k')
#fig.suptitle(
#    r'$r_e\leq 2\left(R-\frac{R^2}{a}+\frac{R^3}{3a^2}\right)$', fontsize=16)
ax1 = fig.add_subplot(111)

ax1.set_xlabel(r'$D\;\;[MeV]$', fontsize=15)
ax1.set_ylabel(r'$\log($B$^{(n)})\;\;[\log($MeV$)]$', fontsize=15)

ax1.plot(
    lecd4,
    [np.log(th2) for ll in lecd4],
    color='g',
    linestyle='dashed',
    linewidth=1.,
    #label=r'B$^{(0)}_2\;\;MeV$',
    alpha=0.5)

for nn in range(len(lecs3)):
    lab = r'B$^{(n)}_3$' if nn == 0 else ''
    ax1.plot(lecs3[nn],
             np.log(np.abs(evs3[nn])),
             color='b',
             linestyle='dashed',
             linewidth=1,
             alpha=0.5)
    plt.fill_between(lecs3[nn],
                     [np.log(np.abs(ev3) - uncert3) for ev3 in evs3[nn]],
                     [np.log(np.abs(ev3) + uncert3) for ev3 in evs3[nn]],
                     color='b',
                     label=lab,
                     alpha=0.2)

for nn in range(len(lecs4)):
    lab = r'B$^{(n)}_4$' if nn == 0 else ''
    ax1.plot(lecs4[nn],
             np.log(np.abs(evs4[nn])),
             color='r',
             linestyle='dashed',
             linewidth=1,
             alpha=0.5)
    plt.fill_between(lecs4[nn],
                     [np.log(np.abs(ev4) - uncert4) for ev4 in evs4[nn]],
                     [np.log(np.abs(ev4) + uncert4) for ev4 in evs4[nn]],
                     label=lab,
                     color='r',
                     alpha=0.2)

#ax1.axhline(y=r, xmin=0, xmax=1)
plt.legend(loc='best', fontsize=22)

fig.savefig("A4_spectrum.pdf", bbox_inches='tight')