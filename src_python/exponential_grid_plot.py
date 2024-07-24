import os
import math
import numpy as np
import matplotlib.pyplot as plt
"""
obtain a list of numbers in a specified interval
which are expontentially distributed
(1/s)*exp(-x/s) 

anz       = how many?
wmin/wmax = bounds of the interval
scale     = s
"""


def expspace(wmin=0.0, wmax=1.0, scal=1.0, anz=10):
    ws = []
    while len(ws) < anz:
        tmp = np.array([
            wmin + ll * (wmax - wmin)
            for ll in np.sort(np.random.exponential(scale=scal, size=anz))
        ])
        condlist = [(tmp < wmax) & (wmin < tmp)]
        choicelist = [tmp]
        ws += [w for w in np.select(condlist, choicelist) if w > 0]

    return np.sort(ws[:anz])


def lapspace(wmin=0.0, wmax=1.0, scal=1.0, loca=1.0, anz=10):
    ws = []
    while len(ws) < anz:
        tmp = np.random.laplace(loc=loca, scale=scal, size=anz)
        condlist = [(tmp < wmax) & (wmin < tmp)]
        choicelist = [tmp]
        ws += [w for w in np.select(condlist, choicelist) if w > 0]

    return np.sort(ws[:anz])


aw = 1000
#wi = expspace(anz=aw, wmax=100, scal=1)
wi = lapspace(anz=aw, wmax=10, scal=1, loca=5)

fig = plt.figure(figsize=[12, 6])
ax1 = plt.subplot(121)
ax1.set_title(r'')
ax1.plot(wi, '.', color='black', label=r'$|d|=%d$' % aw)
leg = ax1.legend(loc='best')
outf = 'tmp.pdf'
fig.savefig(outf)
print('something was plotted in %s' % outf)