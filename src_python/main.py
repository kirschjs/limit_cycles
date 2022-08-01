import subprocess
from rrgm_functions import *

#parse_ev_coeffs(mult=1, infil='../systems/2/OUTPUT', outf='COEFF', bvnr=1)
chans = [[1, 1], [1, 2], [2, 2], [1, 3], [2, 3], [3, 3]]
chans = [[3, 3]]
for chan in chans:
    ph = read_phase(phaout='PHAOUT', ch=chan, meth=1, th_shift='')
    write_phases(ph,
                 filename='scdiagPH_%d-%d.dat' % (chan[0], chan[1]),
                 append=0,
                 comment='')

subprocess.call('gnuplot ../../src_python/phases.gnu', shell=True)
#np = 0
#for n in range(97, 145):
#    while np < 20:
#        np += 1
#        print('%4d' % n, end='')
#        n += 1
#    print('\n')
#    np = 0

#phchans = {
#    "tp": [[1, 1, 0], range(49, 97), range(97, 145)],
#    "hen": [[1, 1, 0], range(1, 49), range(1, 49)],
#    "dd": [[2, 2, 0], range(97, 118),
#           range(145, 166)]
#}
#
#for pch in phchans:
#    for ch in phchans[pch][1]:
#        print('%3d%3d%3d\n%4d%4d\n%4d\n  0  1  0  1  0  1' %
#              (phchans[pch][0][0], phchans[pch][0][1], phchans[pch][0][2], 1,
#               ch, phchans[pch][2][1]))