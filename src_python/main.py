from rrgm_functions import *

#parse_ev_coeffs(mult=1, infil='../systems/2/OUTPUT', outf='COEFF', bvnr=1)
chan = [2, 2]
ph = read_phase(phaout='../systems/4/PHAOUT', ch=chan, meth=1, th_shift='')
write_phases(ph,
             filename='diagPH_%d-%d.dat' % (chan[0], chan[1]),
             append=0,
             comment='')

#np = 0
#for n in range(97, 145):
#    while np < 20:
#        np += 1
#        print('%4d' % n, end='')
#        n += 1
#    print('\n')
#    np = 0
