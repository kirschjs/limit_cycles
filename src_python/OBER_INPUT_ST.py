import os
from itertools import permutations, product  #,izip
# functions: clg(5I), f6 (6I), f9j(9I), s6j(6I)
# clg(s/2,s'/2,j/2,m/2,m'/2)
#import wign
#print 'wigne CLG: ',wign.clg(2,2,0,2,-2)
#
import numpy as np
#
from sympy.physics.quantum.cg import CG
#                       s m s' m' j M
#print 'sympy CLG: ',CG(1,1,1,-1,0,0).doit()
#
dbg = 0
# define coupling structure ---------------------
# RGM convention: total Spin = total m_S
c_scheme = {
    #                  S       T
    'j': [0., 0.],
    's1': [1. / 2., 1. / 2.],
    's2': [1. / 2., 1. / 2.],
    's3': [1. / 2., 1. / 2.],
    's4': [1. / 2., 1. / 2.],
    #'s5'          :[ 1./2., 1./2.],
    #                spin   iso-spin      tz
    's1_s2': [0, 1, 0],
    's1_s2_s3': [1. / 2., 1. / 2., -1. / 2.],
    #'s1_s2_s3_s4': [0, 0, 0]
}
#
#                 mS       mT
jz = [0., 0.]
# deduce number of particles --------------------
A = 0
for k in c_scheme.keys():
    if len(k) == 2:
        A += 1
# deduce final coupling -------------------------
# (S m sA msA|j j) or (S m S' m'|j j)
max1 = []
max2 = []
maxc = 0
#
spin_tripl = []
range_set = []
#
for coupling in c_scheme.keys():
    sc = coupling.rsplit('_', 1)
    #
    if len(coupling.split('_')) >= maxc:
        max2 = max1
        max1 = coupling
        maxc = len(coupling.split('_'))
    # deduce triple contributing to CLG product -
    if len(sc) > 1:
        spin_tripl.append([sc[0], sc[1], coupling])
#
# add CGL tripl <-> final coupling --------------
if maxc == A - 1:
    spin_tripl.append([max1, 's%s' % str(A), 'j'])
else:
    spin_tripl.append([max1, max2, 'j'])
# flatten chain of couplings --------------------
cpl_chain = np.array(spin_tripl).reshape(1, -1)[0]
if dbg:
    print('fr1_max,fr2_max,last_c:  ', max1, max2, maxc)
    print('cpl chain             :  ', cpl_chain, '\n--')
# list m ranges ---------------------------------
#           S T
range_set.append([])
for s in cpl_chain[:-1]:
    range_set[0].append(np.arange(-c_scheme[s][0], c_scheme[s][0] + 1, 1))
range_set[-1].append([jz[0]])
range_set.append([])
for s in cpl_chain[:-1]:
    try:
        range_set[1].append([c_scheme[s][2]])
    except:
        range_set[1].append(np.arange(-c_scheme[s][1], c_scheme[s][1] + 1, 1))
# fix j_z to j (stretched coupling) -------------
range_set[-1].append([jz[1]])
#
emp = []
# sum over all possible m combinations ----------
for nn in range(len(range_set)):
    elementare_produkte = []
    clebsche = []
    mrange = range_set[nn]
    if dbg: print('cs %d: mrange = ' % nn, mrange)
    for elm in product(*mrange):
        #
        clebsch = 1
        for s in range(len(cpl_chain)):
            for m in range(s + 1, len(cpl_chain)):
                if ((cpl_chain[s] == cpl_chain[m]) & (elm[s] != elm[m])):
                    clebsch = 0
        if clebsch == 0: continue
        for s in range(int(len(elm) / 3)):
            clebsch *= CG(abs(c_scheme[cpl_chain[3 * s]][nn]), elm[3 * s],
                          abs(c_scheme[cpl_chain[3 * s + 1]][nn]),
                          elm[3 * s + 1],
                          abs(c_scheme[cpl_chain[3 * s + 2]][nn]),
                          elm[3 * s + 2]).doit()
        if clebsch != 0.0:
            #
            #if dbg: print 'calc. clg: ',elm
            #print c_scheme[cpl_chain[3*s]][nn],elm[3*s],c_scheme[cpl_chain[3*s+1]][nn],elm[3*s+1],c_scheme[cpl_chain[3*s+2]][nn],elm[3*s+2]
            elemprod = 'x' * (3 * A)
            for spin in range(len(cpl_chain)):
                if len(cpl_chain[spin]) == 2:
                    subs = '  3' if elm[spin] > 0 else '  4'
                    elemprod = elemprod[:3 * (
                        int(cpl_chain[spin][-1]) - 1
                    )] + subs + elemprod[3 *
                                         (int(cpl_chain[spin][-1]) - 1) + 3:]
            if dbg: print(clebsch, elm, elemprod, cpl_chain)
            elementare_produkte.append(elemprod)
            clebsche.append(clebsch)
    emp.append([elementare_produkte, clebsche])
#
elementare_produkte = []
clebsche = []
for so in range(len(emp[0][0])):
    for io in range(len(emp[1][0])):
        subs = ''
        sp = emp[0][0][so].split()
        iso = emp[1][0][io].split()
        for s in range(A):
            if ((sp[s] == '3') & (iso[s] == '3')):
                subs += '  1'
            elif ((sp[s] == '3') & (iso[s] == '4')):
                subs += '  3'
            elif ((sp[s] == '4') & (iso[s] == '3')):
                subs += '  2'
            elif ((sp[s] == '4') & (iso[s] == '4')):
                subs += '  4'
        elementare_produkte.append(subs)
        clebsche.append(emp[0][1][so] * emp[1][1][io])
#
if clebsche == []:
    print('Ecce! The state cannot be coupled as defined.')
print('%3d%3d%3d%3d' % (A, len(elementare_produkte), 1, maxc))
for n in range(len(elementare_produkte)):
    print(elementare_produkte[n])
for n in range(len(elementare_produkte)):
    print(np.sign(clebsche[n]), clebsche[n]**2)
exit()