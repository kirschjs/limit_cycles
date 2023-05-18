import numpy as np
from sympy import *
from sympy.combinatorics import *
'''
anti symmetrizes a list of symbols with each specifying a product of
flavour states, e.g. '1 2' = spin-up * spin-down
                or   '2 3' = proton-spin-down * neutron-spin-up

INPUT: file 
l nbr   entry
  1     f1  f2  ... fN  (ECCE: there might be fewer/more flavours than particles, i.e. in general, N<>|f|)
  2     f1' f2' ... fN'
  .
  .
  .
|nProd| fi^(nProd) ...
  1     coeff(1)               format: 'numerator**2 denominator**2' (usual way to express Clebsch-Gordon coeffs.)
  .        .
  .        .
  .        .
|nProd| coeff(nProd)

OUTPUT:

A[ \sum_i^nProd coeff(i)*f1f2...fN(i)]
'''

bose = 0

# read input wave function as a list of flavour products and superposition coefficients
input_waveFunction = [line for line in open('spinWFKT_triton.lst')]

# consider the flavour products separately
elemprod = [
    np.array(ee.strip().split()).astype('int')
    for ee in input_waveFunction[:int(len(input_waveFunction) / 2)]
]

# infer the number of particles from the length of an individual product, e.g. '1 2 3' => 3 identical particles
nA = len(elemprod[0])

# define the symmetric group on nA particles
SN = list(symmetric(nA))

# obtain the list of numerical numerical superposition coefficients
clgc = [
    float(
        float(cc.strip().split(' ', maxsplit=1)[0]) /
        float(cc.strip().split(' ', maxsplit=1)[1]))
    for cc in input_waveFunction[int(len(input_waveFunction) / 2):]
]

assert len(elemprod) == len(clgc)

asy_elemprod = []
asy_clg = []

# antisymmetrization of the wave function
# permutation loop
for perm in SN:

    asy_elemprod.append([])
    asy_clg.append([])

    sign = -1 if perm.is_odd | bose == 0 else +1

    # apply permutation to each product
    for n in range(len(elemprod)):
        asy_elemprod[-1].append(perm(elemprod[n]))
        # ECCE multiply each product with its original superposition coefficient
        asy_clg[-1].append(sign * clgc[n])

# 'flatten' the resultand lists
asy_elemprod = sum(asy_elemprod, [])
asy_clg = sum(asy_clg, [])

assert len(asy_elemprod) == len(asy_clg)

# transform the integer lists into symbols which can be subjected to algebraic manipulations
x = []
for n in range(len(asy_elemprod)):
    tmp = ''.join(np.array(asy_elemprod[n]).astype('str'))
    x.append(asy_clg[n] * Symbol(tmp))

# print the anti-symmetrized wave function
print(nsimplify(sum(x), tolerance=1e-16))
print(sum(x))