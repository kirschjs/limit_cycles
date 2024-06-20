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


def antisymmetrize_list(listfile='', bose=0, id=0):

    if listfile == '':
        print('no list file provided. exiting.')
        exit()

    # read input wave function as a list of flavour products and superposition coefficients
    input_waveFunction = [line for line in open(listfile)]

    # consider the flavour products separately
    elemprod = [
        np.array(ee.strip().split()).astype('int')
        for ee in input_waveFunction[:int(len(input_waveFunction) / 2)]
    ]

    # infer the number of particles from the length of an individual product, e.g. '1 2 3' => 3 identical particles
    nA = len(elemprod[0])
    # define the symmetric group on nA particles
    SN = list(symmetric(nA))
    if id == 1:
        SN = [SN[0]]

    # obtain the list of numerical numerical superposition coefficients
    clgc = [
        float(
            np.sign(float(cc.strip().split(' ', maxsplit=1)[0])) * np.sqrt(
                np.abs(float(cc.strip().split(' ', maxsplit=1)[0])) /
                float(cc.strip().split(' ', maxsplit=1)[1])))
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
    #print(nsimplify(sum(x), tolerance=1e-16))

    return sum(x)


def list_product(lista, listb):

    #assert len(lista.args) == len(listb.args)

    prod_val = 0

    for a in lista.args:
        for b in listb.args:

            if len((a * b).free_symbols) == 1:
                #print(a * b)
                tmp = [
                    cc for cc in (a * b).atoms()
                    if ((cc.is_number) & (cc != 2))
                ]
                prod_val += tmp[0]

    return prod_val


# 4-body: strx \in { dd dqdq nnpp he01 he06 tp01 tp06  }

#chstr = ['tp01', 'tp06', 'hen01', 'hen06', 'dd', 'dqdq', 'nnpp']
chstr = ['123']

dim = len(chstr)
overlap = np.zeros(shape=(dim, dim))
for col in range(dim):
    for row in range(dim):

        str1 = chstr[row]
        str2 = chstr[col]

        listfile1 = 'spinWFKT_%s.lst' % str1
        listfile2 = 'spinWFKT_%s.lst' % str2

        list1 = antisymmetrize_list(listfile1, bose=0, id=1)
        list2 = antisymmetrize_list(listfile2, bose=0, id=0)
        print(list1)
        print(list2)
        exit()

        prod = list_product(list1, list2)
        overlap[col][row] = '%2.4f' % prod

        #print('%s*%s = %f' % (listfile1.split('_')[1].split('.')[0],
        #                      listfile2.split('_')[1].split('.')[0], prod))

print(np.matrix(overlap))