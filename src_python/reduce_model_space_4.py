from parameters_and_constants import *
import subprocess

# Analyses file "OUTPUT"; finds insignificant basis vectors based on their
# contribution to the 1st eigenvector and removes them from the variational
# basis in "INEN"

# calculation directory, upper and lower threshold,
# vectors with an expansion coefficient outside of this interval
# are excluded


def redmod(bin_path):

    max_coeff = 11000
    min_coeff = 90
    target_size = 100
    nbr_cycles = 400
    max_diff = 0.02
    ord = 0

    basis_size = 400000
    bdg_ini = 400000
    bdg_end = 400000
    diff = 0.0
    nc = 0

    while (nc <= nbr_cycles) & (basis_size > target_size):
        # print currently lowest eigenvalue
        lines_output = [line for line in open('OUTPUT')]
        for lnr in range(0, len(lines_output)):
            if lines_output[lnr].find('EIGENWERTE DES HAMILTONOPERATORS') >= 0:
                bdg_ini = float(lines_output[lnr + 3].split()[ord])
        print('Initial binding energy: B(4) = %f MeV' % (bdg_ini))

        # read file OUTPUT
        bv_ent = []
        for lnr in range(0, len(lines_output)):
            if lines_output[lnr].find(
                    'ENTWICKLUNG DES  %1d TEN EIGENVEKTORS,AUSGEDRUECKT DURCH NORMIERTE BASISVEKTOREN'
                    % (ord + 1)) >= 0:
                for llnr in range(lnr + 2, len(lines_output)):
                    if lines_output[llnr] == '\n':
                        break
                    else:
                        try:
                            if (int(lines_output[llnr].split(')')[0]) !=
                                    len(bv_ent) + 1):
                                bv_ent[-1] += lines_output[
                                    llnr][lines_output[llnr].find(')') +
                                          1:].rstrip()[2:]
                            else:
                                bv_ent.append(lines_output[llnr]
                                              [lines_output[llnr].find(')') +
                                               1:].rstrip()[2:])
                        except:
                            continue
        # identify the vectors with insignificant contribution;
        # the result is a pair (bv number, {relw1, relw2, ...})
        bv_to_del = []
        bv_to_del0 = []
        basis_size = 0
        for nn in bv_ent:
            basis_size += len(nn) / 8
        for bv in range(1, len(bv_ent) + 1):
            relw_to_del = []
            relw_to_del0 = []
            tmpt = bv_ent[bv - 1]
            ueco = [
                tmpt[8 * n:8 * (n + 1)]
                for n in range(0, int((len(tmpt.rstrip())) / 8))
            ]
            ueco = [tmp for tmp in ueco if (tmp != '') & (tmp != '\n')]
            for coeff in range(0, len(ueco)):
                try:
                    if (abs(int(ueco[coeff])) > max_coeff) | (
                        (abs(int(ueco[coeff])) < min_coeff) &
                        (abs(int(ueco[coeff])) != 0)):
                        relw_to_del.append(coeff)
                    if (abs(int(ueco[coeff])) == 0):
                        relw_to_del0.append(coeff)
                except:
                    relw_to_del.append(coeff)
            try:
                bv_to_del.append([bv, relw_to_del])
                bv_to_del0.append([bv, relw_to_del0])
            except:
                print('bv %d is relevant!' % bv)
        bv_to_del = [bv for bv in bv_to_del if bv[1] != []]
        bv_to_del0 = [bv for bv in bv_to_del0 if bv[1] != []]
        rednr = sum([len(tmp[1]) for tmp in bv_to_del]) + sum(
            [len(tmp[1]) for tmp in bv_to_del0])

        if ((rednr == 0) | (len(bv_ent[0]) / 8 == target_size)):
            print(
                'after removal of abnormally large/small BV (%2d iterations).'
                % nc)
            break
            # from the input file INEN remove the basis vectors with
            # number bv=bv_to_del[0] and relative widths from the set bv_to_del[1]
            # note: the indices refer to occurance, not abolute number!
            # e.g.: bv is whatever vector was included in INEN as the bv-th, and the
            # rel-width is the n-th calculated for this bv

        lines_inen = [line for line in open('INEN')]
        bv_to_del = [tmp for tmp in bv_to_del if tmp[1] != []]
        bv_to_del0 = [tmp for tmp in bv_to_del0 if tmp[1] != []]

        np.random.shuffle(bv_to_del)
        #print(bv_to_del)
        to_del = int(len(bv_to_del) / 4)
        # 1. loop over all bv from which relw can be deleted
        for rem in bv_to_del[:max(1, min(to_del,
                                         len(bv_to_del) - 1))] + bv_to_del0:
            ll = ''
            # 2. calc line number in INEN where this vector is included
            repl_ind = 7 + 2 * (rem[0])
            # repl_ind = 8
            repl_line = lines_inen[repl_ind]
            repl_ine = []
            #
            for rel_2_del in rem[1]:
                for relnr in range(0, len(repl_line.split())):
                    if int(repl_line.split()[relnr]) == 1:
                        occ = 0
                        for tt in repl_line.split()[:relnr + 1]:
                            occ += int(tt)
                        if occ == rel_2_del + 1:
                            repl_ine.append(relnr)
                break

            ll = ''
            for relnr in range(0, len(repl_line.split())):
                repl = False
                if int(repl_line.split()[relnr]) == 1:
                    for r in repl_ine:
                        if relnr == r:
                            repl = True
                            pass
                    if repl:
                        ll += '  0'
                    else:
                        ll += '%+3s' % repl_line.split()[relnr]
                else:
                    ll += '%+3s' % repl_line.split()[relnr]
            ll += '\n'

            lines_inen[repl_ind] = ll

        s = ''
        for line in lines_inen:
            s += line

        os.system('cp INEN inen_bkp')
        os.system('cp OUTPUT out_bkp')

        with open('INEN', 'w') as outfile:
            outfile.write(s)

        subprocess.run([bin_path + 'DR2END_AK.exe'])

        lines_output = [line for line in open('OUTPUT')]
        for lnr in range(0, len(lines_output)):
            if lines_output[lnr].find('EIGENWERTE DES HAMILTONOPERATORS') >= 0:
                bdg_end = float(lines_output[lnr + 3].split()[ord])
        diff = bdg_end - bdg_ini

        print('%2d: B(4,%d)=%f  dB=%f||' % (nc, basis_size, bdg_end, diff))

        if (diff > max_diff):
            subprocess.call('cp inen_bkp INEN', shell=True)
            subprocess.call('cp out_bkp OUTPUT', shell=True)
        nc = nc + 1

    subprocess.run([bin_path + 'DR2END_AK.exe'])

    lines_output = [line for line in open('OUTPUT')]
    for lnr in range(0, len(lines_output)):
        if lines_output[lnr].find('EIGENWERTE DES HAMILTONOPERATORS') >= 0:
            bdg_end = float(lines_output[lnr + 3].split()[ord])
    for lnr in range(0, len(lines_output)):
        if lines_output[lnr].find('ENTWICKLUNG DES  1 TEN EIGENVEKTORS') >= 0:
            for llnr in range(lnr + 2, len(lines_output)):
                if lines_output[llnr] == '\n':
                    basis_size = int(
                        lines_output[llnr -
                                     1].strip().split('/')[-1][:-1].strip())
                    break
            break
    print(' %d-dim MS: B(4)=%4.3f |' % (basis_size, bdg_end), end='')