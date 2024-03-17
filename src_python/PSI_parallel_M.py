import os
import operator
import shlex

import multiprocessing
from multiprocessing.pool import ThreadPool

from parameters_and_constants import *
from five_particle_functions import *
from four_particle_functions import *
from three_particle_functions import *
from two_particle_functions import *
from functional_assistants import *
from rrgm_functions import *
from scipy.linalg import eigh
from genetic_width_growth import *

MaxProc = int(len(os.sched_getaffinity(0)) / 2)
print('requesting N = %d computing cores.' % MaxProc)
MPIRUN = subprocess.getoutput("which mpirun")
tnni = 11
NEWLINE_SIZE_IN_BYTES = -1
dt = 'float64'


def end2(para, send_end):

    # [widths, sbas, nnpot, Jstreu, binPath, coefstr, civID, minCond, energInt]
    child_id = ''.join(str(x) for x in np.array([para[6]]))

    inqf = 'inq_%s' % child_id
    inenf = 'inen_%s' % child_id

    quaf_to_end = 'quaout_%s' % child_id
    maoutf = 'MATOUTB_%s' % child_id

    outputqf = 'output_qua_%s' % child_id
    outputef = 'endout_%s' % child_id

    # paras: widi,widr,sbas,potNN,potNNN,Jstreu,civ,binPath,coefstr
    # inqua_2(relw=widr, ps2=nnpot)
    inqua_2(relw=para[0], ps2=para[2], inquaout=inqf)
    cmdqua = para[4] + NNhamilEXE_pool + ' %s %s %s' % (inqf, outputqf,
                                                        quaf_to_end)
    #subprocess.run([binPath + 'QUAFL_N.exe'])
    #inen_bdg_2(sbas, j=Jstreu, costr=coefstr)
    nbrop = 14 if bin_suffix == '_v18-uix' else 11
    inen_bdg_2(para[1], j=para[3], costr=para[5], fn=inenf, anzo=nbrop, pari=0)
    #subprocess.run([binPath + 'DR2END_AK.exe'])

    #print(para)

    cmdend = para[4] + spectralEXE_serial_pool + ' %s %s %s %s %s' % (
        quaf_to_end, 'no3bodyfile', inenf, outputef, maoutf)

    pqua = subprocess.Popen(shlex.split(cmdqua),
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE)

    out1, err1 = pqua.communicate()

    pend = subprocess.Popen(shlex.split(cmdend),
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE)
    out3, err3 = pend.communicate()
    #cwd=workdir)
    # <communicate> is needed in order to ensure the process ended before parsing its output!
    try:
        NormHam = np.core.records.fromfile(maoutf, formats='f8', offset=4)
        minCond = para[7]
        smartEV, basCond, smartRAT = smart_ev(NormHam, threshold=minCond)

        anzSigEV = len(
            [bvv for bvv in smartEV if para[8][0] < bvv < para[8][1]])

        EnergySet = [smartEV[ii] for ii in para[9]]
        gsEnergy = smartEV[-1]

        attractiveness = loveliness(EnergySet, basCond, anzSigEV, minCond,
                                    smartRAT)

        subprocess.call('rm -rf ./%s' % inqf, shell=True)
        subprocess.call('rm -rf ./%s' % inenf, shell=True)
        subprocess.call('rm -rf ./%s' % outputef, shell=True)
        subprocess.call('rm -rf ./%s' % outputqf, shell=True)
        subprocess.call('rm -rf ./%s' % quaf_to_end, shell=True)
        subprocess.call('rm -rf ./%s' % maoutf, shell=True)

        #  [ intw,  qualREF, gsREF, basCond ]
        send_end.send([
            para[0],
            attractiveness,
            EnergySet,
            basCond,
        ])

    except:

        subprocess.call('rm -rf ./%s' % inqf, shell=True)
        subprocess.call('rm -rf ./%s' % inenf, shell=True)
        subprocess.call('rm -rf ./%s' % outputef, shell=True)
        subprocess.call('rm -rf ./%s' % quaf_to_end, shell=True)
        subprocess.call('rm -rf ./%s' % maoutf, shell=True)

        print(para[6], child_id)
        print(maoutf)
        #  [ intw,  qual, gsE, basCond ]
        send_end.send([[], 0.0, [0.0], -42.7331])


# new function used in bridgeA2_plus
def span_population2(anz_civ,
                     fragments,
                     Jstreu,
                     coefstr,
                     funcPath,
                     binPath,
                     min_seedE=0.0,
                     mindist=0.1,
                     gridType=['log', 0.005, 0.001],
                     ini_grid_bounds=[0.01, 9.5],
                     ini_dims=8,
                     minC=10**(-8),
                     evWin=[-100, 100],
                     optRange=[-1]):

    os.chdir(funcPath)
    Jstreustring = '%s' % str(Jstreu)[:3]

    lfrags = channels_2[fragments][0]
    sfrags = fragments

    # minimal distance allowed for between width parameters
    rwma = 20

    # lower bound for width parameters '=' IR cutoff (broadest state)
    IRcutoff = 0.00001

    # orbital-angular-momentum dependent upper bound '=' UV cutoff (narrowest state)
    UVcutoff = 951.
    nwrel = ini_dims
    rel_scale = 1.

    ParaSets = []

    for civ in range(anz_civ):
        lit_rw = {}
        he_rw = he_frgs = ob_stru = lu_stru = sbas = []

        #  -- relative widths --------------------------------------------------
        if gridType[0] == 'log':
            lit_w_tmp = np.abs(
                np.geomspace(start=ini_grid_bounds[0],
                             stop=ini_grid_bounds[1],
                             num=nwrel,
                             endpoint=True,
                             dtype=None))
            lit_w_t = [
                test_width * np.random.random() for test_width in lit_w_tmp
            ]

        elif gridType[0] == 'exp':
            lit_w_t, mindist = expspaceS(start=ini_grid_bounds[0],
                                         stop=ini_grid_bounds[1],
                                         num=nwrel,
                                         scal=0.1 + 1.9 * np.random.random())

        elif gridType[0] == 'log_with_density_enhancement':
            wi, wf, lbase = ini_grid_bounds[0], ini_grid_bounds[
                1], 0.01 + 3.98 * np.random.random()
            std_dev = gridType[1]
            lit_w_t = log_with_density_enhancement(wi, wf, lbase, nwrel,
                                                   std_dev)

        else:
            print('\n>>> unspecified grid type.')
            exit()

        lit_w = np.sort(lit_w_t)[::-1]

        lfrags2 = []
        sfrags2 = []
        widr = []

        tmp = np.sort(lit_w)[::-1]
        zer_per_ws = int(np.ceil(len(tmp) / rwma))
        bins = [0 for nmmm in range(zer_per_ws + 1)]
        bins[0] = 0
        for mn in range(len(tmp)):
            bins[1 + mn % zer_per_ws] += 1
        bnds = np.cumsum(bins)
        tmp2 = [list(tmp[bnds[nn]:bnds[nn + 1]]) for nn in range(zer_per_ws)]
        sfrags2 = len(tmp2) * [sfrags]
        lfrags2 = len(tmp2) * [lfrags]
        widr = tmp2

        sbas = []
        bv = 0
        for n in range(len(widr)):
            sbas += [[
                two_body_channels[fragments] + bv,
                [1 for x in range(1, 1 + len(widr[n]))]
            ]]
            bv += len(two_body_channels)

        # [widths, sbas, nnpot, Jstreu, binPath, coefstr, civID, minCond, energInt]
        ParaSets.append([
            widr, sbas, nnpotstring, Jstreu, binPath, coefstr, civ, minC,
            evWin, optRange
        ])

    os.chdir(funcPath)

    inlu_2(anzo=8, anzf=len(widr))
    os.system(binPath + NNorbitalEXE)
    inob_2(anzo=8, anzf=len(widr))
    os.system(binPath + NNspinEXE)
    samp_list = []
    cand_list = []
    pool = ThreadPool(max(min(MaxProc, len(ParaSets)), 2))
    jobs = []
    for procnbr in range(len(ParaSets)):
        recv_end, send_end = multiprocessing.Pipe(False)
        pars = ParaSets[procnbr]
        p = multiprocessing.Process(target=end2, args=(pars, send_end))
        jobs.append(p)

        # sen_end returns [ intw,  qualREF, gsREF, basCond ]
        samp_list.append(recv_end)
        p.start()
    for proc in jobs:
        proc.join()

    samp_ladder = [x.recv() for x in samp_list]

    #for cc in samp_ladder:
    #    print(cc[2:])

    for cand in samp_ladder:
        if ((cand[2][-1] < min_seedE) & (cand[3] > minC)):
            cfgg = np.transpose(np.array([sfrags2, lfrags2])).tolist()

            cand_list.append([cfgg] + cand)

    cand_list.sort(key=lambda tup: np.abs(tup[2]))

    return cand_list, sbas


# deprecated function used in outdated bridgeA2
def span_initial_basis2(channel,
                        Jstreu,
                        coefstr,
                        funcPath,
                        binPath,
                        ini_grid_bounds=[0.01, 9.5],
                        ini_dims=20,
                        mindist=0.02):

    os.chdir(funcPath)

    Jstreustring = '%s' % str(Jstreu)[:3]

    lit_w = {}

    he_iw = he_rw = he_frgs = ob_stru = lu_stru = sbas = []

    # minimal distance allowed for between width parameters
    rwma = 20
    bvma = 20

    # lower bound for width parameters '=' IR cutoff (broadest state)
    rWmin = 0.0001
    # orbital-angular-momentum dependent upper bound '=' UV cutoff (narrowest state)
    iLcutoff = 420.

    wi, wf, nw = ini_grid_bounds[0], ini_grid_bounds[1], ini_dims

    if ini_dims > rwma:
        print(
            'The set number for relative width parameters per basis vector > max!'
        )
        exit()

    wii = wi
    wff = wf

    itera = 1

    lit_w = [wii + np.random.random() * (wff - wii)]

    while len(lit_w) != ini_dims:

        lit_w_tmp = wii + np.random.random() * (wff - wii)
        prox_check = check_dist(width_array1=lit_w, minDist=mindist)
        prox_checkr = np.all([
            check_dist(width_array1=lit_w, width_array2=wsr, minDist=mindist)
            for wsr in widthSet_relative
        ])

        if ((prox_check * prox_checkr) & (lit_w_tmp < iLcutoff)):
            lit_w.append(lit_w_tmp)
        #lit_w_tmp = np.abs(
        #    np.geomspace(start=wii,
        #                 stop=wff,
        #                 num=nw[frg],
        #                 endpoint=True,
        #                 dtype=None))

        #lit_w = sparse_subset(lit_w_tmp, mindist)
        #lit_w = sparse(lit_w_tmp, mindist=mindist)
        #lit_w = [ww for ww in lit_w if rWmin < ww < iLcutoff[0]]

        itera += 1
        assert itera <= 180000

    lit_w = np.sort(lit_w)[::-1]

    widi = []

    zer_per_ws = int(np.ceil(len(lit_w) / bvma))
    bins = [0 for nmmm in range(zer_per_ws + 1)]
    bins[0] = 0
    for mn in range(len(lit_w)):
        bins[1 + mn % zer_per_ws] += 1
    bnds = np.cumsum(bins)
    tmp2 = [list(lit_w[bnds[nn]:bnds[nn + 1]]) for nn in range(zer_per_ws)]

    frags = len(tmp2) * [channel]

    widi = tmp2

    anzBV = sum([len(zer) for zer in widi])

    sbas = []
    bv = two_body_channels[channel][0]
    for n in range(len(frags)):
        sbas += [[bv, [x for x in range(1, 1 + len(widi[n]))]]]
        bv += 13

    os.chdir(funcPath)

    inlu_2(anzo=8, anzf=len(frags))
    os.system(binPath + NNorbitalEXE)
    inob_2(anzo=8, anzf=len(frags))
    os.system(binPath + NNspinEXE)

    #print(widi)
    inqua_2(relw=widi, ps2=nnpotstring)
    #exit()
    subprocess.run([binPath + NNhamilEXE_serial])

    inen_bdg_2(sbas, j=Jstreu, costr=coefstr)

    subprocess.call('cp -rf INQUA_N INQUA_N_V18', shell=True)

    subprocess.run([binPath + spectralEXE_serial])

    suche_fehler()

    matout = np.core.records.fromfile('MATOUTB', formats='f8', offset=4)

    path_relw = funcPath + '/relw.dat'

    if os.path.exists(path_relw): os.remove(path_relw)
    with open(path_relw, 'wb') as f:
        for wss in widi:
            for ws in wss:
                np.savetxt(f, [ws], fmt='%12.6f', delimiter=' ')
        f.seek(NEWLINE_SIZE_IN_BYTES, 2)
        f.truncate()
    f.close()
    #    # write basis structure on tape
    #    exit()
    #
    #    path_bas_dims = funcPath + '/bas_dims.dat'
    #    with open(path_bas_dims, 'wb') as f:
    #        np.savetxt(f, [np.size(wid) for wid in widr], fmt='%d')
    #        f.seek(NEWLINE_SIZE_IN_BYTES, 2)
    #        f.truncate()
    #    f.close()
    #    path_bas_int_rel_pairs = funcPath + '/bas_full.dat'
    #    if os.path.exists(path_bas_int_rel_pairs):
    #        os.remove(path_bas_int_rel_pairs)
    #    with open(path_bas_int_rel_pairs, 'w') as oof:
    #        #np.savetxt(f, [[jj[0], kk] for jj in sbas for kk in jj[1]], fmt='%d')
    #        #f.seek(NEWLINE_SIZE_IN_BYTES, 2)
    #        #f.truncate()
    #        so = ''
    #        for bv in sbas:
    #            so += '%4s' % str(bv[0])
    #            for rww in bv[1]:
    #                so += '%4s' % str(rww)
    #            so += '\n'
    #        oof.write(so)
    #    oof.close()
    #    path_frag_stru = funcPath + '/frags.dat'
    #    if os.path.exists(path_frag_stru): os.remove(path_frag_stru)
    #
    #    with open(path_frag_stru, 'wb') as f:
    #        np.savetxt(f,
    #                   np.column_stack([sfrags2, lfrags2]),
    #                   fmt='%s',
    #                   delimiter=' ',
    #                   newline=os.linesep)
    #        f.seek(NEWLINE_SIZE_IN_BYTES, 2)
    #        f.truncate()
    #    f.close()
    #    path_intw = funcPath + '/intw.dat'
    #    if os.path.exists(path_intw): os.remove(path_intw)
    #    with open(path_intw, 'wb') as f:
    #        for ws in widi:
    #            np.savetxt(f, [ws], fmt='%12.6f', delimiter=' ')
    #        f.seek(NEWLINE_SIZE_IN_BYTES, 2)
    #        f.truncate()
    #    f.close()

    return matout


def end3(para, send_end):

    maxirat = 10**10
    # [widi, widr, sbas, nnpot, nnnpot, Jstreu, civ, binPath, coefstr]

    child_id = ''.join(str(x) for x in np.array([para[6]]))

    inqf = 'inq_%s' % child_id
    indqf = 'indq_%s' % child_id
    inenf = 'inen_%s' % child_id

    quaf_to_end = 'quaout_%s' % child_id
    drquaf_to_end = 'drquaout_%s' % child_id
    maoutf = 'MATOUTB_%s' % child_id

    outputqf = 'output_qua_%s' % child_id
    outputdrqf = 'output_drq_%s' % child_id
    outputef = 'endout_%s' % child_id

    # paras: widi,widr,sbas,potNN,potNNN,Jstreu,civ,binPath,coefstr

    inen_bdg_3(para[2], para[5], para[8], fn=inenf, pari=0, nzop=para[11])

    inqua_3(intwi=para[0], relwi=para[1], potf=para[3], inquaout=inqf)
    inqua_3(intwi=para[0], relwi=para[1], potf=para[4], inquaout=indqf)

    cmdqua = para[7] + NNhamilEXE_pool + ' %s %s %s' % (inqf, outputqf,
                                                        quaf_to_end)
    cmddrqua = para[7] + NNNhamilEXE_pool + ' %s %s %s' % (indqf, outputdrqf,
                                                           drquaf_to_end)

    cmdend = para[7] + spectralEXE_serial_pool + ' %s %s %s %s %s' % (
        quaf_to_end, drquaf_to_end, inenf, outputef, maoutf)

    pqua = subprocess.Popen(shlex.split(cmdqua),
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE)

    out1, err1 = pqua.communicate()
    pdrqua = subprocess.Popen(shlex.split(cmddrqua),
                              stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE)

    out2, err2 = pdrqua.communicate()
    pend = subprocess.Popen(shlex.split(cmdend),
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE)
    out3, err3 = pend.communicate()
    #cwd=workdir)
    # <communicate> is needed in order to ensure the process ended before parsing its output!
    try:
        NormHam = np.core.records.fromfile(maoutf, formats='f8', offset=4)
        minCond = para[9]
        smartEV, basCond, smartRAT = smart_ev(NormHam, threshold=minCond)
        anzSigEV = len(
            [bvv for bvv in smartEV if para[10][0] < bvv < para[10][1]])

        EnergySet = [smartEV[ii] for ii in para[12]]
        gsEnergy = smartEV[-1]

        attractiveness = loveliness(EnergySet, basCond, anzSigEV, minCond,
                                    smartRAT, maxirat)

        os.system('rm -rf ./%s' % inqf)
        os.system('rm -rf ./%s' % indqf)
        os.system('rm -rf ./%s' % inenf)
        os.system('rm -rf ./%s' % outputef)
        os.system('rm -rf ./%s' % outputqf)
        os.system('rm -rf ./%s' % outputdrqf)
        os.system('rm -rf ./%s' % quaf_to_end)
        os.system('rm -rf ./%s' % drquaf_to_end)
        os.system('rm -rf ./%s' % maoutf)

        #  [ intw, relw, qualREF, gsREF, basCond ]
        send_end.send([
            [para[0], para[1]],
            attractiveness,
            EnergySet,
            basCond,
        ])

    except:

        os.system('rm -rf ./%s' % inqf)
        os.system('rm -rf ./%s' % indqf)
        os.system('rm -rf ./%s' % inenf)
        os.system('rm -rf ./%s' % outputef)
        os.system('rm -rf ./%s' % outputqf)
        os.system('rm -rf ./%s' % outputdrqf)
        os.system('rm -rf ./%s' % quaf_to_end)
        os.system('rm -rf ./%s' % drquaf_to_end)
        os.system('rm -rf ./%s' % maoutf)

        print(para[6], child_id)
        print(maoutf)
        #  [ intw, relw, qual, gsE, basCond ]
        send_end.send([[[], []], 0.0, [0.0], -42.7331])


def span_population3(anz_civ,
                     fragments,
                     Jstreu,
                     coefstr,
                     funcPath,
                     binPath,
                     nzo=31,
                     mindists=1.0,
                     ini_grid_bounds=[0.01, 9.5, 0.001, 11.5],
                     ini_dims=[4, 4],
                     gridType=['log', 0.005, 0.005],
                     minC=10**(-8),
                     evWin=[-100, 100],
                     optRange=[-1]):

    os.chdir(funcPath)

    Jstreustring = '%s' % str(Jstreu)[:3]

    lfrags = []
    sfrags = []

    for lcfg in range(len(fragments)):
        sfrags = sfrags + fragments[lcfg][1]
        for scfg in fragments[lcfg][1]:
            lfrags = lfrags + [fragments[lcfg][0]]

    # minimal distance allowed for between width parameters
    rwma = 20
    bvma = 8
    mindist_int = mindists
    mindist_rel = mindists
    # lower bound for width parameters '=' IR cutoff (broadest state)
    rWmin = 0.00001

    # orbital-angular-momentum dependent upper bound '=' UV cutoff (narrowest state)
    iLcutoff = 92.
    rLcutoff = 92.
    nwint = ini_dims[0]
    nwrel = ini_dims[1]
    rel_scale = 1.

    if nwrel > rwma:
        print(
            'The set number for relative width parameters per basis vector > max!'
        )
        exit()

    ParaSets = []

    for civ in range(anz_civ):
        lit_w = {}
        lit_rw = {}
        he_iw = he_rw = he_frgs = ob_stru = lu_stru = sbas = []

        assert len(lfrags) % 2 == 0

        enforceSym = 2 if bin_suffix == 'expl' else 1

        for frg in range(int(len(lfrags) / enforceSym)):

            itera = 1
            #  -- internal widths --------------------------------------------------
            if gridType[0] == 'log':
                lit_w_tmp = np.abs(
                    np.geomspace(start=ini_grid_bounds[0],
                                 stop=ini_grid_bounds[1],
                                 num=nwint,
                                 endpoint=True,
                                 dtype=None))

            elif gridType[0] == 'exp':
                lit_w_tmp, mindist = expspaceS(start=ini_grid_bounds[0],
                                               stop=ini_grid_bounds[1],
                                               num=nwint,
                                               scal=0.1 +
                                               1.9 * np.random.random())

            elif gridType[0] == 'log_with_density_enhancement':
                wi, wf, lbase = ini_grid_bounds[0], ini_grid_bounds[
                    1], 0.01 + 3.98 * np.random.random()
                std_dev = gridType[1]
                lit_w_tmp = log_with_density_enhancement(
                    wi, wf, lbase, nwint, std_dev)

            else:
                print('\n>>> unspecified grid type.')
                exit()

            lit_w_t = []

            while len(lit_w_t) != nwint:

                lit_wi = [
                    test_width * np.random.random() for test_width in lit_w_tmp
                ]
                if np.max(lit_wi) > iLcutoff:
                    continue

                prox_check = check_dist(width_array1=lit_wi,
                                        minDist=mindist_int)

                prox_checkr = np.all([
                    check_dist(width_array1=lit_wi,
                               width_array2=wsr,
                               minDist=mindist_rel)
                    for wsr in widthSet_relative
                ])

                if (prox_check == prox_checkr == False):
                    lit_w_t = lit_wi

                itera += 1
                assert itera <= 180000

            lit_w[frg] = np.sort(lit_w_t)[::-1]

            #  -- relative widths --------------------------------------------------
            if gridType[0] == 'log':
                lit_w_tmp = np.abs(
                    np.geomspace(start=ini_grid_bounds[2],
                                 stop=ini_grid_bounds[3],
                                 num=nwrel,
                                 endpoint=True,
                                 dtype=None))

            elif gridType[0] == 'exp':
                lit_w_tmp, mindist = expspaceS(start=ini_grid_bounds[2],
                                               stop=ini_grid_bounds[3],
                                               num=nwrel,
                                               scal=0.1 +
                                               1.9 * np.random.random())

            elif gridType[0] == 'log_with_density_enhancement':
                wi, wf, lbase = ini_grid_bounds[2], ini_grid_bounds[
                    3], 0.01 + 3.98 * np.random.random()
                std_dev = gridType[2]
                lit_w_tmp = log_with_density_enhancement(
                    wi, wf, lbase, nwrel, std_dev)

            else:
                print('\n>>> unspecified grid type.')
                exit()

            lit_w_t = []

            while len(lit_w_t) != nwrel:

                lit_wr = [
                    test_width * np.random.random() for test_width in lit_w_tmp
                ]
                if np.max(lit_wr) > rLcutoff:
                    continue

                prox_check = check_dist(width_array1=lit_wr,
                                        minDist=mindist_int)
                prox_checkr = np.all([
                    check_dist(width_array1=lit_wr,
                               width_array2=wsr,
                               minDist=mindist_rel)
                    for wsr in widthSet_relative
                ])

                if (prox_check == prox_checkr == False):
                    lit_w_t = lit_wr

                itera += 1
                assert itera <= 180000

            lit_rw[frg] = np.sort(lit_w_t)[::-1]

        if enforceSym == 2:
            for nfrag in range(int(len(lfrags) / 2)):
                lit_w[int(len(lfrags) / 2 + nfrag)] = lit_w[nfrag]
                lit_rw[int(len(lfrags) / 2 + nfrag)] = lit_rw[nfrag]

        lfrags2 = []
        sfrags2 = []
        widi = []
        widr = []
        for n in range(len(lit_w)):
            tmp = np.sort(lit_w[n])[::-1]
            zer_per_ws = int(np.ceil(len(tmp) / bvma))
            bins = [0 for nmmm in range(zer_per_ws + 1)]
            bins[0] = 0
            for mn in range(len(tmp)):
                bins[1 + mn % zer_per_ws] += 1
            bnds = np.cumsum(bins)
            tmp2 = [
                list(tmp[bnds[nn]:bnds[nn + 1]]) for nn in range(zer_per_ws)
            ]
            tmp3 = [list(lit_rw[n]) for nn in range(zer_per_ws)]
            sfrags2 += len(tmp2) * [sfrags[n]]
            lfrags2 += len(tmp2) * [lfrags[n]]
            widi += tmp2
            widr += tmp3

        anzBV = sum([len(zer) for zer in widi])

        sbas = []
        bv = 1
        for n in range(len(lfrags2)):
            off = np.mod(n, 2)
            for m in range(len(widi[n])):
                sbas += [[
                    bv, [x for x in range(1 + off, 1 + len(widr[n]), 2)]
                ]]
                bv += 1
        ParaSets.append([
            widi, widr, sbas, nnpotstring, nnnpotstring, Jstreu, civ, binPath,
            coefstr, minC, evWin, nzo, optRange
        ])

    assert len(ParaSets) == anz_civ
    os.chdir(funcPath)

    inlu_3(8, fn='INLU', fr=lfrags2, indep=0)
    os.system(binPath + NNNorbitalEXE)
    inlu_3(8, fn='INLUCN', fr=lfrags2, indep=0)
    os.system(binPath + NNorbitalEXE)
    inob_3(sfrags2, 8, fn='INOB', indep=0)
    os.system(binPath + NNspinEXE)
    inob_3(sfrags2, 15, fn='INOB', indep=0)
    os.system(binPath + NNNspinEXE)

    samp_list = []
    cand_list = []
    pool = ThreadPool(max(min(MaxProc, len(ParaSets)), 2))
    jobs = []
    for procnbr in range(len(ParaSets)):
        recv_end, send_end = multiprocessing.Pipe(False)
        pars = ParaSets[procnbr]
        p = multiprocessing.Process(target=end3, args=(pars, send_end))
        jobs.append(p)

        # sen_end returns [ intw, relw, qualREF, gsREF, basCond ]
        samp_list.append(recv_end)
        p.start()
    for proc in jobs:
        proc.join()

    samp_ladder = [x.recv() for x in samp_list]

    for cand in samp_ladder:
        # admit candidate basis as soon as the smallest EV is <0
        if ((cand[2][-1] < 0) &
                # admit the basis only if the smallest N EVs (as def. by optRange) are <0
                #if ((np.all(np.less(cand[2], np.zeros(len(cand[2]))))) &
            (cand[3] > minC)):
            cfgg = np.transpose(np.array([sfrags2, lfrags2],
                                         dtype=object)).tolist()

            cand_list.append([cfgg] + cand)

    cand_list.sort(key=lambda tup: np.abs(tup[2]))

    #for cc in samp_ladder:
    #    print(cc[2:])

    return cand_list, sbas


def prepare_einzel3(funcPath, binPath):

    if os.path.isdir(funcPath + '/eob') == False:
        subprocess.check_call(['mkdir', '-p', funcPath + '/eob'])
        os.chdir(funcPath + '/eob')
        inob_3([
            'he_no1',
            'he_no1y',
            'he_no2',
            'he_no2y',
            'he_no3',
            'he_no3y',
            'he_no5',
            'he_no5y',
            'he_no6',
            'he_no6y',
            't_no1',
            't_no2',
            't_no6',
        ],
               8,
               fn='INOB',
               indep=+1)
        os.system(binPath + NNspinEXE)

    if os.path.isdir(funcPath + '/eob-tni') == False:
        subprocess.check_call(['mkdir', '-p', funcPath + '/eob-tni'])
        os.chdir(funcPath + '/eob-tni')
        inob_3([
            'he_no1',
            'he_no1y',
            'he_no2',
            'he_no2y',
            'he_no3',
            'he_no3y',
            'he_no5',
            'he_no5y',
            'he_no6',
            'he_no6y',
            't_no1',
            't_no2',
            't_no6',
        ],
               15,
               fn='INOB',
               indep=+1)
        os.system(binPath + NNNspinEXE)

    if os.path.isdir(funcPath + '/elu') == False:
        subprocess.check_call(['mkdir', '-p', funcPath + '/elu'])
        os.chdir(funcPath + '/elu')
        inlu_3(8,
               fn='INLUCN',
               fr=[
                   '000',
                   '202',
                   '022',
                   '110',
                   '101',
                   '011',
                   '111',
                   '112',
                   '211',
                   '212',
                   '213',
                   '123',
                   '121',
                   '122',
                   '212',
                   '222',
                   '221',
                   '220',
               ],
               indep=+1)
        os.system(binPath + NNorbitalEXE)

    if os.path.isdir(funcPath + '/elu-tni') == False:
        subprocess.check_call(['mkdir', '-p', funcPath + '/elu-tni'])
        os.chdir(funcPath + '/elu-tni')
        inlu_3(8,
               fn='INLU',
               fr=[
                   '000',
                   '202',
                   '022',
                   '110',
                   '101',
                   '011',
                   '111',
                   '112',
                   '211',
                   '212',
                   '213',
                   '123',
                   '121',
                   '122',
                   '212',
                   '222',
                   '221',
                   '220',
               ],
               indep=+1)
        os.system(binPath + NNNorbitalEXE)


def prepare_einzel4(funcPath, binPath, channels):

    frgsS = [ss for ch in channels for ss in ch[1]]
    frgsL = [ss for ch in channels for ss in ch[0]]

    if os.path.isdir(funcPath + '/eob') == False:
        subprocess.check_call(['mkdir', '-p', funcPath + '/eob'])
    os.chdir(funcPath + '/eob')
    inob_4(frgsS, 8, fn='INOB', indep=+1)
    os.system(binPath + NNspinEXE)

    if os.path.isdir(funcPath + '/eob-tni') == False:
        subprocess.check_call(['mkdir', '-p', funcPath + '/eob-tni'])
    os.chdir(funcPath + '/eob-tni')
    inob_4(frgsS, 15, fn='INOB', indep=+1)
    os.system(binPath + NNNspinEXE)

    if os.path.isdir(funcPath + '/elu') == False:
        subprocess.check_call(['mkdir', '-p', funcPath + '/elu'])
    os.chdir(funcPath + '/elu')
    inlu_4(8, fn='INLUCN', fr=frgsL, indep=+1)
    os.system(binPath + NNorbitalEXE)

    if os.path.isdir(funcPath + '/elu-tni') == False:
        subprocess.check_call(['mkdir', '-p', funcPath + '/elu-tni'])
    os.chdir(funcPath + '/elu-tni')
    inlu_4(8, fn='INLU', fr=frgsL, indep=+1)
    os.system(binPath + NNNorbitalEXE)


def prepare_einzel5(funcPath, binPath, channels):

    frgsS = [ss for ch in channels for ss in ch[1]]
    frgsL = [ss for ch in channels for ss in ch[0]]

    if os.path.isdir(funcPath + '/eob') == False:
        subprocess.check_call(['mkdir', '-p', funcPath + '/eob'])
    os.chdir(funcPath + '/eob')
    inob_5(frgsS, 8, fn='INOB', indep=+1)
    os.system(binPath + NNspinEXE)

    if os.path.isdir(funcPath + '/eob-tni') == False:
        subprocess.check_call(['mkdir', '-p', funcPath + '/eob-tni'])
    os.chdir(funcPath + '/eob-tni')
    inob_5(frgsS, 15, fn='INOB', indep=+1)
    os.system(binPath + NNNspinEXE)

    if os.path.isdir(funcPath + '/elu') == False:
        subprocess.check_call(['mkdir', '-p', funcPath + '/elu'])
    os.chdir(funcPath + '/elu')
    inlu_5(8, fn='INLUCN', fr=frgsL, indep=+1)
    os.system(binPath + NNorbitalEXE)

    if os.path.isdir(funcPath + '/elu-tni') == False:
        subprocess.check_call(['mkdir', '-p', funcPath + '/elu-tni'])
    os.chdir(funcPath + '/elu-tni')
    inlu_5(8, fn='INLU', fr=frgsL, indep=+1)
    os.system(binPath + NNNorbitalEXE)


def end4(para, send_end):

    # [widi, widr, sbas, nnpot, nnnpot, Jstreu, civ, binPath, coefstr]

    # nOpt: 0: ground-state optimization
    #       n: n-th excited-state energy is minimized
    nOpt = 0

    child_id = ''.join(str(x) for x in np.array([para[6]]))

    inqf = 'inq_%s' % child_id
    indqf = 'indq_%s' % child_id
    inenf = 'inen_%s' % child_id

    quaf_to_end = 'quaout_%s' % child_id
    drquaf_to_end = 'drquaout_%s' % child_id
    maoutf = 'MATOUTB_%s' % child_id

    outputqf = 'output_qua_%s' % child_id
    outputdrqf = 'output_drq_%s' % child_id
    outputef = 'endout_%s' % child_id

    # paras: widi,widr,sbas,potNN,potNNN,Jstreu,civ,binPath,coefstr

    inen_bdg_4(para[2], para[5], para[8], fn=inenf, pari=0, nzop=para[12])

    inqua_4(intwi=para[0], relwi=para[1], potf=para[3], inquaout=inqf)

    cmdqua = para[7] + NNhamilEXE_pool + ' %s %s %s' % (inqf, outputqf,
                                                        quaf_to_end)

    cmdend = para[7] + spectralEXE_serial_pool + ' %s %s %s %s %s' % (
        quaf_to_end, drquaf_to_end, inenf, outputef, maoutf)

    pqua = subprocess.Popen(shlex.split(cmdqua),
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE)

    out1, err1 = pqua.communicate()

    if para[12] >= 14:
        inqua_4(intwi=para[0], relwi=para[1], potf=para[4], inquaout=indqf)
        cmddrqua = para[7] + NNNhamilEXE_pool + ' %s %s %s' % (
            indqf, outputdrqf, drquaf_to_end)

        pdrqua = subprocess.Popen(shlex.split(cmddrqua),
                                  stdout=subprocess.PIPE,
                                  stderr=subprocess.PIPE)
        out2, err2 = pdrqua.communicate()

    pend = subprocess.Popen(shlex.split(cmdend),
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE)
    out3, err3 = pend.communicate()

    # <communicate> is needed in order to ensure the process ended before parsing its output!
    try:
        NormHam = np.core.records.fromfile(maoutf, formats='f8', offset=4)
        minCond = para[9]
        maxRa = para[11]
        smartEV, basCond, smartRAT = smart_ev(NormHam, threshold=minCond)
        anzSigEV = len(
            [bvv for bvv in smartEV if para[10][0] < bvv < para[10][1]])

        EnergySet = [smartEV[ii] for ii in para[13]]
        gsEnergy = smartEV[para[13][0]]

        attractiveness = loveliness(EnergySet, basCond, anzSigEV, minCond,
                                    smartRAT, maxRa) if basCond > 0 else -2.0

        os.system('rm -rf ./%s' % inqf)
        os.system('rm -rf ./%s' % indqf)
        os.system('rm -rf ./%s' % inenf)
        os.system('rm -rf ./%s' % outputef)
        os.system('rm -rf ./%s' % outputqf)
        os.system('rm -rf ./%s' % outputdrqf)
        os.system('rm -rf ./%s' % quaf_to_end)
        os.system('rm -rf ./%s' % drquaf_to_end)
        os.system('rm -rf ./%s' % maoutf)

        #  [ [intw, relw], qualREF, gsREF, basCond ]
        #print('%12.4e%12.4f%12.4e' % (attractiveness, gsEnergy, basCond))
        send_end.send([
            [para[0], para[1]],
            attractiveness,
            EnergySet,
            basCond,
        ])

    except:

        os.system('rm -rf ./%s' % inqf)
        os.system('rm -rf ./%s' % indqf)
        os.system('rm -rf ./%s' % inenf)
        os.system('rm -rf ./%s' % outputef)
        os.system('rm -rf ./%s' % outputqf)
        os.system('rm -rf ./%s' % outputdrqf)
        os.system('rm -rf ./%s' % quaf_to_end)
        os.system('rm -rf ./%s' % drquaf_to_end)
        os.system('rm -rf ./%s' % maoutf)

        print(para[6], child_id)
        print(maoutf)
        #  [ intw, relw, qual, gsE, basCond ]
        send_end.send([[[], []], 0.0, [0.0], -42.7331])


def span_population4(anz_civ,
                     fragments,
                     Jstreu,
                     coefstr,
                     funcPath,
                     binPath,
                     nzo=31,
                     mindists=0.001,
                     ini_grid_bounds=[0.01, 9.5, 0.001, 11.5],
                     ini_dims=[4, 4],
                     gridType=['log', 0.005, 0.005],
                     minC=10**(-8),
                     maxR=10**5,
                     evWin=[-100, 100],
                     optRange=[-1]):

    os.chdir(funcPath)

    Jstreustring = '%s' % str(Jstreu)[:3]

    lfrags = []
    sfrags = []

    for lcfg in range(len(fragments)):
        sfrags = sfrags + fragments[lcfg][1]
        for scfg in fragments[lcfg][1]:
            lfrags = lfrags + [fragments[lcfg][0]]

    # minimal distance allowed for between width parameters
    rwma = 30
    bvma = 8
    mindist_int = mindists
    mindist_rel = mindists
    # lower bound for width parameters '=' IR cutoff (broadest state)
    rWmin = 0.0001

    # orbital-angular-momentum dependent upper bound '=' UV cutoff (narrowest state)
    iLcutoff = 192.
    rLcutoff = 192.
    nwint = ini_dims[0]
    nwrel = ini_dims[1]
    rel_scale = 1.

    if nwrel > rwma:
        print(
            'The set number for relative width parameters per basis vector > max!'
        )
        exit()
    ParaSets = []

    for civ in range(anz_civ):
        lit_w = {}
        lit_rw = {}
        he_iw = he_rw = he_frgs = ob_stru = lu_stru = sbas = []

        for frg in range(len(lfrags)):

            #  -- internal widths --------------------------------------------------

            if gridType[0] == 'log':
                wi, wf, lbase = ini_grid_bounds[0], ini_grid_bounds[1], (
                    0.01 + 3.98 * np.random.random())
                sta = np.log(wi) / np.log(lbase)
                sto = np.log(wf) / np.log(lbase)
                lit_w_tmp = np.abs(
                    np.logspace(start=sta,
                                stop=sto,
                                base=lbase,
                                num=int(2 * nwint),
                                endpoint=True,
                                dtype=None))

            elif gridType[0] == 'exp':
                wi, wf, scale = ini_grid_bounds[0], ini_grid_bounds[1], (
                    0.01 + 0.9 * np.random.random())
                lit_w_tmp, mindist_int = expspaceS(start=wi,
                                                   stop=wf,
                                                   scal=scale,
                                                   num=int(2 * nwint),
                                                   deltam=mindist_int)
                #print(wi, wf, scale)
                #print(lit_w_tmp)
                #exit()

            elif gridType[0] == 'log_with_density_enhancement':
                wi, wf, lbase = ini_grid_bounds[0], ini_grid_bounds[
                    1], 0.01 + 3.98 * np.random.random()
                std_dev = gridType[1]
                lit_w_tmp = log_with_density_enhancement(
                    wi, wf, lbase, int(2 * nwint), std_dev)

            else:
                print('\n>>> unspecified grid type.')
                exit()

            itera = 1
            lit_w_t = []
            while lit_w_t == []:

                lit_wi = [
                    test_width * (0.15 + 0.86 * (1 - np.random.random()))
                    for test_width in lit_w_tmp
                ]
                if np.max(lit_wi) > iLcutoff:
                    continue

                tooCloseInternal = False  #check_dist(width_array1=lit_wi,minDist=mindist_int)
                if tooCloseInternal == False:
                    lit_w_t = lit_wi
                #print(lit_wi, '\nNotSoClose = ', tooCloseInternal)
                #exit()

                itera += 1
                assert itera <= 180

            # do not sort in order to allow for narrow/broad width combinations
            #lit_w[frg] = np.sort(lit_w_t)[::-1]
            #print(lit_w_t)
            np.random.shuffle(lit_w_t)
            #print(lit_w_t)
            #exit()
            lit_w[frg] = lit_w_t

            #  -- relative widths --------------------------------------------------

            if gridType[0] == 'log':
                wi, wf, lbase = ini_grid_bounds[2], ini_grid_bounds[3], (
                    0.01 + 3.98 * np.random.random())
                sta = np.log(wi) / np.log(lbase)
                sto = np.log(wf) / np.log(lbase)
                lit_w_tmp = abs(
                    np.logspace(start=sta,
                                stop=sto,
                                base=lbase,
                                num=nwrel,
                                endpoint=True,
                                dtype=None))

            elif gridType[0] == 'exp':
                wi, wf, scale = ini_grid_bounds[2], ini_grid_bounds[3], (
                    0.01 + 0.9 * np.random.random())
                lit_w_tmp, mindist_rel = expspaceS(start=wi,
                                                   stop=wf,
                                                   scal=scale,
                                                   num=nwrel,
                                                   deltam=mindist_rel)

            elif gridType[0] == 'log_with_density_enhancement':
                wi, wf, lbase = ini_grid_bounds[0], ini_grid_bounds[
                    1], 0.01 + 3.98 * np.random.random()
                std_dev = gridType[2]
                lit_w_tmp = log_with_density_enhancement(
                    wi, wf, lbase, nwrel, std_dev)

            else:
                print('\n>>> unspecified grid type.')
                exit()

            lit_w_r = []
            while lit_w_r == []:

                lit_wi = [
                    test_width * (1 - np.random.random())
                    for test_width in lit_w_tmp
                ]
                if np.max(lit_wi) > rLcutoff:
                    continue

                tooCloseRelative = check_dist(width_array1=lit_wi,
                                              minDist=mindist_rel)

                if tooCloseRelative == False:
                    lit_w_r = lit_wi

                itera += 1
                assert itera <= 180

            lit_rw[frg] = np.sort(lit_w_r)[::-1]

        lfrags2 = []
        sfrags2 = []
        widi = []
        widr = []

        for n in range(len(lit_w)):
            tmp = lit_w[n]  #np.sort(lit_w[n])[::-1]
            assert len(tmp) % 2 == 0

            modlist = [m % bvma for m in range(int(0.5 * len(tmp)))]

            bnds = [2 * nn for nn in range(len(modlist)) if modlist[nn] == 0]

            bnds.append(len(tmp))
            tmp2 = [
                list(tmp[bnds[nn]:bnds[nn + 1]])
                for nn in range(len(bnds) - 1)
            ]

            tmp3 = [list(lit_rw[n]) for nn in range(len(bnds) - 1)]

            sfrags2 += len(tmp2) * [sfrags[n]]
            lfrags2 += len(tmp2) * [lfrags[n]]
            widi += tmp2
            widr += tmp3

        anzBV = sum([len(zer) for zer in widi])

        sbas = []
        bv = 1
        for n in range(len(lfrags2)):
            off = np.mod(n, 2)
            assert len(widi[n]) % 2 == 0
            for m in range(int(len(widi[n]) / 2)):
                sbas += [[
                    bv, [np.mod(off + x, 2) for x in range(len(widr[n]))]
                ]]
                bv += 1
        ParaSets.append([
            widi, widr, sbas, nnpotstring, nnnpotstring, Jstreu, civ, binPath,
            coefstr, minC, evWin, maxR, nzo, optRange
        ])

    assert len(ParaSets) == anz_civ
    os.chdir(funcPath)

    inlu_4(8, fn='INLU', fr=sum(lfrags2, []), indep=0)
    os.system(binPath + NNNorbitalEXE)
    inlu_4(8, fn='INLUCN', fr=sum(lfrags2, []), indep=0)
    os.system(binPath + NNorbitalEXE)

    inob_4(sfrags2, 8, fn='INOB', indep=0)
    os.system(binPath + NNspinEXE)
    inob_4(sfrags2, 15, fn='INOB', indep=0)
    os.system(binPath + NNNspinEXE)

    samp_list = []
    cand_list = []
    pool = ThreadPool(max(min(MaxProc, len(ParaSets)), 2))

    jobs = []
    for procnbr in range(len(ParaSets)):
        recv_end, send_end = multiprocessing.Pipe(False)
        pars = ParaSets[procnbr]
        p = multiprocessing.Process(target=end4, args=(pars, send_end))
        jobs.append(p)

        # sen_end returns [ intw, relw, qualREF, gsREF, basCond ]
        samp_list.append(recv_end)
        p.start()
    for proc in jobs:
        proc.join()

    samp_ladder = [x.recv() for x in samp_list]

    for cand in samp_ladder:
        if (np.all(np.less(cand[2], np.zeros(len(cand[2])))) &
            (cand[3] > minC)):
            cfgg = np.transpose(np.array([sfrags2, lfrags2],
                                         dtype=object)).tolist()

            cand_list.append([cfgg] + cand)

    cand_list.sort(key=lambda tup: np.linalg.norm(tup[2]))

    for cc in samp_ladder:
        print(cc[1:])

    return cand_list, sbas


# deprecated routine used in outdated bridge3
def endmat(para, send_end):

    child_id = ''.join(str(x) for x in np.array(para[5]))

    inenf = 'inen_%s' % child_id
    outf = 'endout_%s' % child_id
    maoutf = 'MATOUTB_%s' % child_id

    #           basis
    #           jay
    #           costring
    #           nzopt
    #           tnnii
    inen_bdg_3(para[0],
               para[1],
               para[2],
               fn=inenf,
               pari=0,
               nzop=para[3],
               tni=para[4])

    cmdend = para[6] + spectralEXE_mpi_pool + ' %s %s %s' % (inenf, outf,
                                                             maoutf)

    pend = subprocess.Popen(shlex.split(cmdend),
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE)
    #cwd=workdir)

    # <communicate> is needed in order to ensure the process ended before parsing its output!
    out, err = pend.communicate()

    try:
        NormHam = np.core.records.fromfile(maoutf, formats='f8', offset=4)
        dim = int(np.sqrt(len(NormHam) * 0.5))

        # read Norm and Hamilton matrices
        normat = np.reshape(
            np.array(NormHam[:dim**2]).astype(float), (dim, dim))
        hammat = np.reshape(
            np.array(NormHam[dim**2:]).astype(float), (dim, dim))
        # diagonalize normalized norm (using "eigh(ermitian)" to speed-up the computation)
        ewN, evN = eigh(normat)
        idx = ewN.argsort()[::-1]
        ewN = [eww for eww in ewN[idx]]
        evN = evN[:, idx]

        try:
            ewH, evH = eigh(hammat, normat)
            idx = ewH.argsort()[::-1]
            ewH = [eww for eww in ewH[idx]]
            evH = evH[:, idx]
            #print('lowest eigen values (N): ', ewN[-4:])
            #print('lowest eigen values (H): ', ewH[-4:])

        except:
            #print(
            #    'failed to solve generalized eigenvalue problem (norm ev\'s < 0 ?)'
            #)
            attractiveness = 0.
            basCond = 0.
            gsEnergy = 0.
            ewH = []

        if ewH != []:

            anzSigEV = len(
                [bvv for bvv in ewH if para[8][0] < bvv < para[8][1]])

            gsEnergy = ewH[-1]

            basCond = np.min(np.abs(ewN)) / np.max(np.abs(ewN))

            minCond = para[7]

            attractiveness = loveliness(gsEnergy, basCond, anzSigEV, minCond)

        os.system('rm -rf ./%s' % inenf)
        os.system('rm -rf ./%s' % outf)
        os.system('rm -rf ./%s' % maoutf)

        send_end.send([basCond, attractiveness, gsEnergy, para[5], para[0]])

    except:

        os.system('rm -rf ./%s' % inenf)
        os.system('rm -rf ./%s' % outf)
        os.system('rm -rf ./%s' % maoutf)

        print(para[5], child_id)
        print(maoutf)
        send_end.send([0.0, 0.0, -42.7331, para[5], para[0]])


def blunt_ev2(cfgs, widi, basis, nzopt, costring, binpath, potNN, jay,
              funcPath):

    #assert basisDim(basis) == len(sum(sum(relws, []), []))
    anzcfg = len(cfgs)
    os.chdir(funcPath)

    inlu_2(anzo=8, anzf=anzcfg)
    os.system(binpath + NNorbitalEXE)
    inob_2(anzo=8, anzf=anzcfg)
    os.system(binpath + NNspinEXE)

    inqua_2(relw=widi, ps2=potNN)
    subprocess.run([binpath + NNhamilEXE_serial])

    inen_bdg_2(basis, j=jay, costr=costring, fn='INEN', anzo=nzopt)
    sc = jay
    inen_str_2(costring,
               anzrelw=len(basis[0][1]),
               j=jay,
               sc=sc,
               ch=basis[0][0],
               anzo=nzopt,
               fn='INEN_STR')

    subprocess.call('cp -rf INQUA_N INQUA_N_V18', shell=True)

    subprocess.run([binpath + spectralEXE_serial])
    print(funcPath)

    NormHam = np.core.records.fromfile('MATOUTB', formats='f8', offset=4)

    suche_fehler()

    return NormHam


def blunt_ev3(cfgs,
              intws,
              relws,
              basis,
              nzopt,
              costring,
              bin_path,
              mpipath,
              potNN,
              potNNN='',
              parall=-1,
              tnnii=10,
              jay=0.5,
              anzcores=6,
              funcPath='',
              dia=True):

    #assert basisDim(basis) == len(sum(sum(relws, []), []))

    lfrag = np.array(cfgs)[:, 1].tolist()
    sfrag = np.array(cfgs)[:, 0].tolist()
    insam(len(lfrag))

    inlu_3(8, fn='INLUCN', fr=lfrag, indep=parall)
    os.system(bin_path + NNorbitalEXE)
    inob_3(sfrag, 8, fn='INOB', indep=parall)
    os.system(bin_path + NNspinEXE)

    inqua_3(intwi=intws, relwi=relws, potf=potNN, inquaout='INQUA_N')
    parallel_mod_of_3inqua(lfrag,
                           sfrag,
                           infile='INQUA_N',
                           outfile='INQUA_N',
                           einzel_path='')

    inen_bdg_3(basis, jay, costring, fn='INEN', pari=0, nzop=nzopt, tni=tnnii)

    if parall == -1:
        diskfil = disk_avail(funcPath)

        if diskfil < 0.2:
            print('more than %s of disk space is already used!' %
                  (str(diskfil) + '%'))
            exit()

        subprocess.run(
            [mpipath, '-np',
             '%d' % anzcores, bin_path + NNhamilEXE_mpi])
        subprocess.run([bin_path + NNcollect_mpi])
        subprocess.call('rm -rf DMOUT.*', shell=True)
    else:
        subprocess.run([bin_path + NNhamilEXE_serial])

    subprocess.call('cp -rf INQUA_N INQUA_N_V18', shell=True)
    if tnnii == 11:
        inlu_3(8, fn='INLU', fr=lfrag, indep=parall)
        os.system(bin_path + NNNorbitalEXE)
        inob_3(sfrag, 15, fn='INOB', indep=parall)
        os.system(bin_path + NNNspinEXE)

        inqua_3(intwi=intws, relwi=relws, potf=potNNN, inquaout='INQUA_N')
        parallel_mod_of_3inqua(lfrag,
                               sfrag,
                               infile='INQUA_N',
                               outfile='INQUA_N',
                               tni=1,
                               einzel_path='')

        if parall == -1:
            diskfil = disk_avail(funcPath)

            if diskfil < 0.2:
                print('more than %s of disk space is already used!' %
                      (str(diskfil) + '%'))
                exit()

            subprocess.run(
                [mpipath, '-np',
                 '%d' % anzcores, bin_path + NNNhamilEXE_mpi])
            subprocess.run([bin_path + NNNcollect_mpi])
            subprocess.call('rm -rf DRDMOUT.*', shell=True)
            subprocess.run(
                [bin_path + spectralEXE_mpi_pool, 'INEN', 'OUTPUT', 'MATOUTB'],
                capture_output=True,
                text=True)

        else:
            subprocess.run([bin_path + NNNhamilEXE_serial])
            subprocess.run([bin_path + spectralEXE_serial])
    elif tnnii == 10:
        if parall == -1:
            subprocess.run(
                [bin_path + spectralEXE_mpi_pool, 'INEN', 'OUTPUT', 'MATOUTB'],
                capture_output=True,
                text=True)
        else:
            subprocess.run([bin_path + spectralEXE_serial])

    subprocess.call('cp -rf INQUA_N INQUA_N_UIX', shell=True)
    NormHam = np.core.records.fromfile('MATOUTB', formats='f8', offset=4)

    return NormHam


def blunt_ev4t(cfgs,
               intws,
               relws,
               basis,
               nzopt,
               costring,
               bin_path,
               mpipath,
               potNN,
               potNNN='',
               parall=-1,
               tnnii=10,
               jay=0.5,
               anzcores=6,
               funcPath='',
               dia=True):

    #assert basisDim(basis) == len(sum(sum(relws, []), []))

    lfrag = sum(np.array(cfgs, dtype=object)[:, 1].tolist(), [])
    sfrag = np.array(cfgs, dtype=object)[:, 0].tolist()

    insam(len(lfrag))

    inlu_4(8, fn='INLUCN', fr=lfrag, indep=parall)
    os.system(bin_path + NNorbitalEXE)
    inob_4(sfrag, 8, fn='INOB', indep=parall)
    os.system(bin_path + NNspinEXE)

    inqua_4(intwi=intws, relwi=relws, potf=potNN, inquaout='INQUA_N')
    parallel_mod_of_3inqua(lfrag,
                           sfrag,
                           infile='INQUA_N',
                           outfile='INQUA_N',
                           einzel_path='')

    inen_bdg_4(basis, jay, costring, fn='INEN', pari=0, nzop=nzopt, tni=tnnii)

    if parall == -1:
        diskfil = disk_avail(funcPath)

        if diskfil < 0.2:
            print('more than %s of disk space is already used!' %
                  (str(diskfil) + '%'))
            exit()

        subprocess.run(
            [mpipath, '-np',
             '%d' % anzcores, bin_path + NNhamilEXE_mpi])
        subprocess.run([bin_path + NNcollect_mpi])
        subprocess.call('rm -rf DMOUT.*', shell=True)
    else:
        subprocess.run([bin_path + NNhamilEXE_serial])

    subprocess.call('cp -rf INQUA_N INQUA_N_V18', shell=True)
    if tnnii == 11:
        inlu_4(8, fn='INLU', fr=lfrag, indep=parall)
        os.system(bin_path + NNNorbitalEXE)
        inob_4(sfrag, 8, fn='INOB', indep=parall)
        os.system(bin_path + NNNspinEXE)

        inqua_4(intwi=intws, relwi=relws, potf=potNNN, inquaout='INQUA_N')
        parallel_mod_of_3inqua(lfrag,
                               sfrag,
                               infile='INQUA_N',
                               outfile='INQUA_N',
                               tni=1,
                               einzel_path='')

        if parall == -1:
            diskfil = disk_avail(funcPath)

            if diskfil < 0.2:
                print('more than %s of disk space is already used!' %
                      (str(diskfil) + '%'))
                exit()

            subprocess.run(
                [mpipath, '-np',
                 '%d' % anzcores, bin_path + NNNhamilEXE_mpi])
            subprocess.run([bin_path + NNNcollect_mpi])
            subprocess.call('rm -rf DRDMOUT.*', shell=True)
            subprocess.run([
                bin_path + spectralEXE_mpi_pool + spectralEXE_mpi_pool, 'INEN',
                'OUTPUT', 'MATOUTB'
            ],
                           capture_output=True,
                           text=True)
        else:
            subprocess.run([bin_path + NNNhamilEXE_serial])
            subprocess.run([bin_path + spectralEXE_serial])
    elif tnnii == 10:
        if parall == -1:
            subprocess.run(
                [bin_path + spectralEXE_mpi_pool, 'INEN', 'OUTPUT', 'MATOUTB'],
                capture_output=True,
                text=True)
        else:
            subprocess.run([bin_path + spectralEXE_serial])

    subprocess.call('cp -rf INQUA_N INQUA_N_UIX', shell=True)
    NormHam = np.core.records.fromfile('MATOUTB', formats='f8', offset=4)

    return NormHam


def blunt_ev3_parallel(cfgs,
                       intws,
                       relws,
                       basis,
                       nzopt,
                       costring,
                       bin_path,
                       potNN,
                       potNNN,
                       tnnii=10,
                       jay=0.5):

    #assert basisDim(basis) == len(sum(sum(relws, []), []))

    lfrag = np.array(cfgs)[:, 1].tolist()
    sfrag = np.array(cfgs)[:, 0].tolist()

    inqua_3(intwi=intws, relwi=relws, potf=potNN, inquaout='INQUA_N')
    inen_bdg_3(basis, jay, costring, fn='INEN', pari=0, nzop=nzopt, tni=tnnii)

    subprocess.run([bin_path + NNhamilEXE_serial])

    if tnnii == 11:

        inqua_3(intwi=intws, relwi=relws, potf=potNNN, inquaout='INQUA_N')
        subprocess.run([bin_path + NNNhamilEXE_serial])

        subprocess.run([bin_path + spectralEXE_serial])

    elif tnnii == 10:

        subprocess.run([bin_path + spectralEXE_serial])

    NormHam = np.core.records.fromfile('MATOUTB', formats='f8', offset=4)

    return NormHam


def blunt_ev4(cfgs,
              bas,
              nzopt,
              costring,
              bin_path,
              mpipath,
              potNN,
              frgCoff,
              j1j2sc,
              dmaa=[1, 0, 1, 0, 1, 0, 0],
              potNNN='',
              parall=-1,
              tnnii=10,
              jay=0.5,
              anzcores=6,
              funcPath='',
              dia=True,
              nchtot=1,
              distchannels=''):

    #assert basisDim(basis) == len(sum(sum(relws, []), []))

    tt = np.reshape(np.array(cfgs), (-1, 2))

    lfrag = np.array(tt)[:, 0].tolist()
    sfrag = np.array(tt)[:, 1].tolist()
    insam(len(lfrag))
    inlu_4(8, fn='INLUCN', fr=lfrag, indep=parall)
    os.system(bin_path + NNorbitalEXE)
    inob_4(sfrag, 8, fn='INOB', indep=parall)
    os.system(bin_path + NNspinEXE)

    repl_line('INQUA_N', 1, potNN + '\n')

    parallel_mod_of_3inqua(lfrag,
                           sfrag,
                           infile='INQUA_N',
                           outfile='INQUA_N',
                           einzel_path='')

    inen_bdg_4(bas,
               jay,
               costring,
               fn='INEN_BDGp',
               pari=0,
               nzop=nzopt,
               tni=tnnii)

    inen_str_4(phys_chan=j1j2sc,
               coeff=costring,
               wr=widthSet_relative[-1],
               bvs=bas,
               uec=frgCoff,
               dma=dmaa,
               jay=jay,
               anzch=nchtot,
               pari=0,
               nzop=nzopt,
               tni=tnnii,
               fn='INEN',
               diCh=distchannels)

    if parall == -1:

        subprocess.run(
            [mpipath, '-np',
             '%d' % anzcores, bin_path + NNhamilEXE_mpi])
        subprocess.run([bin_path + NNcollect_mpi])
        subprocess.call('rm -rf DMOUT.*', shell=True)
    else:
        subprocess.run([bin_path + NNhamilEXE_serial])

    subprocess.call('cp -rf INQUA_N INQUA_N_V18', shell=True)
    if tnnii == 11:
        inlu_4(8, fn='INLU', fr=lfrag, indep=parall)
        os.system(bin_path + NNNorbitalEXE)
        inob_4(sfrag, 15, fn='INOB', indep=parall)
        os.system(bin_path + NNNspinEXE)

        repl_line('INQUA_N', 1, potNNN + '\n')

        parallel_mod_of_3inqua(lfrag,
                               sfrag,
                               infile='INQUA_N',
                               outfile='INQUA_N',
                               tni=1,
                               einzel_path='')

        subprocess.call('cp -rf INQUA_N INQUA_N_UIX', shell=True)

        if parall == -1:
            subprocess.run(
                [mpipath, '-np',
                 '%d' % anzcores, bin_path + NNNhamilEXE_mpi])

            subprocess.run([bin_path + NNNcollect_mpi])
            subprocess.call('rm -rf DRDMOUT.*', shell=True)
            if dia:
                subprocess.run([
                    bin_path + spectralEXE_mpi_pool, 'INEN_BDGp', 'OUTPUT_BDG',
                    'MATOUTBNDG'
                ],
                               capture_output=True,
                               text=True)
            subprocess.run(
                [bin_path + spectralEXE_mpi_pool, 'INEN', 'OUTPUT', 'MATOUTB'],
                capture_output=True,
                text=True)

        else:
            subprocess.run([bin_path + NNNhamilEXE_serial])
            if dia:
                subprocess.call('cp -rf INEN INEN_STR', shell=True)
                subprocess.call('cp -rf INEN_BDGp INEN', shell=True)
                subprocess.run([bin_path + spectralEXE_serial])
                subprocess.call('cp -rf OUTPUT OUTPUT_BDG', shell=True)
                subprocess.call('cp -rf INEN_STR INEN', shell=True)

            subprocess.run([bin_path + spectralEXE_serial])

    elif tnnii == 10:
        if parall == -1:
            if dia:
                subprocess.run([
                    bin_path + spectralEXE_mpi_pool, 'INEN_BDGp', 'OUTPUT_BDG',
                    'MATOUTBNDG'
                ],
                               capture_output=True,
                               text=True)

            subprocess.run(
                [bin_path + spectralEXE_mpi_pool, 'INEN', 'OUTPUT', 'MATOUTB'],
                capture_output=True,
                text=True)
        else:
            if dia:
                subprocess.call('cp -rf INEN INEN_STR', shell=True)
                subprocess.call('cp -rf INEN_BDGp INEN', shell=True)
                subprocess.run([bin_path + spectralEXE_serial])
                subprocess.call('cp -rf OUTPUT OUTPUT_BDG', shell=True)
                subprocess.call('cp -rf INEN_STR INEN', shell=True)

            subprocess.run([bin_path + spectralEXE_serial])

    NormHam = np.core.records.fromfile('MATOUTB', formats='f8', offset=4)

    return NormHam


def blunt_ev5(cfgs,
              bas,
              nzopt,
              costring,
              bin_path,
              mpipath,
              potNN,
              frgCoff,
              j1j2sc,
              dmaa=[1, 0, 1, 0, 1, 0, 0],
              potNNN='',
              parall=-1,
              tnnii=10,
              jay=0.5,
              anzcores=6,
              funcPath='',
              dia=True,
              nchtot=1):

    #assert basisDim(basis) == len(sum(sum(relws, []), []))

    tt = np.reshape(np.array(cfgs), (-1, 2))

    lfrag = np.array(tt)[:, 0].tolist()
    sfrag = np.array(tt)[:, 1].tolist()
    insam(len(lfrag))
    inlu_5(8, fn='INLUCN', fr=lfrag, indep=parall)
    os.system(bin_path + NNorbitalEXE)
    inob_5(sfrag, 8, fn='INOB', indep=parall)
    os.system(bin_path + NNspinEXE)

    repl_line('INQUA_N', 1, potNN + '\n')

    parallel_mod_of_3inqua(lfrag,
                           sfrag,
                           infile='INQUA_N',
                           outfile='INQUA_N',
                           einzel_path='')

    inen_bdg_5(bas,
               jay,
               costring,
               fn='INEN_BDGp',
               pari=0,
               nzop=nzopt,
               tni=tnnii)

    inen_str_5(phys_chan=j1j2sc,
               coeff=costring,
               wr=widthSet_relative[-1],
               bvs=bas,
               uec=frgCoff,
               dma=dmaa,
               jay=jay,
               anzch=nchtot,
               pari=0,
               nzop=nzopt,
               tni=tnnii,
               fn='INEN')

    if parall == -1:

        subprocess.run(
            [mpipath, '-np',
             '%d' % anzcores, bin_path + NNhamilEXE_mpi])
        subprocess.run([bin_path + NNcollect_mpi])
        subprocess.call('rm -rf DMOUT.*', shell=True)
    else:
        subprocess.run([bin_path + NNhamilEXE_serial])

    subprocess.call('cp -rf INQUA_N INQUA_N_V18', shell=True)

    if tnnii == 11:
        inlu_5(8, fn='INLU', fr=lfrag, indep=parall)
        os.system(bin_path + NNNorbitalEXE)
        inob_5(sfrag, 8, fn='INOB', indep=parall)
        os.system(bin_path + NNNspinEXE)

        repl_line('INQUA_N', 1, potNNN + '\n')

        parallel_mod_of_3inqua(lfrag,
                               sfrag,
                               infile='INQUA_N',
                               outfile='INQUA_N',
                               tni=1,
                               einzel_path='')

        subprocess.call('cp -rf INQUA_N INQUA_N_UIX', shell=True)

        if parall == -1:
            subprocess.run(
                [mpipath, '-np',
                 '%d' % anzcores, bin_path + NNNhamilEXE_mpi])

            subprocess.run([bin_path + NNNcollect_mpi])
            subprocess.call('rm -rf DRDMOUT.*', shell=True)
            subprocess.run(
                [bin_path + spectralEXE_mpi_pool, 'INEN', 'OUTPUT', 'MATOUTB'],
                capture_output=True,
                text=True)

        else:
            subprocess.run([bin_path + NNNhamilEXE_serial])
            subprocess.run([bin_path + spectralEXE_serial])

    elif tnnii == 10:
        if parall == -1:
            subprocess.run(
                [bin_path + spectralEXE_mpi_pool, 'INEN', 'OUTPUT', 'MATOUTB'],
                capture_output=True,
                text=True)
        else:
            subprocess.run([bin_path + spectralEXE_serial])

    NormHam = np.core.records.fromfile('MATOUTB', formats='f8', offset=4)

    return NormHam