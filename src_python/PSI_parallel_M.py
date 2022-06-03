import os
import operator
import shlex

from parameters_and_constants import *
from three_particle_functions import *
from two_particle_functions import *
from functional_assistants import *
from rrgm_functions import *
from scipy.linalg import eigh
from genetic_width_growth import *

MaxProc = int(len(os.sched_getaffinity(0)) / 2)
MPIRUN = subprocess.getoutput("which mpirun")
tnni = 11
NEWLINE_SIZE_IN_BYTES = -1
dt = 'float64'


def endmat2(para, send_end):

    child_id = ''.join(str(x) for x in np.array(para[5]))

    inenf = 'inen_%s' % child_id
    outf = 'endout_%s' % child_id
    maoutf = 'MATOUTB_%s' % child_id

    #           basis
    #           jay
    #           costring
    #           nzopt
    #           tnnii

    h2_inen_bs(para[0], j=para[1], costr=para[2], fn=inenf, pari=0)

    cmdend = para[6] + 'DR2END_AK_PYpoolnoo.exe %s %s %s' % (inenf, outf,
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
    lit_rw = {}

    he_iw = he_rw = he_frgs = ob_stru = lu_stru = sbas = []

    # minimal distance allowed for between width parameters
    rwma = 20
    bvma = 20

    # lower bound for width parameters '=' IR cutoff (broadest state)
    rWmin = 0.0001

    # orbital-angular-momentum dependent upper bound '=' UV cutoff (narrowest state)
    iLcutoff = [120., 4., 3.]

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
        dists = [np.linalg.norm(wp - lit_w_tmp) for wp in lit_w]
        if ((np.min(dists) > mindist) & (lit_w_tmp < iLcutoff[0])):
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
        assert itera <= 10000

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
    bv = two_body_channels[channel]
    for n in range(len(frags)):
        sbas += [[bv, [x for x in range(1, 1 + len(widi[n]))]]]
        bv += 13

    os.chdir(funcPath)

    h2_inlu(anzo=8, anzf=len(frags))
    os.system(binPath + 'LUDW_CN.exe')
    h2_inob(anzo=8, anzf=len(frags))
    os.system(binPath + 'KOBER.exe')

    h2_inqua(relw=widi, ps2='./nn_pot')
    subprocess.run([binPath + 'QUAFL_N.exe'])

    h2_inen_bs(sbas, j=Jstreu, costr=coefstr)

    subprocess.call('cp -rf INQUA_N INQUA_N_V18', shell=True)

    subprocess.run([binPath + 'DR2END_AK.exe'])

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


def span_initial_basis(fragments,
                       Jstreu,
                       coefstr,
                       funcPath,
                       binPath,
                       ini_grid_bounds=[0.01, 9.5, 0.001, 11.5],
                       ini_dims=[4, 4, 4, 4],
                       parall=-1):

    os.chdir(funcPath)

    Jstreustring = '%s' % str(Jstreu)[:3]

    lfrags = []
    sfrags = []
    lfrags2 = []
    sfrags2 = []

    for lcfg in range(len(fragments)):
        sfrags = sfrags + fragments[lcfg][1]
        for scfg in fragments[lcfg][1]:
            lfrags = lfrags + [fragments[lcfg][0]]

    lit_w = {}
    lit_rw = {}

    he_iw = he_rw = he_frgs = ob_stru = lu_stru = sbas = []

    # minimal distance allowed for between width parameters
    rwma = 20
    bvma = 8
    mindist_int = 0.001
    mindist_rel = 0.001

    # lower bound for width parameters '=' IR cutoff (broadest state)
    rWmin = 0.0001

    # orbital-angular-momentum dependent upper bound '=' UV cutoff (narrowest state)
    iLcutoff = [12., 4., 3.]
    rLcutoff = [12., 4., 3.]
    nwint = ini_dims[0]
    nwrel = ini_dims[1]
    rel_scale = 1.
    wi, wf, nw = ini_grid_bounds[0], ini_grid_bounds[1], [
        nwint for n in lfrags
    ]  # initial helion bound state

    if nwrel >= rwma:
        print(
            'The set number for relative width parameters per basis vector > max!'
        )
        exit()

    lit_rw_sparse = np.empty(len(sfrags), dtype=list)
    for frg in range(len(lfrags)):
        #Lsum = np.sum([int(ie) for ie in lfrags[frg]])

        #  -- internal widths --------------------------------------------------
        offset = 1.
        #if (sfrags[frg][-1] == 'y'):
        if nw[frg] != 1:
            offset += 0.1 * frg / (1 + len(lfrags[frg]))

        wii = wi * offset
        wff = wf * offset

        lit_w_tmp = wii + np.random.random(nw[frg]) * (wff - wii)
        #lit_w_tmp = np.abs(
        #    np.geomspace(start=wii,
        #                 stop=wff,
        #                 num=nw[frg],
        #                 endpoint=True,
        #                 dtype=None))

        if nw[frg] != 1:
            lit_w_tmp = np.sort([wd * np.random.random()
                                 for wd in lit_w_tmp])[::-1]

        lit_w[frg] = lit_w_tmp

        lit_w[frg] = [
            ww for ww in sparse(lit_w[frg], mindist=mindist_int)
            if rWmin < ww < iLcutoff[int(
                np.max([float(lfrags[frg][0]),
                        float(lfrags[frg][1])]))]
        ]
        #  -- relative widths --------------------------------------------------

        wir, wfr, nwr = rel_scale * ini_grid_bounds[
            2], rel_scale * ini_grid_bounds[3], nwrel * len(lit_w[frg])

        offset = 0.5 + np.random.random() if nwr != 1 else 1.0

        wiir = wir
        wffr = wfr

        lit_w_tmp = wiir + np.random.random(nwr) * (wffr - wiir)

        #lit_w_tmp = np.geomspace(start=wiir,
        #                         stop=wffr,
        #                         num=nwr,
        #                         endpoint=True,
        #                         dtype=None)
        lit_w_tmp = offset * lit_w_tmp

        if nwr != 1:
            lit_w_tmp = np.sort([wd * np.random.random()
                                 for wd in lit_w_tmp])[::-1]

        lit_rw_tmp = [
            ww for ww in sparse(np.abs(
                np.sort(np.array(lit_w_tmp).flatten())[::-1].tolist()),
                                mindist=mindist_rel)
            if rWmin < ww < rLcutoff[int(
                np.max([float(lfrags[frg][0]),
                        float(lfrags[frg][1])]))]
        ]

        if lit_rw_tmp == []:
            lit_rw_tmp = [np.random.random()]

        lit_rw[frg] = []
        for bv in range(len(lit_w[frg])):
            lit_rw[frg].append(lit_rw_tmp[bv::len(lit_w[frg])])
        minL = np.min([len(ws) for ws in lit_rw[frg]])
        lit_rw[frg] = [wd[-minL:] for wd in lit_rw[frg]]

    widi = []
    widr = []
    for n in range(len(lit_w)):
        tmp = np.sort(lit_w[n])[::-1]
        #tmp = sparse(tmp, mindist_int)
        zer_per_ws = int(np.ceil(len(tmp) / bvma))
        bins = [0 for nmmm in range(zer_per_ws + 1)]
        bins[0] = 0
        for mn in range(len(tmp)):
            bins[1 + mn % zer_per_ws] += 1
        bnds = np.cumsum(bins)
        tmp2 = [list(tmp[bnds[nn]:bnds[nn + 1]]) for nn in range(zer_per_ws)]
        tmp3 = [lit_rw[n][bnds[nn]:bnds[nn + 1]] for nn in range(zer_per_ws)]
        sfrags2 += len(tmp2) * [sfrags[n]]
        lfrags2 += len(tmp2) * [lfrags[n]]
        widi += tmp2
        widr += tmp3
    anzBV = sum([len(zer) for zer in widi])
    print(
        'seed state with (%d) basis-vector blocks with [orbital][(iso)spin] configurations:'
        % anzBV)
    print(lfrags2, sfrags2, '\n')

    sbas = []
    bv = 1
    for n in range(len(lfrags2)):
        bvv = 0
        for m in range(len(widi[n])):
            bvv += 1
            #sbas += [[bv, [(bvv) % (1 + len(widr[n][m]))]]]
            sbas += [[
                bv,
                [
                    x for x in range(1, 1 + max([len(wid)
                                                 for wid in widr[n]]), 1)
                ]
            ]]
            bv += 1

    os.chdir(funcPath)

    n3_inlu(8, fn='INLU', fr=lfrags2, indep=parall)
    os.system(binPath + 'DRLUD.exe')
    n3_inlu(8, fn='INLUCN', fr=lfrags2, indep=parall)
    os.system(binPath + 'LUDW_CN.exe')
    n3_inob(sfrags2, 8, fn='INOB', indep=parall)
    os.system(binPath + 'KOBER.exe')
    n3_inob(sfrags2, 15, fn='INOB', indep=parall)
    os.system(binPath + 'DROBER.exe')

    he3inquaBS(intwi=widi, relwi=widr, potf='./nn_pot')
    parallel_mod_of_3inqua(lfrags2,
                           sfrags2,
                           infile='INQUA_M',
                           outfile='INQUA_M',
                           einzel_path=funcPath + '/')

    insam(len(lfrags2))

    anzproc = max(2, min(len(lfrags2), MaxProc))
    #print('Anzahl der Sklaven + 1: %d' % anzproc)
    #exit()

    n3_inen_bdg(sbas, Jstreu, coefstr, fn='INEN', pari=0)

    if parall == -1:
        diskfil = disk_avail(funcPath)

        if diskfil < 0.2:
            print('more than %s of disk space is already used!' %
                  (str(diskfil) + '%'))
            exit()

        subprocess.run(
            [MPIRUN, '-np',
             '%d' % anzproc, binPath + 'V18_PAR/mpi_quaf_v7'])
        subprocess.run([binPath + 'V18_PAR/sammel'])
        subprocess.call('rm -rf DMOUT.*', shell=True)

    else:
        subprocess.run([binPath + 'QUAFL_M.exe'])

    subprocess.call('cp -rf INQUA_M INQUA_M_V18', shell=True)

    if tnni == 11:
        he3inquaBS(intwi=widi, relwi=widr, potf='./nnn_pot')
        parallel_mod_of_3inqua(lfrags2,
                               sfrags2,
                               infile='INQUA_M',
                               outfile='INQUA_M',
                               tni=1,
                               einzel_path=funcPath + '/')
        if parall == -1:
            diskfil = disk_avail(funcPath)

            if diskfil < 0.2:
                print('more than %s of disk space is already used!' %
                      (str(diskfil) + '%'))
                exit()

            subprocess.run([
                MPIRUN, '-np',
                '%d' % anzproc, binPath + 'UIX_PAR/mpi_drqua_v7'
            ])

            subprocess.run([binPath + 'UIX_PAR/SAMMEL-uix'])
            subprocess.call('rm -rf DRDMOUT.*', shell=True)

            subprocess.run([
                binPath + 'TDR2END_PYpoolnoo.exe', 'INEN',
                'OUTPUT_TDR2END_PYpoolnoo', 'MATOUTB'
            ])
            subprocess.call('cp OUTPUT out_normal', shell=True)
        else:
            subprocess.run([binPath + 'DRQUA_AK_M.exe'])
            subprocess.run([binPath + 'DR2END_AK.exe'])

    elif tnni == 10:
        if parall == -1:
            subprocess.run([
                binPath + 'TDR2END_PYpoolnoo.exe', 'INEN',
                'OUTPUT_TDR2END_PYpoolnoo', 'MATOUTB'
            ])
            subprocess.call('cp OUTPUT out_normal', shell=True)
        else:
            subprocess.run([binPath + 'DR2END_NORMAL.exe'])

    subprocess.call('cp -rf INQUA_M INQUA_M_UIX', shell=True)
    suche_fehler()

    matout = np.core.records.fromfile('MATOUTB', formats='f8', offset=4)

    # write basis structure on tape

    path_bas_dims = funcPath + '/bas_dims.dat'
    with open(path_bas_dims, 'wb') as f:
        np.savetxt(f, [np.size(wid) for wid in widr], fmt='%d')
        f.seek(NEWLINE_SIZE_IN_BYTES, 2)
        f.truncate()
    f.close()
    path_bas_int_rel_pairs = funcPath + '/bas_full.dat'
    if os.path.exists(path_bas_int_rel_pairs):
        os.remove(path_bas_int_rel_pairs)
    with open(path_bas_int_rel_pairs, 'w') as oof:
        #np.savetxt(f, [[jj[0], kk] for jj in sbas for kk in jj[1]], fmt='%d')
        #f.seek(NEWLINE_SIZE_IN_BYTES, 2)
        #f.truncate()
        so = ''
        for bv in sbas:
            so += '%4s' % str(bv[0])
            for rww in bv[1]:
                so += '%4s' % str(rww)
            so += '\n'
        oof.write(so)
    oof.close()
    path_frag_stru = funcPath + '/frags.dat'
    if os.path.exists(path_frag_stru): os.remove(path_frag_stru)

    with open(path_frag_stru, 'wb') as f:
        np.savetxt(f,
                   np.column_stack([sfrags2, lfrags2]),
                   fmt='%s',
                   delimiter=' ',
                   newline=os.linesep)
        f.seek(NEWLINE_SIZE_IN_BYTES, 2)
        f.truncate()
    f.close()
    path_intw = funcPath + '/intw.dat'
    if os.path.exists(path_intw): os.remove(path_intw)
    with open(path_intw, 'wb') as f:
        for ws in widi:
            np.savetxt(f, [ws], fmt='%12.6f', delimiter=' ')
        f.seek(NEWLINE_SIZE_IN_BYTES, 2)
        f.truncate()
    f.close()
    path_relw = funcPath + '/relw.dat'
    if os.path.exists(path_relw): os.remove(path_relw)
    with open(path_relw, 'wb') as f:
        for wss in widr:
            for ws in wss:
                np.savetxt(f, [ws], fmt='%12.6f', delimiter=' ')
        f.seek(NEWLINE_SIZE_IN_BYTES, 2)
        f.truncate()
    f.close()

    return matout


def span_initial_basis3(fragments,
                        Jstreu,
                        coefstr,
                        funcPath,
                        binPath,
                        mindists=[0.01, 0.01],
                        ini_grid_bounds=[0.01, 9.5, 0.001, 11.5],
                        ini_dims=[4, 4],
                        parall=-1):

    os.chdir(funcPath)

    Jstreustring = '%s' % str(Jstreu)[:3]

    lfrags = []
    sfrags = []
    lfrags2 = []
    sfrags2 = []

    for lcfg in range(len(fragments)):
        sfrags = sfrags + fragments[lcfg][1]
        for scfg in fragments[lcfg][1]:
            lfrags = lfrags + [fragments[lcfg][0]]

    lit_w = {}
    lit_rw = {}

    he_iw = he_rw = he_frgs = ob_stru = lu_stru = sbas = []

    # minimal distance allowed for between width parameters
    rwma = 20
    bvma = 2
    mindist_int = mindists[0]
    mindist_rel = mindists[1]

    # lower bound for width parameters '=' IR cutoff (broadest state)
    rWmin = 0.0001

    # orbital-angular-momentum dependent upper bound '=' UV cutoff (narrowest state)
    iLcutoff = [12., 4., 3.]
    rLcutoff = [12., 4., 3.]
    nwint = ini_dims[0]
    nwrel = ini_dims[1]
    rel_scale = 1.

    if nwrel >= rwma:
        print(
            'The set number for relative width parameters per basis vector > max!'
        )
        exit()

    lit_rw_sparse = np.empty(len(sfrags), dtype=list)
    for frg in range(len(lfrags)):

        #  -- internal widths --------------------------------------------------
        itera = 1
        lit_w_t = [
            ini_grid_bounds[0] + np.random.random() *
            (ini_grid_bounds[1] - ini_grid_bounds[0])
        ]
        while len(lit_w_t) != nwint:

            lit_w_tmp = ini_grid_bounds[0] + np.random.random() * (
                ini_grid_bounds[1] - ini_grid_bounds[0])
            dists = [np.linalg.norm(wp - lit_w_tmp) for wp in lit_w_t]
            if ((np.min(dists) > mindist_int) & (lit_w_tmp < iLcutoff[0])):
                lit_w_t.append(lit_w_tmp)

            itera += 1
            assert itera <= 10000

        lit_w[frg] = np.sort(lit_w_t)[::-1]

        #  -- relative widths --------------------------------------------------
        itera = 1
        lit_w_t = [
            ini_grid_bounds[2] + np.random.random() *
            (ini_grid_bounds[3] - ini_grid_bounds[2])
        ]
        while len(lit_w_t) != nwrel:

            lit_w_tmp = ini_grid_bounds[2] + np.random.random() * (
                ini_grid_bounds[3] - ini_grid_bounds[2])
            dists = [np.linalg.norm(wp - lit_w_tmp) for wp in lit_w_t]
            if ((np.min(dists) > mindist_rel) & (lit_w_tmp < rLcutoff[0])):
                lit_w_t.append(lit_w_tmp)

            itera += 1
            assert itera <= 10000

        lit_rw[frg] = np.sort(lit_w_t)[::-1]

    widi = []
    widr = []
    for n in range(len(lit_w)):
        tmp = np.sort(lit_w[n])[::-1]
        #tmp = sparse(tmp, mindist_int)
        zer_per_ws = int(np.ceil(len(tmp) / bvma))
        bins = [0 for nmmm in range(zer_per_ws + 1)]
        bins[0] = 0
        for mn in range(len(tmp)):
            bins[1 + mn % zer_per_ws] += 1
        bnds = np.cumsum(bins)
        tmp2 = [list(tmp[bnds[nn]:bnds[nn + 1]]) for nn in range(zer_per_ws)]
        tmp3 = [list(lit_rw[nn]) for nn in range(zer_per_ws)]
        sfrags2 += len(tmp2) * [sfrags[n]]
        lfrags2 += len(tmp2) * [lfrags[n]]
        widi += tmp2
        widr += tmp3

    anzBV = sum([len(zer) for zer in widi])

    sbas = []
    bv = 1
    for n in range(len(lfrags2)):
        for m in range(len(widi[n])):
            sbas += [[bv, [x for x in range(1 + bv % 2, 1 + len(widr[n]), 2)]]]
            bv += 1

    os.chdir(funcPath)

    n3_inlu(8, fn='INLU', fr=lfrags2, indep=parall)
    os.system(binPath + 'DRLUD.exe')
    n3_inlu(8, fn='INLUCN', fr=lfrags2, indep=parall)
    os.system(binPath + 'LUDW_CN.exe')
    n3_inob(sfrags2, 8, fn='INOB', indep=parall)
    os.system(binPath + 'KOBER.exe')
    n3_inob(sfrags2, 15, fn='INOB', indep=parall)
    os.system(binPath + 'DROBER.exe')
    n3_inqua_N(intwi=widi, relwi=widr, potf='nn_pot', inquaout='INQUA_N')
    parallel_mod_of_3inqua(lfrags2,
                           sfrags2,
                           infile='INQUA_N',
                           outfile='INQUA_N',
                           einzel_path=funcPath + '/')

    insam(len(lfrags2))

    anzproc = max(2, min(len(lfrags2), MaxProc))

    n3_inen_bdg(sbas, Jstreu, coefstr, fn='INEN', pari=0)

    if parall == -1:
        diskfil = disk_avail(funcPath)

        if diskfil < 0.2:
            print('more than %s of disk space is already used!' %
                  (str(diskfil) + '%'))
            exit()

        subprocess.run(
            [MPIRUN, '-np',
             '%d' % anzproc, binPath + 'V18_PAR/mpi_quaf_v6'])
        subprocess.run([binPath + 'V18_PAR/sammel'])
        subprocess.call('rm -rf DMOUT.*', shell=True)

    else:
        subprocess.run([binPath + 'QUAFL_N.exe'])

    subprocess.call('cp -rf INQUA_N INQUA_N_V18', shell=True)

    if tnni == 11:
        n3_inqua_N(intwi=widi, relwi=widr, potf='nnn_pot', inquaout='INQUA_N')
        parallel_mod_of_3inqua(lfrags2,
                               sfrags2,
                               infile='INQUA_N',
                               outfile='INQUA_N',
                               tni=1,
                               einzel_path=funcPath + '/')
        if parall == -1:
            diskfil = disk_avail(funcPath)

            if diskfil < 0.2:
                print('more than %s of disk space is already used!' %
                      (str(diskfil) + '%'))
                exit()

            subprocess.run([
                MPIRUN, '-np',
                '%d' % anzproc, binPath + 'UIX_PAR/mpi_drqua_uix'
            ])

            subprocess.run([binPath + 'UIX_PAR/SAMMEL-uix'])
            subprocess.call('rm -rf DRDMOUT.*', shell=True)

            subprocess.run([binPath + 'TDR2END_AK.exe'])
            subprocess.call('cp OUTPUT out_normal', shell=True)
        else:
            subprocess.run([binPath + 'DRQUA_AK_N.exe'])
            subprocess.run([binPath + 'DR2END_AK.exe'])

    elif tnni == 10:
        if parall == -1:
            subprocess.run([binPath + 'TDR2END_AK.exe'])
            subprocess.call('cp OUTPUT out_normal', shell=True)
        else:
            subprocess.run([binPath + 'DR2END_AK.exe'])

    subprocess.call('cp -rf INQUA_N INQUA_N_UIX', shell=True)
    suche_fehler()

    matout = np.core.records.fromfile('MATOUTB', formats='f8', offset=4)

    # write basis structure on tape

    path_frag_stru = funcPath + '/frags.dat'
    if os.path.exists(path_frag_stru): os.remove(path_frag_stru)
    with open(path_frag_stru, 'wb') as f:
        np.savetxt(f,
                   np.column_stack([sfrags2, lfrags2]),
                   fmt='%s',
                   delimiter=' ',
                   newline=os.linesep)
        f.seek(NEWLINE_SIZE_IN_BYTES, 2)
        f.truncate()
    f.close()

    path_intw = funcPath + '/intw.dat'
    if os.path.exists(path_intw): os.remove(path_intw)
    with open(path_intw, 'wb') as f:
        for ws in widi:
            np.savetxt(f, [ws], fmt='%12.6f', delimiter=' ')
        f.seek(NEWLINE_SIZE_IN_BYTES, 2)
        f.truncate()
    f.close()

    path_relw = funcPath + '/relw.dat'
    if os.path.exists(path_relw): os.remove(path_relw)
    with open(path_relw, 'wb') as f:
        for ws in widr:
            np.savetxt(f, [ws], fmt='%12.6f', delimiter=' ')
        f.seek(NEWLINE_SIZE_IN_BYTES, 2)
        f.truncate()
    f.close()

    return matout


def prepare_einzel(funcPath, binPath):

    if os.path.isdir(funcPath + '/eob') == False:
        subprocess.check_call(['mkdir', '-p', funcPath + '/eob'])
        os.chdir(funcPath + '/eob')
        n3_inob([
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
        ],
                8,
                fn='INOB',
                indep=+1)
        os.system(binPath + 'KOBER.exe')

    if os.path.isdir(funcPath + '/eob-tni') == False:
        subprocess.check_call(['mkdir', '-p', funcPath + '/eob-tni'])
        os.chdir(funcPath + '/eob-tni')
        n3_inob([
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
        ],
                15,
                fn='INOB',
                indep=+1)
        os.system(binPath + 'DROBER.exe')

    if os.path.isdir(funcPath + '/elu') == False:
        subprocess.check_call(['mkdir', '-p', funcPath + '/elu'])
        os.chdir(funcPath + '/elu')
        n3_inlu(8,
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
        os.system(binPath + 'LUDW_CN.exe')

    if os.path.isdir(funcPath + '/elu-tni') == False:
        subprocess.check_call(['mkdir', '-p', funcPath + '/elu-tni'])
        os.chdir(funcPath + '/elu-tni')
        n3_inlu(8,
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
        os.system(binPath + 'DRLUD.exe')


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
    n3_inen_bdg(para[0],
                para[1],
                para[2],
                fn=inenf,
                pari=0,
                nzop=para[3],
                tni=para[4])

    cmdend = para[6] + 'TDR2END_PYpoolnoo.exe %s %s %s' % (inenf, outf, maoutf)

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


def blunt_ev(cfgs,
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

    n3_inlu(8, fn='INLUCN', fr=lfrag, indep=parall)
    os.system(bin_path + 'LUDW_CN.exe')
    n3_inob(sfrag, 8, fn='INOB', indep=parall)
    os.system(bin_path + 'KOBER.exe')

    he3inquaBS(intwi=intws, relwi=relws, potf=potNN, inquaout='INQUA_M_0')
    parallel_mod_of_3inqua(lfrag,
                           sfrag,
                           infile='INQUA_M_0',
                           outfile='INQUA_M',
                           einzel_path=funcPath + '/')

    n3_inen_bdg(basis, jay, costring, fn='INEN', pari=0, nzop=nzopt, tni=tnnii)

    if parall == -1:
        diskfil = disk_avail(funcPath)

        if diskfil < 0.2:
            print('more than %s of disk space is already used!' %
                  (str(diskfil) + '%'))
            exit()

        subprocess.run([
            mpipath, '-np',
            '%d' % anzcores, bin_path + 'V18_PAR/mpi_quaf_v7'
        ])
        subprocess.run([bin_path + 'V18_PAR/sammel'])
        subprocess.call('rm -rf DMOUT.*', shell=True)
    else:
        subprocess.run([bin_path + 'QUAFL_M.exe'])

    subprocess.call('cp -rf INQUA_M INQUA_M_V18', shell=True)
    if tnnii == 11:
        n3_inlu(8, fn='INLU', fr=lfrag, indep=parall)
        os.system(bin_path + 'DRLUD.exe')
        n3_inob(sfrag, 15, fn='INOB', indep=parall)
        os.system(bin_path + 'DROBER.exe')

        he3inquaBS(intwi=intws, relwi=relws, potf=potNNN, inquaout='INQUA_M_0')
        parallel_mod_of_3inqua(lfrag,
                               sfrag,
                               infile='INQUA_M_0',
                               outfile='INQUA_M',
                               tni=1,
                               einzel_path=funcPath + '/')

        if parall == -1:
            diskfil = disk_avail(funcPath)

            if diskfil < 0.2:
                print('more than %s of disk space is already used!' %
                      (str(diskfil) + '%'))
                exit()

            subprocess.run([
                mpipath, '-np',
                '%d' % anzcores, bin_path + 'UIX_PAR/mpi_drqua_v7'
            ])
            subprocess.run([bin_path + 'UIX_PAR/SAMMEL-uix'])
            subprocess.call('rm -rf DRDMOUT.*', shell=True)
            subprocess.run([
                bin_path + 'TDR2END_PYpoolnoo.exe', 'INEN',
                'OUTPUT_TDR2END_PYpoolnoo', 'MATOUTB'
            ],
                           capture_output=True,
                           text=True)
        else:
            subprocess.run([bin_path + 'DRQUA_AK_M.exe'])
            subprocess.run([bin_path + 'DR2END_AK.exe'])
    elif tnnii == 10:
        if parall == -1:
            subprocess.run([
                bin_path + 'TDR2END_PYpoolnoo.exe', 'INEN',
                'OUTPUT_TDR2END_PYpoolnoo', 'MATOUTB'
            ],
                           capture_output=True,
                           text=True)
        else:
            subprocess.run([bin_path + 'DR2END_NORMAL.exe'])

    subprocess.call('cp -rf INQUA_M INQUA_M_UIX', shell=True)
    NormHam = np.core.records.fromfile('MATOUTB', formats='f8', offset=4)

    return NormHam


def blunt_ev2(cfgs, widi, basis, nzopt, costring, binpath, potNN, jay,
              funcPath):

    #assert basisDim(basis) == len(sum(sum(relws, []), []))
    anzcfg = len(cfgs)

    os.chdir(funcPath)

    h2_inlu(anzo=8, anzf=anzcfg)
    os.system(binpath + 'LUDW_CN.exe')
    h2_inob(anzo=8, anzf=anzcfg)
    os.system(binpath + 'KOBER.exe')

    h2_inqua(relw=widi, ps2='./nn_pot')
    subprocess.run([binpath + 'QUAFL_N.exe'])

    h2_inen_bs(basis, j=jay, costr=costring)

    subprocess.call('cp -rf INQUA_N INQUA_N_V18', shell=True)

    subprocess.run([binpath + 'DR2END_AK.exe'])

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

    n3_inlu(8, fn='INLUCN', fr=lfrag, indep=parall)
    os.system(bin_path + 'LUDW_CN.exe')
    n3_inob(sfrag, 8, fn='INOB', indep=parall)
    os.system(bin_path + 'KOBER.exe')

    n3_inqua_N(intwi=intws, relwi=relws, potf=potNN, inquaout='INQUA_N')
    parallel_mod_of_3inqua(lfrag,
                           sfrag,
                           infile='INQUA_N',
                           outfile='INQUA_N',
                           einzel_path=funcPath + '/')

    n3_inen_bdg(basis, jay, costring, fn='INEN', pari=0, nzop=nzopt, tni=tnnii)

    if parall == -1:
        diskfil = disk_avail(funcPath)

        if diskfil < 0.2:
            print('more than %s of disk space is already used!' %
                  (str(diskfil) + '%'))
            exit()

        subprocess.run([
            mpipath, '-np',
            '%d' % anzcores, bin_path + 'V18_PAR/mpi_quaf_v6'
        ])
        subprocess.run([bin_path + 'V18_PAR/sammel'])
        subprocess.call('rm -rf DMOUT.*', shell=True)
    else:
        subprocess.run([bin_path + 'QUAFL_N.exe'])

    subprocess.call('cp -rf INQUA_N INQUA_N_V18', shell=True)
    if tnnii == 11:
        n3_inlu(8, fn='INLU', fr=lfrag, indep=parall)
        os.system(bin_path + 'DRLUD.exe')
        n3_inob(sfrag, 15, fn='INOB', indep=parall)
        os.system(bin_path + 'DROBER.exe')

        n3_inqua_N(intwi=intws, relwi=relws, potf=potNNN, inquaout='INQUA_N')
        parallel_mod_of_3inqua(lfrag,
                               sfrag,
                               infile='INQUA_N',
                               outfile='INQUA_N',
                               tni=1,
                               einzel_path=funcPath + '/')

        if parall == -1:
            diskfil = disk_avail(funcPath)

            if diskfil < 0.2:
                print('more than %s of disk space is already used!' %
                      (str(diskfil) + '%'))
                exit()

            subprocess.run([
                mpipath, '-np',
                '%d' % anzcores, bin_path + 'UIX_PAR/mpi_drqua_uix'
            ])
            subprocess.run([bin_path + 'UIX_PAR/SAMMEL-uix'])
            subprocess.call('rm -rf DRDMOUT.*', shell=True)
            subprocess.run([
                bin_path + 'TDR2END_PYpoolnoo.exe', 'INEN',
                'OUTPUT_TDR2END_PYpoolnoo', 'MATOUTB'
            ],
                           capture_output=True,
                           text=True)
        else:
            subprocess.run([bin_path + 'DRQUA_AK_N.exe'])
            subprocess.run([bin_path + 'DR2END_AK.exe'])
    elif tnnii == 10:
        if parall == -1:
            subprocess.run([
                bin_path + 'TDR2END_PYpoolnoo.exe', 'INEN',
                'OUTPUT_TDR2END_PYpoolnoo', 'MATOUTB'
            ],
                           capture_output=True,
                           text=True)
        else:
            subprocess.run([bin_path + 'DR2END_AK.exe'])

    subprocess.call('cp -rf INQUA_N INQUA_N_UIX', shell=True)
    NormHam = np.core.records.fromfile('MATOUTB', formats='f8', offset=4)

    return NormHam