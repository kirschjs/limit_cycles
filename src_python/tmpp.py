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

    inen_bdg_2(para[0], j=para[3], costr=para[5], fn=inenf, pari=0)

    inqua_2(relw=para[0], ps2=para[2],, inquaout=inqf)

    cmdqua = para[4] + 'QUAFL_N_pop.exe %s %s %s' % (inqf, outputqf,
                                                     quaf_to_end)

    cmdend = para[4] + 'DR2END_AK_pop.exe %s %s %s %s %s' % (
        quaf_to_end, '', inenf, outputef, maoutf)

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
        smartEV, basCond = smart_ev(NormHam, threshold=minCond)

        anzSigEV = len(
            [bvv for bvv in smartEV if para[8][0] < bvv < para[8][1]])

        gsEnergy = smartEV[-1]
        attractiveness = loveliness(gsEnergy, basCond, anzSigEV, minCond)

        os.system('rm -rf ./%s' % inqf)
        os.system('rm -rf ./%s' % inenf)
        os.system('rm -rf ./%s' % outputef)
        os.system('rm -rf ./%s' % outputqf)
        os.system('rm -rf ./%s' % quaf_to_end)
        os.system('rm -rf ./%s' % maoutf)

        #  [ intw, relw, qualREF, gsREF, basCond ]
        send_end.send([
            para[0],
            attractiveness,
            gsEnergy,
            basCond,
        ])

    except:

        os.system('rm -rf ./%s' % inqf)
        os.system('rm -rf ./%s' % inenf)
        os.system('rm -rf ./%s' % outputef)
        os.system('rm -rf ./%s' % quaf_to_end)
        os.system('rm -rf ./%s' % maoutf)

        print(para[6], child_id)
        print(maoutf)
        #  [ intw, relw, qual, gsE, basCond ]
        send_end.send([[], 0.0, 0.0, -42.7331])

def span_population2(anz_civ,
                     fragments,
                     Jstreu,
                     coefstr,
                     funcPath,
                     binPath,
                     mindists=[0.01, 0.01],
                     ini_grid_bounds=[0.01, 9.5, 0.001, 11.5],
                     ini_dims=[4, 4],
                     minC=10**(-8),
                     evWin=[-100, 100]):

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
    mindist_int = mindists[0]
    mindist_rel = mindists[1]

    # lower bound for width parameters '=' IR cutoff (broadest state)
    rWmin = 0.0001

    # orbital-angular-momentum dependent upper bound '=' UV cutoff (narrowest state)
    iLcutoff = [22., 4., 3.]
    rLcutoff = [22., 4., 3.]
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
            itera = 1

            lit_w_tmp = np.abs(
                np.geomspace(start=ini_grid_bounds[0],
                             stop=ini_grid_bounds[1],
                             num=nwint,
                             endpoint=True,
                             dtype=None))

            lit_w_t = []
            while len(lit_w_t) != nwint:

                lit_w_t = [
                    test_width * (1 + 0.51 * (np.random.random() - 1))
                    for test_width in lit_w_tmp
                ]
                dists = [
                    np.linalg.norm(wp1 - wp2) for wp1 in lit_w_t
                    for wp2 in lit_w_t if wp1 != wp2
                ]

                if ((np.min(dists) < mindist_int) &
                    (np.max(lit_w_t) > iLcutoff[0])):
                    lit_w_t = []

                itera += 1
                assert itera <= 180000

            lit_w[frg] = np.sort(lit_w_t)[::-1]

            #  -- relative widths --------------------------------------------------
            lit_w_tmp = np.abs(
                np.geomspace(start=ini_grid_bounds[2],
                             stop=ini_grid_bounds[3],
                             num=nwrel,
                             endpoint=True,
                             dtype=None))

            lit_w_t = []

            while len(lit_w_t) != nwrel:

                lit_w_t = [
                    test_width * (1 + 0.51 * (np.random.random() - 1))
                    for test_width in lit_w_tmp
                ]
                dists = [
                    np.linalg.norm(wp1 - wp2) for wp1 in lit_w_t
                    for wp2 in lit_w_t if wp1 != wp2
                ]

                if ((np.min(dists) < mindist_rel) &
                    (np.max(lit_w_t) > rLcutoff[0])):
                    lit_w_t = []

                itera += 1
                assert itera <= 180000

            lit_rw[frg] = np.sort(lit_w_t)[::-1]

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
            widi, widr, sbas, nnpot, nnnpot, Jstreu, civ, binPath, coefstr,
            minC, evWin
        ])

    os.chdir(funcPath)

    inlu_3(8, fn='INLU', fr=lfrags2, indep=0)
    os.system(binPath + 'DRLUD.exe')
    inlu_3(8, fn='INLUCN', fr=lfrags2, indep=0)
    os.system(binPath + 'LUDW_CN.exe')
    inob_3(sfrags2, 8, fn='INOB', indep=0)
    os.system(binPath + 'KOBER.exe')
    inob_3(sfrags2, 15, fn='INOB', indep=0)
    os.system(binPath + 'DROBER.exe')

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
        if ((cand[2] < 0) & (cand[3] > minC)):
            cfgg = np.transpose(np.array([sfrags2, lfrags2])).tolist()

            cand_list.append([cfgg] + cand)

    cand_list.sort(key=lambda tup: np.abs(tup[2]))

    for cc in samp_ladder:
        print(cc[2:])

    return cand_list, sbas