import numpy as np
from functools import partial

from sacred import Experiment
from sacred.utils import apply_backspaces_and_linefeeds
from sacred.observers import MongoObserver

from mongodburi import mongo_uri # A uri string refer to mongodb

import incense
from incense import ExperimentLoader

from data import gen_random, gen_clusters, gen_topics, gen_scholar_net, gen_movielens
import intra


def new_exp(db_name='sacred'):
    ex = Experiment('jupyter_ex', interactive=True)
    ex.captured_out_filter = apply_backspaces_and_linefeeds
    ex.observers.append(MongoObserver(url=mongo_uri, db_name=db_name))

    @ex.config
    def my_config():
        which='random'
        whichalgo='GE|GELS|GV|LSI|LSG|MC|RN'
        dataset = ''
        alphas = [0.95]
        nvec, ncls, ncls_per_vec, nsel, dim = 0, 0, 0, 0, 0
        noise = 0
        docname, passes, topic_threshold = '', 20, 0.25
        sz_clique, lambd = 0, 1

    @ex.main
    def my_main(_run, which, whichalgo, dataset, alphas,
                nvec, ncls, ncls_per_vec, nsel, dim,
                noise,
                docname, passes, topic_threshold,
                sz_clique, lambd):
        # _run.log_scalar(metric_name, value[, step])


        # Datasets
        if which == 'random':
            vecs, clss, vec2cls = gen_random(nvec=nvec, ncls=ncls, ncls_per_vec=ncls_per_vec, dim=dim)
            intra_ = intra.Intra(vecs, vec2cls, len(clss), nsel, metric='euclidean', eps=1)
        elif which == 'proto':
            proto, vecs, vec2cls = gen_clusters(nvec=nvec, ncls=ncls, dim=dim, noise=noise)
            intra_ = intra.Intra(vecs, vec2cls, ncls, nsel, metric='euclidean', eps=1)
        elif which == 'topic':
            dv, doc2cls, ntopic, _ = gen_topics(docname, ncls, passes, topic_threshold)
            intra_ = intra.Intra(dv, doc2cls, ntopic, nsel, metric='cosine', eps=1)
            nvec = len(dv)
        elif which == 'scholar':
            kws, i2c, coms, i2v, vs, c2i = gen_scholar_net(sz_clique)
            intra_ = intra.Intra(np.arange(len(kws)), i2c, len(coms), nsel, cls2vec=c2i,
                                 dist=partial(intra.dist_jaccard, i2v=i2v),
                                 quality=intra.Quality(i2v),
                                 tradeoff=lambd, eps=1)
            nvec = len(kws)
            ncls = len(coms)
        elif which == 'movielens':
            vecs, tag2t, t2tag, t2c, cls2c, t2v, v2c = gen_movielens()
            intra_ = intra.Intra(vecs, [t2c[i] for i in range(len(vecs))], len(cls2c), nsel,
                                 metric='cosine',
                                 quality=intra.Quality(t2v),
                                 tradeoff=lambd, eps=1)
            nvec = len(vecs)
            ncls = len(cls2c)
        else:
            raise ValueError('Which data? {}'.format(which))
        print('nvec, ncls:', nvec, ncls)


        # Algorithms
        whichalgo = whichalgo.lower()
        if 'gv' in whichalgo:
            run_permutation(1, 'gv', intra_.greedy_vertex, intra_, ncls, _run)

        if 'lsi' in whichalgo:
            f = partial(intra_.local_search, intra=True, print_=False)
            run_permutation(1, 'lsi', f, intra_, ncls, _run)

        if 'lsg' in whichalgo:
            f = partial(intra_.local_search, intra=False, print_=False)
            run_once('lsg', f, intra_, ncls, _run)

        if 'rn' in whichalgo:
            f = partial(intra_.random)
            run_once('rn', f, intra_, ncls, _run)

        for alpha in alphas:
            if 'ge' in whichalgo:
                alpha = np.round(alpha,2)
                f = partial(intra_.greedy_edge, exact=False, alpha=alpha)
                sel = run_once('ge{}'.format(int(alpha*100)), f, intra_, ncls, _run)

                if 'gels' in whichalgo:
                    f = partial(intra_.local_search, init=sel, intra=True, print_=False)
                    run_once('gels{}'.format(int(alpha*100)), f, intra_, ncls, _run)

        if 'mc' in whichalgo:
            if intra_.quality is not None:
                run_permutation(1, 'mc', intra_.max_coverage, intra_, ncls, _run)

    return ex


def get_loader(db_name='sacred'):
    loader = ExperimentLoader(
        mongo_uri=mongo_uri,
        db_name=db_name
    )
    return loader


def run_once(name, f, intra_, ncls, _run):
    sel = f()
    disp = intra_.get_intra_disp(sel)
    v = np.average(disp)
    v_ = v
    if _run is not None:
        [_run.log_scalar('{}'.format(name), disp[i], i) for i in range(ncls)]
        [_run.log_scalar('P{}'.format(name), len(sel[i]), i) for i in range(ncls)]
        if intra_.quality is not None:
            intra_.quality.base(None)
            q = intra_.quality.quality(np.concatenate(sel))
            z = q + intra_.tradeoff*np.sum(disp)
            _run.log_scalar('Q{}'.format(name), q)
            _run.log_scalar('Z{}'.format(name), z)
            v_ = z

    print('{}: {:.2f}'.format(name, v_))

    return sel


def run_permutation(nth, name, f, intra_, ncls, _run, ncls_th=10):
    if ncls > ncls_th:
        run_once(name, f, intra_, ncls, _run)
        return None

    sels, disps, vs = intra_.permutation(nth, f)

    if _run is not None:
        if intra_.quality is None:
            disp_min = disps[np.argmin(vs)]
            disp_max = disps[np.argmax(vs)]
            sel_min = sels[np.argmin(vs)]
            sel_max = sels[np.argmax(vs)]
            [_run.log_scalar('{}_min'.format(name), disp_min[i], i) for i in range(ncls)]
            _run.log_scalar('{}_ave'.format(name), np.average(vs))
            [_run.log_scalar('{}_max'.format(name), disp_max[i], i) for i in range(ncls)]
            [_run.log_scalar('P{}_min'.format(name), len(sel_min[i]), i) for i in range(ncls)]
            [_run.log_scalar('P{}_max'.format(name), len(sel_max[i]), i) for i in range(ncls)]
            vs_ = vs
        else:
            intra_.quality.base(None)
            qs = [intra_.quality.quality(np.concatenate(sel)) for sel in sels]
            zs = [q + intra_.tradeoff*np.sum(disp) for q,disp in zip(qs, disps)]
            disp_min = disps[np.argmin(zs)]
            disp_max = disps[np.argmax(zs)]
            sel_min = sels[np.argmin(zs)]
            sel_max = sels[np.argmax(zs)]

            _run.log_scalar('Q{}_min'.format(name), qs[np.argmin(zs)])
            _run.log_scalar('Q{}_min'.format(name), np.average(qs))
            _run.log_scalar('Q{}_max'.format(name), qs[np.argmax(zs)])
            _run.log_scalar('Z{}min'.format(name), np.min(zs))
            _run.log_scalar('Z{}_min'.format(name), np.average(zs))
            _run.log_scalar('Z{}max'.format(name), np.max(zs))
            [_run.log_scalar('P{}_min'.format(name), len(sel_min[i]), i) for i in range(ncls)]
            [_run.log_scalar('P{}_max'.format(name), len(sel_max[i]), i) for i in range(ncls)]
            vs_ = zs

    print('{}: ave {:.2f}, min {:.2f}, max {:.2f}'.format(name,
                                            np.average(vs_),
                                            np.min(vs_),
                                            np.max(vs_)))

    return sels, disps, vs
