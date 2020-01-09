import numpy as np
from scipy.spatial import distance
import itertools
from functools import partial
np.random.seed(123)


def dist_jaccard(l1, l2, i2v):
    r = []
    for i1,i2 in itertools.product(l1, l2):
        v1 = i2v[i1]
        v2 = i2v[i2]
        d = 1 - (len(v1.intersection(v2)) / len(v1.union(v2)))
        r.append(d)

    return np.array(r).reshape(len(l1), len(l2))


class Quality(object):
    def __init__(self, i2v):
        self.i2v = i2v
        self.base_ = None

    def base(self, base):
        self.base_ = base
        if self.base_ is None or len(self.base_)==0:
            self.taken = {}
        else:
            self.taken = set.union(*[self.i2v[i] for i in self.base_])

    def quality(self, S):
        totake = set.union(*[self.i2v[i] for i in S])
        return len(totake.difference(self.taken))


class Intra(object):
    def __init__(self, vectors, vec2cls, ncls, nsel, cls2vec=None,
                 metric='euclidean', dist=None, quality=None, tradeoff=1., eps=1.):
        self.eps = eps

        self.vectors = vectors
        self.nvec = len(vectors)
        self.vec2cls = vec2cls
        self.ncls = ncls
        self.nsel = nsel

        self.metric = metric
        self.dist = partial(distance.cdist, metric=metric, p=2) if dist is None else dist
        self.quality = quality
        self.tradeoff = tradeoff

        self.cls2vec = cls2vec
        if self.cls2vec is None:
            self.cls2vec = []
            for c in np.arange(self.ncls):
                vecs = [i if c in clss else -1 for i,clss in enumerate(self.vec2cls)]
                vecs = list(filter(lambda x: x!=-1, vecs))
                self.cls2vec.append(np.array(vecs))

    #############
    # Utilities
    #############
    def get_intra_disp(self, selections):
        vals = []
        for sel in selections:
            vs = self.vectors[sel]
            mat = self.dist(vs, vs)
            vals.append(np.sum(mat))
        return vals

    def permutation(self, nfirst, method):
        disps, sels, vs = [], [], []

        o = np.arange(self.ncls)
        for first3 in itertools.permutations(o, nfirst):
            rest = np.setdiff1d(o, first3)
            p = np.concatenate((first3, rest))
            cls_sel = method(ordered_clusters=p)
            disp = self.get_intra_disp(cls_sel)
            v = np.average(disp)

            disps.append(disp)
            sels.append(cls_sel)
            vs.append(v)

        return sels, disps, vs

    def _check_solution(self, cls_sel, ordered_clusters=None):
        all = np.concatenate(cls_sel)
        all = np.unique(all)
        assert len(all) == sum([len(s) for s in cls_sel]), \
            '{} vs. {}'.format(len(all),
                               sum([len(s) for s in cls_sel]))

        ordered_clusters = ordered_clusters if ordered_clusters is not None else np.arange(self.ncls)
        for c,sel in zip(ordered_clusters, cls_sel):
            for v in sel:
                assert c in self.vec2cls[v]

        if len(all) != self.ncls * self.nsel:
            print('Warning: less selections than specified {}/{}'.format(len(all),
                                                                         self.ncls * self.nsel))

    #############
    # Algorithms
    #############
    def greedy_vertex(self, ordered_clusters=None):
        cls_sel = []
        ordered_clusters = ordered_clusters if ordered_clusters is not None else np.arange(self.ncls)
        for c in ordered_clusters:
            ind = np.setdiff1d(self.cls2vec[c], np.concatenate(cls_sel) if len(cls_sel)>0 else [])
            if len(ind) == 0:
                cls_sel.append([])
                continue
            vs = self.vectors[ind]

            if self.quality is None:
                first = np.random.randint(0, len(vs))
            else:
                self.quality.base(None)
                first = np.argmax([self.quality.quality(np.array([v])) for v in ind])
            sel = np.array([first]) # Selected indice wrt 'ind'

            for i in range(self.nsel-1):
                mat = self.dist(vs, vs[sel])
                mat_ = np.sum(mat, axis=1) # Row sum
                mat_[sel] = -1

                if self.quality is not None:
                    self.quality.base(np.array([ind[_] for _ in sel]))
                    for j in range(len(vs)):
                        q = self.quality.quality(np.array([ind[j]]))
                        mat_[j] = q + self.tradeoff * 2 * mat_[j]

                v = np.argmax(mat_)
                if mat_[v] < 0:
                    break
                else:
                    sel = np.concatenate((sel, [v]))
            cls_sel.append(ind[sel]) # Back to true indice wrt 'vectors'

        self._check_solution(cls_sel, ordered_clusters)
        return cls_sel

    def random(self, ordered_clusters=None):
        ordered_clusters = ordered_clusters if ordered_clusters is not None else np.arange(self.ncls)

        init = [None] * self.ncls
        _init = []
        for c in ordered_clusters:
            _ = np.setdiff1d(self.cls2vec[c], np.concatenate(_init) if len(_init)>0 else [])[:self.nsel]
            init[c] = _ # init is in original order.
            _init.append(_)

        self._check_solution(init)
        return init

    def local_search(self, init=None, intra=True, print_=True, ordered_clusters=None):
        ordered_clusters = ordered_clusters if ordered_clusters is not None else np.arange(self.ncls)
        if init is None:
            init = self.random(ordered_clusters)

        it = 0
        while True:
            found, c, i, candi = self._find_swap_candidate(init, ordered_clusters, intra)
            if not found:
                break
            init[c][i] = candi
            it = it + 1

        if print_:
            print('iterations: {}'.format(it))
        self._check_solution(init)
        return init

    def _find_swap_candidate(self, init, ordered_clusters, intra=True):
        for ic, c in enumerate(ordered_clusters):
            vs_sel = self.vectors[init[c]]
            vs_sel_against = vs_sel if intra else\
                    self.vectors[np.concatenate([init[c] for c in ordered_clusters])]
            mat_sel = self.dist(vs_sel_against, vs_sel)
            val_sel = np.sum(mat_sel, axis=0)

            ind = np.setdiff1d(self.cls2vec[c], np.concatenate(init))
            if len(ind) == 0: continue # No valid swap within this cluster.
            vs = self.vectors[ind]
            nsel_ = len(init[c]) # nsel_ may not equal self.nsel
            nall = sum([len(init[ordered_clusters[i]]) for i in range(self.ncls)])
            for i in range(nsel_):
                if intra:
                    ind_no_i = list(range(nsel_))[0:i] + list(range(nsel_))[i+1:]
                else:
                    before = sum([len(init[ordered_clusters[ic_]]) for ic_ in range(ic)])
                    ind_no_i = list(range(nall))[0:before+i] +\
                        list(range(nall))[before+i+1:]
                vs_sel_i = vs_sel_against[np.array(ind_no_i)]
                mat = self.dist(vs_sel_i, vs)
                vals = np.sum(mat, axis=0)
                candi = np.argmax(vals, axis=0)
                if vals[candi] - val_sel[i] > self.eps:
                    return True,c,i,ind[candi] # i is the i^th selected v in cluster c

        return False,None,None,None

    def max_coverage(self, ordered_clusters=None):
        assert self.quality is not None

        ordered_clusters = ordered_clusters if ordered_clusters is not None else np.arange(self.ncls)
        cls_sel = [[] for _ in ordered_clusters]
        for c in ordered_clusters:
            sel = []
            for _ in range(self.nsel-1):
                base = np.concatenate(cls_sel if len(cls_sel)>0 else [])
                pool = np.setdiff1d(self.cls2vec[c], base)
                if len(pool) == 0: break

                self.quality.base(base)
                u = np.argmax([self.quality.quality(np.array([v])) for v in pool])
                cls_sel[c].append(pool[u])

        self._check_solution(cls_sel)
        return cls_sel

    def greedy_edge(self, exact=True, alpha=1.):
        nedge = self.nsel // 2
        cls_edges = [np.array([], dtype=int) for _ in range(self.ncls)]
        while True:
            # Mark saturated clusters
            clusters = [c if len(es)<nedge*2 else -1 for c,es in enumerate(cls_edges)]
            unsat_clusters = list(filter(lambda x: x!=-1, clusters))
            if len(unsat_clusters) == 0: break

            if exact:
                c, u, v = self._find_edge(clusters, np.concatenate(cls_edges))
            else:
                c, u, v = self._find_edge_x(clusters, cls_edges, alpha)
            if c is None:
                break
            cls_edges[c] = np.concatenate((cls_edges[c], [u,v]))

        # If self.nsel is not an even number
        # TODO

        self._check_solution(cls_edges)
        return cls_edges

    def _find_edge(self, clusters, v_sel):
        e_max = -1
        uv_max = None
        c_max = None
        for c in clusters:
            if c == -1: continue
            ind = np.setdiff1d(self.cls2vec[c], v_sel)
            if len(ind) < 2: continue
            vs = self.vectors[ind]
            uv, e = self._find_diameter(vs, v_sel, ind)
            if e > e_max:
                e_max = e
                uv_max = (ind[uv[0]], ind[uv[1]])
                c_max = c

        if c_max is None: return None,None,None
        return c_max, uv_max[0], uv_max[1]

    def _find_diameter(self, vs, base, index):
        mat = self.dist(vs, vs)

        if self.quality is not None:
            self.quality.base(base)
            for i,j in itertools.product(range(mat.shape[0]), range(mat.shape[1])):
                i_, j_ = index[i], index[j]
                q = self.quality.quality(np.array([i_,j_]))
                mat[i,j] = q + self.tradeoff * (self.nsel-1) *  mat[i,j]

        mat[np.arange(len(vs)), np.arange(len(vs))] = -1 # Diagonal
        uv = np.unravel_index(np.argmax(mat, axis=None), mat.shape)
        return (uv[0],uv[1]), mat[uv[0],uv[1]]

    def _find_edge_x(self, clusters, cls_edges, alpha):
        '''
        alpha: 2/alpha-approx of diameter
        '''
        v_sel = np.concatenate(cls_edges)
        univ = []
        ok_clusters = []
        for c in clusters:
            if c == -1: continue
            ind = np.setdiff1d(self.cls2vec[c], v_sel)
            if len(ind) < 2: continue
            ok_clusters.append(c)
            univ.append(ind)
        if len(ok_clusters)==0: return None,None,None

        # Run quality for 1st endpoint only once
        univ = np.unique(np.concatenate(univ))
        if self.quality is not None:
            self.quality.base(v_sel)
            qs = dict([(v, self.quality.quality(np.array([v]))) for v in univ])

        # Repeat for each cluster
        e_max = -1
        uv_max = None
        c_max = None
        for c in ok_clusters:
            if self.quality is not None: self.quality.base(v_sel) # Reset base every time
            e, uv = self._find_edge_x_1cls(c, cls_edges, alpha, qs_univ=None if self.quality is None else qs)
            if e > e_max:
                e_max = e
                uv_max = uv
                c_max = c

        return c_max, uv_max[0], uv_max[1]

    def _find_edge_x_1cls(self, c, cls_edges, alpha, qs_univ=None):
        v_sel = np.concatenate(cls_edges)
        ind = np.setdiff1d(self.cls2vec[c], v_sel)

        # Select first endpoint
        if self.quality is None:
            if len(cls_edges[c]) == 0:
                u = np.random.randint(0, len(ind))
                u = ind[u]
            else:
                max_ = -1
                mat = self.dist(self.vectors[ind], self.vectors[cls_edges[c]])
                mat_ = np.sum(mat, axis=1) # Row sum
                u = np.argmax(mat_)
                u = ind[u]
        else:
            qs = np.array([qs_univ[v] for v in ind])
            u = np.argmax(qs)
            q_max = qs[u]
            u = ind[u]
            if len(v_sel) > 0:
                good_ind = qs >= q_max * alpha
                ind2 = ind[good_ind]
                if len(ind2) > 1 and len(cls_edges[c]) > 0:
                    mat = self.dist(self.vectors[ind2], self.vectors[cls_edges[c]])
                    mat_ = np.sum(mat, axis=1) # Row sum
                    u = np.argmax(mat_)
                    u = ind2[u]

        base = np.concatenate((v_sel, [u]))
        if self.quality is not None:
            q_u = qs_univ[u]
            self.quality.base(base)

        # Select second endpoint that forms the best edge.
        ind = np.setdiff1d(self.cls2vec[c], base)
        vs = self.vectors[ind]
        v, e = self._find_diameter_x(u, vs, cls_edges[c], ind, alpha, q_u=q_u if self.quality else None)

        return e, (u,v)

    def _find_diameter_x(self, u, vs, base, index, alpha, q_u=None):
        mat = self.dist(vs, np.array([self.vectors[u]])).flatten()

        if self.quality is not None:
            for i in range(mat.shape[0]):
                q = self.quality.quality(np.array([index[i]])) # u included
                mat[i] = q_u + q + self.tradeoff * 2*(self.nsel-1) * mat[i]

        v1 = np.argmax(mat)
        e1 = mat[v1]
        if len(base) == 0:
            return index[v1], e1

        # alpha approx
        good_v_ind = mat >= e1*alpha
        mat2 = self.dist(vs[good_v_ind], self.vectors[base])
        mat2_ = np.sum(mat2, axis=1) # Row sum
        v = np.argmax(mat2_)
        v = np.argwhere(good_v_ind)[v][0]
        e = mat[v]
        assert e >= e1*alpha
        # Only search within this cluster for a good alpha approx.
        return index[v], e
