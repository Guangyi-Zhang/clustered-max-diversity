import numpy as np
from scipy.spatial import distance

import gensim
import gensim.downloader as api
import text
from sklearn.datasets import fetch_20newsgroups

from os import path
import pickle


wv = None


def gen_random(nvec, ncls, ncls_per_vec, dim):    
    vecs = np.random.rand(nvec, dim)
    clss = np.arange(ncls)
    vec2cls = np.random.choice(clss, (nvec, ncls_per_vec))
    return vecs, clss, vec2cls


def gen_clusters(nvec, ncls, dim, noise):
    proto = np.random.rand(ncls, dim)
    vecs = []
    vec2cls = []
    for c in range(ncls):
        p = proto[c]
        vecs.append(p + noise*np.random.randn(nvec//ncls, dim))
        [vec2cls.append([c]) for _ in range(nvec//ncls)]
    vecs = np.concatenate(vecs)
    
    for i,v in enumerate(vecs):
        mat = distance.cdist(v.reshape((1,-1)), proto, metric='euclidean', p=2)
        ind = np.argmin(mat)
        if ind not in vec2cls[i]:
            vec2cls[i].append(ind)
            
    return proto, vecs, vec2cls


def gen_topics(name, ntopic, passes, topic_threshold):
    with open('dataset/topic_{}_ntopic{}_pass{}_th{}'.format(name,ntopic,passes,topic_threshold), 'rb') as fin:
        dv, doc2cls, ntopic, lda_model = pickle.load(fin)
    return dv, doc2cls, ntopic, lda_model


def _extract_topics(topic_val, threshold):
    ts = [t for t, v in topic_val if v >= threshold]
    if len(ts) == 0:
        i = np.argmax([v for t, v in topic_val])
        ts = [topic_val[i][0]]
    return ts


def gen_scholar_net(k):
    with open(path.join('dataset', 'aminer-communities-k{}-w2.pkl'.format(k)), 'rb') as f:
        vs, kws, i2c, kw2v, clss, c2i = pickle.load(f)

    i2kw = dict()
    for i,kw in enumerate(kws):
        i2kw[i] = kw

    i2v = dict()
    for i,kw in enumerate(kws):
        i2v[i] = kw2v[kw]

    return kws, i2c, clss, i2v, vs, c2i


def gen_movielens():
    '''v starts from 1.'''
    with open(path.join('dataset', 'movielens.pkl'), 'rb') as f:
        vecs, tag2t, t2tag, t2c, cls2c, t2v, v2c = pickle.load(f)

    return vecs, tag2t, t2tag, t2c, cls2c, t2v, v2c


def gen_topics_from_docs(name, docs, ntopic, passes, topic_threshold):
    ndoc = len(docs)
    print('ndoc: ', ndoc)
    processed_docs = [text.preprocess(doc) for doc in docs]
    print('processed_docs: ', len(processed_docs))
    dictionary = gensim.corpora.Dictionary(processed_docs)
    dictionary.filter_extremes(no_below=15, no_above=0.1, keep_n= 100000)
    print('dictionary: ', len(dictionary))
    bow_corpus = [dictionary.doc2bow(doc) for doc in processed_docs]

    lda_model = gensim.models.LdaMulticore(bow_corpus,
                                       num_topics = ntopic,
                                       id2word = dictionary,
                                       passes = passes,
                                       workers = 2)

    doc2cls = [_extract_topics(lda_model[d], topic_threshold) for d in bow_corpus]

    # fasttext_model300 = api.load('fasttext-wiki-news-subwords-300')
    # glove_model300 = api.load('glove-wiki-gigaword-300')
    global wv
    if wv is None:
        wvmodel = api.load('word2vec-google-news-300')
        wv = wvmodel.wv
        del wvmodel

    # doc vectors
    dv = [np.sum([wv[dictionary[wid]]*cnt for (wid,cnt) in doc if dictionary[wid] in wv.vocab], axis=0) for doc in bow_corpus]

    good_idx = [i for i,d in enumerate(dv) if d.shape!=()]
    dv = np.stack([dv[i] for i in good_idx])
    doc2cls = [doc2cls[i] for i in good_idx]
    bow_corpus = [bow_corpus[i] for i in good_idx]
    docs = [docs[i] for i in good_idx]
    processed_docs = [processed_docs[i] for i in good_idx]
    
    with open('dataset/topic_{}_ntopic{}_pass{}_th{}'.format(name,ntopic,passes,topic_threshold), 'wb') as fout:
        pickle.dump([dv, doc2cls, ntopic, lda_model], fout)