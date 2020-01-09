import numpy as np
import pandas as pd
import matplotlib
import matplotlib.cm as cm


md = {'nvec': 'N',
      'nsel': 'P',
      'dim': 'D',
      'noise': '\sigma',
      'topic_threshold': 'th',

      'ge10_mean': 'alpha=0.1',
      'ge20_mean': 'alpha=0.2',
      'ge30_mean': 'alpha=0.3',
      'ge40_mean': 'alpha=0.4',
      'ge50_mean': 'alpha=0.5',
      'ge60_mean': 'alpha=0.6',
      'ge70_mean': 'alpha=0.7',
      'ge80_mean': 'alpha=0.8',
      'ge90_mean': 'alpha=0.9',
      'ge95_mean': 'alpha=0.95',

      'ge_mean': 'GE',
      'gels_mean': 'GELS',
      'ge_min': 'GEmin',
      'ge_ave': 'GEavg',
      'ge_max': 'GEmax',
      'gels_min': 'GELSmin',
      'gels_ave': 'GELSavg',
      'gels_max': 'GELSmax',
      'gv_mean': 'GV',
      'gv_min_mean': 'GVmin',
      'gv_ave_mean': 'GVavg',
      'gv_max_mean': 'GVmax',
      'lsi_mean': 'LSI',
      'lsi_min_mean': 'LSImin',
      'lsi_ave_mean': 'LSIavg',
      'lsi_max_mean': 'LSImax',
      'lsg_mean': 'LSG',
      'rn_mean': 'RN',

      'ge_var': 'GE',
      'gels_var': 'GELS',
      'gv_min_var': 'GVmin',
      'gv_ave_var': 'GVavg',
      'gv_max_var': 'GVmax',
      'lsi_min_var': 'LSImin',
      'lsi_ave_var': 'LSIavg',
      'lsi_max_var': 'LSImax',
      'lsg_var': 'LSG',

      'Zge95_mean': 'QDGE',
      'Zgv_mean': 'QDGV',
      'Zmc_mean': 'QDMC',
      'Qge95_mean': 'QGE',
      'Qgv_mean': 'QGV',
      'Qmc_mean': 'QMC',
      'ge95_sum': 'DGE',
      'gv_sum': 'DGV',
      'mc_sum': 'DMC',
      'Pge95_mean': 'PGE',
      'Pgv_mean': 'PGV',
      'Pmc_mean': 'PMC',
     }


def get_colors_by_cls(vec2cls):
    '''A vector belongs to ONLY one cluster'''
    vec2cls1 = np.concatenate(vec2cls)
    norm = matplotlib.colors.Normalize(vmin=min(vec2cls1), vmax=max(vec2cls1), clip=True)
    mapper = cm.ScalarMappable(norm=norm, cmap=cm.rainbow)
    return mapper.to_rgba(vec2cls1)


def highlight_max_min(s, start):
    '''
    highlight the maximum in a Series yellow.
    # df.columns = list(map(lambda x: md[x], df.columns))
    # print(df.round(2).to_latex())
    '''
    is_max = s == s[start:].max()
    is_min = s == s[start:].min()
    return ['color: green' if vmax else ('color: red' if vmin else '') \
            for vmax,vmin in zip(is_max,is_min)]


def relative_tbl(t, beg, end=None):
    for i in range(len(t)):
        line = t.iloc[i]
        mx = line[beg:end].max()
        t.iloc[i,beg:end] = line[beg:end] / mx
    return t


def aggregate_tbl(df, names=['ge','gels'], alphas=[10,30,50,70,95]):
    cols = [['{}{}_mean'.format(name,i) for i in alphas] for name in names]
    cols_all = np.concatenate(cols)

    others = df[df.columns[~df.columns.isin(cols_all) & df.columns.str.contains('_mean')]]
    params = df[df.columns[~df.columns.isin(cols_all) & ~df.columns.str.contains('_mean')]]
    #others = df[df.columns[~df.columns.isin(cols_all) & df.columns.str.contains('_')]]
    #params = df[df.columns[~df.columns.isin(cols_all) & ~df.columns.str.contains('_')]]

    mid = pd.concat(sum([[df[c].min(1),df[c].mean(1),df[c].max(1)] for c in cols], []), ignore_index=True, sort=False, axis=1)
    mid_cols = np.concatenate([['{}_min'.format(name),'{}_ave'.format(name),'{}_max'.format(name)] for name in names])
    newdf = pd.concat([params, mid, others],
              ignore_index=True, sort=False, axis=1)
    newdf.columns=np.concatenate([params.columns,mid_cols,others.columns])
    return newdf


def bold_tab(s, beg=0, end=None, max_=True):
    f = max if max_ else min

    for line in s.splitlines():
        if '&' not in line:
            print(line)
            continue

        es = [e.rstrip('\\') if e.endswith('\\') else e for e in line.split('&')]

        #print('test: ', es[beg:end])
        try:
            nums = list(map(lambda x: float(x), es[beg:end]))
        except:
            print(line)
            continue

        ext = f(nums)
        newline = line.replace(str(ext), '\\textbf{{{}}}'.format(ext))
        print(newline)
