import os
from collections import OrderedDict, defaultdict

import pandas as pd
import numpy as np
import sys
import xgboost as xgb
import json
from time import time

from pandas.core.common import SettingWithCopyWarning
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from tables import *
from sklearn.metrics import roc_auc_score
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=SettingWithCopyWarning)

pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 10000)
pd.set_option('display.max_rows', 5000)
pd.set_option('display.max_colwidth', 5000)

sizes = {'train': 595212, 'test': 892816}
target = 'target'
categoricals = [
    'ps_car_01_cat',
    'ps_car_02_cat',
    'ps_car_03_cat',
    'ps_car_04_cat',
    'ps_car_05_cat',
    'ps_car_06_cat',
    'ps_car_07_cat',
    'ps_car_08_cat',
    'ps_car_09_cat',
    'ps_car_10_cat',
    'ps_car_11_cat',
    'ps_ind_02_cat',
    'ps_ind_04_cat',
    'ps_ind_05_cat'
]

lo_categoricals=[
    'ps_car_02_cat', #3
    'ps_car_03_cat', #3
    'ps_car_05_cat', #3
    'ps_car_07_cat', #3
    'ps_car_08_cat', #2
    'ps_car_09_cat', #6
    'ps_car_10_cat', #3
    'ps_ind_02_cat', #5
    'ps_ind_04_cat', #3
    'ps_ind_05_cat', #8
]

hi_categoricals = [
    'ps_car_01_cat', #13
    'ps_car_04_cat', #10
    'ps_car_06_cat', #18
    'ps_car_11_cat', #104
]

combs = [
    ('ps_reg_01', 'ps_car_02_cat'),
    ('ps_reg_01', 'ps_car_04_cat'),
]

comb_cols = ['comb_{}'.format(i) for i in range(len(combs))]

fp_train = '../../../data/porto/train.csv'
fp_test = '../../../data/porto/test.csv'

def gini(actual, pred):
    assert (len(actual) == len(pred))
    all = np.asarray(np.c_[actual, pred, np.arange(len(actual))], dtype=np.float)
    all = all[np.lexsort((all[:, 2], -1 * all[:, 1]))]
    totalLosses = all[:, 0].sum()
    giniSum = all[:, 0].cumsum().sum() / totalLosses

    giniSum -= (len(actual) + 1) / 2.
    return giniSum / len(actual)


def gini_normalized(actual, pred):
    return gini(actual, pred) / gini(actual, actual)

def preprocess_df_for_xgb(df):
    df = df.replace(-1, np.nan)
    cat = list(set(categoricals).intersection(set(df.columns)))
    # df = pd.get_dummies(df, columns=cat)
    # del df['id']
    return df

def load_train_test():
    train_df = pd.read_csv(fp_train, index_col='id')
    test_df = pd.read_csv(fp_test, index_col='id')

    train_df = preprocess_df_for_xgb(train_df)
    test_df = preprocess_df_for_xgb(test_df)

    return train_df, test_df


def cv_to_json(x):
    keys = {
        'indexes',
        'probs',
        'folds',
        'performance',
        'params',
        'folds_info',
        'param_id',
        'run_id',
        'global_id',
    }
    res = {k: x[k] for k in keys}

    for k in ('params', 'folds_info'):
        res[k] = json.loads(res[k])

    return res


def submit_to_json(x):
    keys = {
        'probs',
        'params',
        'global_id',
    }
    res = {k: x[k] for k in keys}

    for k in ['params']:
        res[k] = json.loads(res[k])

    return res


def performance_df_from_cv_res(bl):
    res = []
    param_names = set(bl[0]['params'].keys()).difference({'eval_metric', 'objective', 'tree_method'})
    for i in range(len(bl) / 2):
        a = bl[2 * i]
        b = bl[2 * i + 1]
        assert a['params'] == b['params']
        perf = list(a['performance']) + list(b['performance'])
        best_ntree = [x['best_ntree_limit'] for x in a['folds_info']]+\
            [x['best_ntree_limit'] for x in b['folds_info']]
        m = [
            ('mean', np.mean(perf)),
            ('std', np.std(perf))
        ]
        params = a['params']
        for n in param_names:
            m.append((n, a['params'][n]))
        m.append(('limit', best_ntree))
        median = np.median(best_ntree)
        m.append(('limit_median', median))
        params['num_boost_round'] = int(median)
        m.append(('performance', perf))
        m.append(('params', params))
        res.append(OrderedDict(m))

    return pd.DataFrame(res).sort_values('mean', ascending=False)

def performance_df_from_cv_res_new(bl):
    res = []
    param_names = set(bl[0]['params'].keys()).difference({'eval_metric', 'objective', 'tree_method'})
    for i in range(len(bl)):
        a = bl[i]
        perf = list(a['performance'])
        best_ntree = [x['best_ntree_limit'] for x in a['folds_info']]
        m = [
            ('mean', np.mean(perf)),
            ('std', np.std(perf))
        ]
        params = a['params']
        for n in param_names:
            m.append((n, a['params'][n]))
        m.append(('limit', best_ntree))
        median = np.median(best_ntree)
        m.append(('limit_median', median))
        params['num_boost_round'] = int(median)
        m.append(('performance', perf))
        m.append(('params', params))
        res.append(OrderedDict(m))

    return pd.DataFrame(res).sort_values('mean', ascending=False)


def load_cv_results(fp):
    with open_file(fp, 'r') as ff:
        table = ff.root.experiments.experiments
        return [cv_to_json(x) for x in table.iterrows()]


def load_submit_results(fp):
    with open_file(fp, 'r') as ff:
        table = ff.root.experiments.experiments
        return [submit_to_json(x) for x in table.iterrows()]


def df_from_cv_entry(entry):
    return pd.DataFrame(OrderedDict([('probs', entry['probs']), ('folds', entry['folds'])]), index=entry['indexes'])

def create_submit_df(test_df, s_entry, fp):
    test_df[target] = s_entry['probs']
    test_df[[target]].to_csv(fp, index_label='id')

def MeanStacker(probs, folds):
    return np.mean(probs, axis=0)

def GMeanStacker(probs, folds):
    return np.exp(np.mean(np.log(probs), axis=0))


def RankStacker(probs, folds):
    sz = len(probs)
    m = {'folds':folds}
    for i, p in enumerate(probs):
        m[i] = p
    df = pd.DataFrame(m)
    for f_id in df['folds'].unique():
        for i in range(sz):
            fold = df[df['folds'] == f_id]
            df.loc[fold.index, i] = fold[i].rank()

    return list(df[range(sz)].mean(axis=1) / len(df))


def get_performance_of_avg(bl, train_df, indexes, stacker):
    train_df = train_df[[target]]
    bl=[bl[i] for i in indexes]
    indexes = bl[0]['indexes']
    folds = bl[0]['folds']
    probs = stacker([x['probs'] for x in bl], folds)
    df = pd.DataFrame({'probs':probs, 'folds':folds}, index=indexes)
    df = pd.merge(df, train_df, left_index=True, right_index=True)
    assert len(train_df) == len(df)
    res = []
    for i in df['folds'].unique():
        f = df[df['folds']==i]
        g = gini_normalized(f[target], f['probs'])
        res.append(g)
        # print g

    # print 'Avg = {}'.format(np.mean(res))
    return np.mean(res)

def explore_corr(bl):
    bl.sort(key=lambda s: np.mean(s['performance']), reverse=True)
    df = pd.DataFrame({i:bl[i]['probs'] for i in range(len(bl))})
    return df.corr()

def get_performance_of_avg_batch(bl, train_df, max_num):
    m = []
    g=[]
    r=[]
    ii = range(1, max_num + 1)
    for i in tqdm(ii):
        m.append(get_performance_of_avg(bl, train_df, df[:i].index, MeanStacker))
        g.append(get_performance_of_avg(bl, train_df, df[:i].index, GMeanStacker))
        r.append(get_performance_of_avg(bl, train_df, df[:i].index, RankStacker))

    plt.plot(ii, m, label='m')
    plt.plot(ii, g, label='g')
    plt.plot(ii, r, label='r')
    return pd.DataFrame(OrderedDict([
        ('m',m), ('g', g), ('r', r)
    ]), index=ii)


cv_fp = '../../../data/porto/grid_cv_res_v2_seed_110.hdf5'
cv_fp_new = '../../../data/porto/grid_cv_res_v3_seed_110.hdf5'
cv_fp_oliver = '../../../data/porto/with_oliver_features_grid_cv_res_seed_110.hdf5'
cv_fp_new_hcc = '../../../data/porto/grid_cv_res_different_hcc_seed_110.hdf5'

ss_fp = '../../../data/porto/submitting_v2_seed_100.hdf5'
ss_fp_new = '../../../data/porto/submitting_v3_seed_110.hdf5'
ss_fp_oliver = '../../../data/porto/submitting_with_oliver_features_seed_110.hdf5'
# ss = load_submit_results(ss_fp_oliver)


fp = '../../../data/porto/optimizing_xgb_d3_e0,1_seed110.hdf5'
bl = load_cv_results(fp)
df = performance_df_from_cv_res_new(bl)

# bl_new = load_cv_results(cv_fp_new)
# df_new = performance_df_from_cv_res(bl_new)
#
# bl_oliver = load_cv_results(cv_fp_oliver)
# df_oliver = performance_df_from_cv_res(bl_oliver)
#
# bl_new_hcc = load_cv_results(cv_fp_new_hcc)
# df_new_hcc = performance_df_from_cv_res(bl_new_hcc)

train_df = pd.read_csv(fp_train, index_col='id')