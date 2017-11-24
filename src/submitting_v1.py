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

fp_train = '../../../data/porto/train.csv'
fp_test = '../../../data/porto/test.csv'

target = 'target'

sizes = {'train': 595212, 'test': 892816}


class Experiment(IsDescription):
    global_id = Int32Col()
    performance = Float32Col()

    probs = Float32Col(shape=(sizes['test'],))
    indexes = Int32Col(shape=(sizes['test'],))

    params = StringCol(1000)


def shuffle_df(df, random_state):
    np.random.seed(random_state)
    return df.iloc[np.random.permutation(len(df))]


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
    df = pd.get_dummies(df, columns=cat)
    # del df['id']
    return df


def load_train_test():
    train_df = pd.read_csv(fp_train, index_col='id')
    test_df = pd.read_csv(fp_test, index_col='id')

    train_df = preprocess_df_for_xgb(train_df)
    test_df = preprocess_df_for_xgb(test_df)

    return train_df, test_df


class Hdf5Writer():
    def __init__(self, out_fp, table_name, row_class):
        self.out_fp = out_fp
        self.table_name = table_name
        self.row_class = row_class

    def __enter__(self):
        self.file = open_file(self.out_fp, 'a')
        self.group = self.file.create_group("/", 'experiments', '')
        self.table = self.file.create_table(self.group, self.table_name, self.row_class, "")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.file.close()

    def write(self, m):
        exp = self.table.row
        for k, v in m.items():
            exp[k] = v
        exp.append()
        self.table.flush()


def Postprocessor(w):
    for k in ['params']:
        w[k] = json.dumps(w[k])

    return w


class Exiter():
    def __init__(self, fp):
        self.fp = fp

    def must_we_exit(self):
        if not os.path.isfile(self.fp):
            self.clear()

        with open(self.fp) as f:
            ll = f.readlines()
            close = len(ll) != 0

        if close:
            self.clear()

        return close

    def clear(self):
        with open(self.fp, 'w'):
            pass


def NativeXgbEst(train, test, target, params):
    params = params.copy()
    num_boost_round = params['num_boost_round']
    del params['num_boost_round']
    train, test = train.copy(), test.copy()

    train_target = train[target]

    del train[target]

    d_train = xgb.DMatrix(train, train_target)
    d_test = xgb.DMatrix(test)
    booster = xgb.train(params, d_train, num_boost_round=num_boost_round)

    probs = booster.predict(d_test)
    additional_info = {}
    return probs, additional_info


def perform_submit(train_df, test_df, params_list, est, postprocessor, metric, writer, target, exiter):
    counter = 0
    for params in tqdm(params_list):
        if exiter.must_we_exit():
            exiter.clear()
            sys.exit(1)

        t = time()
        probs, additional_info = est(train_df, test_df, target, params)
        w = {
            'probs': probs,
            'performance': None,
            'params': params,
            'global_id': counter
        }
        w = postprocessor(w)
        writer.write(w)
        counter += 1
        print 'time={}'.format(time() - t)


def do_predict(out_fp, params_fp):
    seed = 110
    np.random.seed(seed)
    train_df, test_df = load_train_test()
    print 'Columns num {}'.format(len(train_df.columns))
    train_df = shuffle_df(train_df, seed)
    est = NativeXgbEst
    postprocessor = Postprocessor
    metric = gini_normalized
    exiter = Exiter('exit.txt')
    with open(params_fp) as f:
        params_list = json.load(f)

    with Hdf5Writer(out_fp, 'experiments', Experiment) as writer:
        perform_submit(train_df, test_df, params_list, est, postprocessor, metric, writer, target, exiter)

    print 'Done!'


params_fp = '../params/submitting_params.json'
out_fp = 'submitting_v2_seed_110.hdf5'

do_predict(out_fp, params_fp)
