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
from sklearn.preprocessing import LabelEncoder
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

# from olivier
oliver_features = [
    "ps_car_13",  #            : 1571.65 / shadow  609.23
    "ps_reg_03",  #            : 1408.42 / shadow  511.15
    "ps_ind_05_cat",  #        : 1387.87 / shadow   84.72
    "ps_ind_03",  #            : 1219.47 / shadow  230.55
    "ps_ind_15",  #            :  922.18 / shadow  242.00
    "ps_reg_02",  #            :  920.65 / shadow  267.50
    "ps_car_14",  #            :  798.48 / shadow  549.58
    "ps_car_12",  #            :  731.93 / shadow  293.62
    "ps_car_01_cat",  #        :  698.07 / shadow  178.72
    "ps_car_07_cat",  #        :  694.53 / shadow   36.35
    "ps_ind_17_bin",  #        :  620.77 / shadow   23.15
    "ps_car_03_cat",  #        :  611.73 / shadow   50.67
    "ps_reg_01",  #            :  598.60 / shadow  178.57
    "ps_car_15",  #            :  593.35 / shadow  226.43
    "ps_ind_01",  #            :  547.32 / shadow  154.58
    "ps_ind_16_bin",  #        :  475.37 / shadow   34.17
    "ps_ind_07_bin",  #        :  435.28 / shadow   28.92
    "ps_car_06_cat",  #        :  398.02 / shadow  212.43
    "ps_car_04_cat",  #        :  376.87 / shadow   76.98
    "ps_ind_06_bin",  #        :  370.97 / shadow   36.13
    "ps_car_09_cat",  #        :  214.12 / shadow   81.38
    "ps_car_02_cat",  #        :  203.03 / shadow   26.67
    "ps_ind_02_cat",  #        :  189.47 / shadow   65.68
    "ps_car_11",  #            :  173.28 / shadow   76.45
    "ps_car_05_cat",  #        :  172.75 / shadow   62.92
    "ps_calc_09",  #           :  169.13 / shadow  129.72
    "ps_calc_05",  #           :  148.83 / shadow  120.68
    "ps_ind_08_bin",  #        :  140.73 / shadow   27.63
    "ps_car_08_cat",  #        :  120.87 / shadow   28.82
    "ps_ind_09_bin",  #        :  113.92 / shadow   27.05
    "ps_ind_04_cat",  #        :  107.27 / shadow   37.43
    "ps_ind_18_bin",  #        :   77.42 / shadow   25.97
    "ps_ind_12_bin",  #        :   39.67 / shadow   15.52
    "ps_ind_14",  #            :   37.37 / shadow   16.65
    "ps_car_11_cat" # Very nice spot from Tilii : https://www.kaggle.com/tilii7
]


combs = [
    ('ps_reg_01', 'ps_car_02_cat'),
    ('ps_reg_01', 'ps_car_04_cat'),
]

comb_cols = ['comb_{}'.format(i) for i in range(len(combs))]

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


def load_train_test():
    train_df = pd.read_csv(fp_train, index_col='id')
    test_df = pd.read_csv(fp_test, index_col='id')

    add_combs(train_df, test_df)
    train_df = preprocess_df_for_xgb(train_df)
    test_df = preprocess_df_for_xgb(test_df)

    print 'Loaded!'

    return train_df, test_df

def preprocess_df_for_xgb(df):
    df = df.replace(-1, np.nan)
    df['ps_car_13_x_ps_reg_03'] = df['ps_car_13'] * df['ps_reg_03']
    good_cols = oliver_features+[target]+comb_cols
    for col in df.columns:
        if col not in good_cols:
            del df[col]
    df = add_small_categ_dummies(df)
    return df


def add_small_categ_dummies(df):
    dummies_cols = []
    for col in lo_categoricals:
        if col not in df.columns:
            continue
        dummies_cols.append(col)
        df['{}_orig'.format(col)] = df[col]
    df = pd.get_dummies(df, dummy_na=True, columns=dummies_cols, sparse=True)
    for col in lo_categoricals:
        if col not in df.columns:
            continue
        df[col] = df['{}_orig'.format(col)]

    return df

def add_combs(train, test):
    tmp='tmp'
    for i, (a,b) in enumerate(combs):
        for df in [train, test]:
            df[tmp] = df[a].apply(str)+'_'+df[b].apply(str)

        ll = list(train[tmp].unique())+list(test[tmp].unique())
        enc = LabelEncoder()
        enc.fit(ll)
        for df in [train, test]:
            df['comb_{}'.format(i)]=enc.transform(df[tmp])

    for df in [train, test]:
        del df[tmp]

    print 'Added combs'

def hcc_encode(train_df, test_df, variable, binary_target, k=5, f=1, g=1, r_k=0.01, folds=5):
    """
    See "A Preprocessing Scheme for High-Cardinality Categorical Attributes in
    Classification and Prediction Problems" by Daniele Micci-Barreca
    """
    prior_prob = train_df[binary_target].mean()
    hcc_name = "_".join(["hcc", variable, binary_target])

    skf = StratifiedKFold(folds)
    for big_ind, small_ind in skf.split(np.zeros(len(train_df)), train_df[binary_target]):
        big = train_df.iloc[big_ind]
        small = train_df.iloc[small_ind]
        grouped = big.groupby(variable)[binary_target].agg({"size": "size", "mean": "mean"})
        grouped["lambda"] = 1 / (g + np.exp((k - grouped["size"]) / f))
        grouped[hcc_name] = grouped["lambda"] * grouped["mean"] + (1 - grouped["lambda"]) * prior_prob

        if hcc_name in small.columns:
            del small[hcc_name]
        small = pd.merge(small, grouped[[hcc_name]], left_on=variable, right_index=True, how='left')
        small.loc[small[hcc_name].isnull(), hcc_name] = prior_prob
        small[hcc_name] = small[hcc_name] * np.random.uniform(1 - r_k, 1 + r_k, len(small))
        train_df.loc[small.index, hcc_name] = small[hcc_name]

    grouped = train_df.groupby(variable)[binary_target].agg({"size": "size", "mean": "mean"})
    grouped["lambda"] = 1 / (g + np.exp((k - grouped["size"]) / f))
    grouped[hcc_name] = grouped["lambda"] * grouped["mean"] + (1 - grouped["lambda"]) * prior_prob

    test_df = pd.merge(test_df, grouped[[hcc_name]], left_on=variable, right_index=True, how='left')
    test_df.loc[test_df[hcc_name].isnull(), hcc_name] = prior_prob

    return train_df, test_df, hcc_name


def Preprocessor(train, test):
    for col in hi_categoricals+comb_cols:
        if col not in train.columns:
            continue
        print col
        train, test, tmp = hcc_encode(train, test, col, target)

    return train, test


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
    for col in train_df.columns:
        print col

    train_df, test_df = Preprocessor(train_df, test_df)
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
out_fp = 'submitting_with_oliver_features_seed_110.hdf5'

do_predict(out_fp, params_fp)
