import os
from collections import OrderedDict, defaultdict

import pandas as pd
import numpy as np
import sys
import xgboost as xgb
import json
from time import time

from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import roc_auc_score

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

def calculate_aucs(df, target):
    # df = df[~df[target].isnull()]
    cols = []
    aucs = []
    coverages = []
    not_zero =[]
    counter = 0
    for col in df.columns:
        # print col
        if counter % 100 == 0:
            print counter
        counter += 1
        cols.append(col)
        bl = df[[col, target]]
        bl = bl[~bl[col].isnull()]
        freq = float(len(bl)) / len(df)
        coverages.append(freq)
        try:
            auc = roc_auc_score(bl[target], bl[col])
            aucs.append(auc)
        except:
            aucs.append(None)
        bl = bl[bl[col]!=0]
        freq = float(len(bl)) / len(df)
        not_zero.append(freq)

    return pd.DataFrame({'col': cols, 'auc': aucs, 'coverage': coverages, 'not_zero':not_zero})[['col', 'auc', 'coverage', 'not_zero']].sort_values(
        by='auc')


def get_importances_df(est, train):
    importances_df = pd.DataFrame(
        OrderedDict([('name', train.columns.values), ('importance', est.feature_importances_)]))
    importances_df.sort_values(by='importance', ascending=False, inplace=True)
    return importances_df


def eval_gini(y_true, y_prob):
    y_true = np.asarray(y_true)
    y_true = y_true[np.argsort(y_prob)]
    ntrue = 0
    gini = 0
    delta = 0
    n = len(y_true)
    for i in range(n - 1, -1, -1):
        y_i = y_true[i]
        ntrue += y_i
        gini += y_i * delta
        delta += 1 - y_i
    gini = 1 - 2 * gini / (ntrue * (n - ntrue))
    return gini


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


def gini_xgboost_eval(preds, dtrain):
    labels = dtrain.get_label()
    return 'gini', -gini_normalized(labels, preds)


def explore_cols(df):
    dd = OrderedDict([
        ('col', list(df.columns)),
        ('dtype', [df[c].dtype for c in df.columns]),
        ('sz', [len(set(df[c])) for c in df.columns]),
        ('sample', [list(set(df[c]))[:9] for c in df.columns])
    ])

    return pd.DataFrame(dd)


def preprocess_df_for_xgb(df):
    cat = list(set(categoricals).intersection(set(df.columns)))
    df = pd.get_dummies(df, columns=cat)
    # del df['id']
    return df


train_df, test_df = pd.read_csv(fp_train), pd.read_csv(fp_test)


def submit_xgb(train_df, test_df, params):
    train_df = preprocess_df_for_xgb(train_df)
    test_df = preprocess_df_for_xgb(test_df)
    print set(train_df.columns).difference(set(test_df.columns))
    est = xgb.XGBClassifier(**params)
    cols = list(train_df.columns)
    cols.remove(target)
    cols.remove('id')
    est.fit(train_df[cols], train_df[target])
    probs = est.predict_proba(test_df[cols])[:, 1]
    test_df['target'] = probs

    test_df[['id', 'target']].to_csv('submission_naive_v2_noid_300.csv', index=False)


# 0.2782496
def xgb_cv(train_df, test_df, seed=42, params={}):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    res = defaultdict(list)
    train_df = preprocess_df_for_xgb(train_df)
    print 'cols_num {}'.format(len(train_df.columns))
    for big_ind, small_ind in skf.split(np.zeros(len(train_df)), train_df[target]):
        t = time()
        est = xgb.XGBClassifier(n_estimators=1000, seed=seed, **params)
        big = train_df.iloc[big_ind]
        small = train_df.iloc[small_ind]
        cols = list(big.columns)
        cols.remove(target)
        cols.remove('id')

        train, train_target = big[cols], big[target]
        test, test_target = small[cols], small[target]
        evals = ([train, train_target], (test, test_target))
        est.fit(train, train_target, eval_set=evals, eval_metric='auc', early_stopping_rounds=100)
        print 'time {}'.format(time() - t)
        xgb_results = est.evals_result()

        train_auc = xgb_results['validation_0']['auc'][est.best_iteration]
        test_auc = xgb_results['validation_1']['auc'][est.best_iteration]
        gini_train = 2 * train_auc - 1
        res['train'].append(gini_train)
        gini_test = 2 * test_auc - 1
        res['test'].append(gini_test)
        res['n_est'].append(est.best_iteration)
        print gini_train, gini_test

        imp = get_importances_df(est, train)

    res = pd.DataFrame(res)
    res.to_csv('results_cv_noid.csv')
    return res


seed = 42
np.random.seed(seed)
MAX_ROUNDS = 400
LEARNING_RATE = 0.07
params_from_kernel = {
    # 'n_estimators': MAX_ROUNDS,
    'max_depth': 4,
    'objective': "binary:logistic",
    'learning_rate': LEARNING_RATE,
    'subsample': .8,
    'min_child_weight': 6,
    'colsample_bytree': .8,
    'scale_pos_weight': 1.6,
    'gamma': 10,
    'reg_alpha': 8,
    'reg_lambda': 1.3
}
# res = xgb_cv(train_df, test_df, seed, params_from_kernel)
submit_xgb(train_df, test_df, params_from_kernel)