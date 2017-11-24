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


# sizes = {'train': 2000, 'test': 892816}


class Experiment(IsDescription):
    param_id = Int32Col()
    run_id = Int32Col()
    global_id = Int32Col()

    performance = Float32Col(shape=(5,))

    probs = Float32Col(shape=(sizes['train'],))
    indexes = Int32Col(shape=(sizes['train'],))
    folds = Int32Col(shape=(sizes['train'],))

    params = StringCol(1000)
    folds_info = StringCol(1000)


def walk_through_grid(grid):
    res = []
    ll = list(grid.iteritems())
    keys = [x[0] for x in ll]
    values = [x[1] for x in ll]
    indexes = [[i for i in range(len(x))] for x in values]
    indexes_flat = generate_grid_indexes(indexes)
    for arr in indexes_flat:
        m = []
        for level, pos in enumerate(arr):
            key = keys[level]
            value = values[level][pos]
            m.append((key, value))

        res.append(OrderedDict(m))

    return res


def generate_grid_indexes(indexes):
    num = reduce(lambda x, y: x * y, [len(x) for x in indexes])
    print "We're about to create {} indexes".format(num)
    res = []
    sz = len(indexes)
    level = sz - 1
    vector = [0] * sz
    while True:
        pos = vector[level]
        size_at_level = len(indexes[level])
        # print vector, level
        if level == sz - 1:
            if pos < size_at_level:
                res.append(list(vector))
                vector[level] += 1
            else:
                vector[level] = 0
                level -= 1
                vector[level] += 1

        else:
            if pos < size_at_level:
                level += 1
            else:
                if level > 0:
                    vector[level] = 0
                    level -= 1
                    vector[level] += 1
                else:
                    break

    return res


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


def create_folds_list(df, target, folds_num, runs_num, seed):
    folds_list = []
    for run_id in range(runs_num):
        folds = []
        random_state = seed + run_id
        skf = StratifiedKFold(n_splits=folds_num, shuffle=True, random_state=random_state)
        for fold_id, (big_ind, small_ind) in enumerate(skf.split(np.zeros(len(df)), df[target])):
            train = df.iloc[big_ind]
            test = df.iloc[small_ind]
            folds.append((fold_id, train, test))

        folds_list.append((run_id, df, folds))

    return folds_list


def XgbEstWithEarlyStop(train, test, target, params):
    train, test = train.copy(), test.copy()

    train_target = train[target]
    test_target = test[target]
    for col in [target]:
        del train[col], test[col]

    est = xgb.XGBClassifier(**params)
    evals = [(test, test_target)]
    est.fit(train, train_target, eval_set=evals, early_stopping_rounds=100, eval_metric='auc', verbose=False)
    probs = est.predict_proba(test)[:, 1]
    additional_info = {'best_iteration': est.best_iteration}

    return probs, additional_info


def NativeXgbEstWithEarlyStop(train, test, target, params):
    params = params.copy()
    objective = params['objective']

    train, test = train.copy(), test.copy()

    train_target = train[target]
    test_target = test[target]
    for col in [target]:
        del train[col], test[col]

    d_train = xgb.DMatrix(train, train_target)
    d_test = xgb.DMatrix(test, test_target)
    evals = [(d_test,'test')]
    booster = xgb.train(params, d_train, evals=evals, num_boost_round=2000, early_stopping_rounds=100, verbose_eval=False)

    probs = booster.predict(d_test, ntree_limit=booster.best_ntree_limit)
    additional_info = {'best_ntree_limit': booster.best_ntree_limit}

    return probs, additional_info


def Postprocessor(w):
    for k in ['params', 'folds_info']:
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
            close = len(ll)!=0

        if close:
            self.clear()

        return close

    def clear(self):
        with open(self.fp, 'w'):
            pass


def perform_cv_experiments(params_list, folds_list, est, postprocessor, metric, writer, target, exiter):
    counter = 0
    for param_id, params in tqdm(enumerate(params_list)):
        for run_id, df, folds in folds_list:
            t = time()
            print run_id, params
            folds_info = []
            performance = []
            for fold_id, train, test in folds:
                if exiter.must_we_exit():
                    exiter.clear()
                    sys.exit(1)

                probs, additional_info = est(train, test, target, params)
                df.loc[test.index, 'probs'] = probs
                df.loc[test.index, 'fold'] = fold_id
                folds_info.append(additional_info)
                g = metric(test[target], probs)
                performance.append(g)
                print '#{} {}={}'.format(fold_id, metric.__name__, g)
            print 'Avg {}={}'.format(metric.__name__, np.mean(performance))
            global_id = counter
            counter += 1
            w = {
                'indexes': df.index,
                'probs': df['probs'],
                'folds': df['fold'],
                'performance': performance,
                'params': params,
                'folds_info': folds_info,
                'param_id': param_id,
                'run_id': run_id,
                'global_id': global_id
            }
            w = postprocessor(w)
            writer.write(w)
            print 'time={}'.format(time()-t)


def try_it(grid, out_fp):
    seed = 110
    np.random.seed(seed)
    train_df, test_df = load_train_test()
    print 'Columns num {}'.format(len(train_df.columns))
    df = shuffle_df(train_df, seed)
    params_list = walk_through_grid(grid)
    params_list = np.random.permutation(params_list)
    folds_list = create_folds_list(df, target, 5, 2, seed)
    est = NativeXgbEstWithEarlyStop
    postprocessor = Postprocessor
    metric = gini_normalized
    exiter = Exiter('exit.txt')

    with Hdf5Writer(out_fp, 'experiments', Experiment) as writer:
        perform_cv_experiments(params_list, folds_list, est, postprocessor, metric, writer, target, exiter)

    print 'Done!'

grid = {
    'nthread': [-1],
    'n_estimators': [2000],
    'reg_lambda':[0,1,2,5,10, 50],
    'reg_alpha':[0,1,2,5,10],
    'gamma':[0,1,2,5,10],
    'min_child_weight':[0,1,2,5,10]
}

grid_native = {
    'tree_method':['hist'],
    'objective': ['binary:logistic'],
    'eval_metric': ['auc'],
    'max_depth':[3],
    'lambda':[0,1,2,5,10, 50],
    'alpha':[0,1,2,5,10],
    'gamma':[0,1,2,5,10],
    'min_child_weight':[0,1,2,5,10]
}


# test_grid_native = {
#     'tree_method':['hist'],
#     'objective': ['binary:logistic'],
#     'eval_metric': ['auc'],
#     # 'nthread': [-1],
#     'max_depth':[3],
#     'lambda':[0, 10],
#     'alpha':[0,10],
#     'gamma':[1],
#     'min_child_weight':[1]
# }
# out_fp ='blja.hdf5'


out_fp = 'grid_cv_res_v2_seed_110.hdf5'

try_it(grid_native, out_fp)