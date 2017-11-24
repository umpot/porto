import os
import json
import sys
from time import time
from collections import OrderedDict

import numpy as np
import pandas as pd
import xgboost as xgb

from pymongo import MongoClient
from sklearn.model_selection import StratifiedKFold
from tables import open_file
from itertools import combinations
import logging

logging.basicConfig(filename='log.log', level=logging.DEBUG, format='%(message)s')


def log(msg, *args):
    if len(args) > 0:
        msg = msg.format(*args)
    print msg
    logging.info(msg)

def shuffle_df(df, random_state):
    np.random.seed(random_state)
    return df.iloc[np.random.permutation(len(df))]

def hcc_encode(train_df, test_df, variable, binary_target, k=5, f=1, g=1, r_k=0, folds=5):
    for df in [train_df, test_df]:
        df.fillna(-1, inplace=True)
    """
    See "A Preprocessing Scheme    for High-Cardinality Categorical Attributes in
    Classification and Prediction Problems" by Daniele Micci-Barreca
    """
    prior_prob = train_df[binary_target].mean()
    hcc_name = "_".join(["hcc", variable])

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

    for df in [train_df, test_df]:
        df.replace(-1, np.nan, inplace=True)

    return train_df, test_df, hcc_name


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
    log("We're about to create {} indexes", num)
    res = []
    sz = len(indexes)
    level = sz - 1
    vector = [0] * sz
    while True:
        pos = vector[level]
        size_at_level = len(indexes[level])
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


class MongoWriter():
    def __init__(self, user, password, host, db_name, collection_name, postprocessor):
        self.client = MongoClient(host, 27017)
        self.client['admin'].authenticate(user, password)
        self.db = self.client[db_name]
        self.collection = self.db[collection_name]
        self.postprocessor = postprocessor

    def write(self, m):
        m = self.postprocessor(m)
        self.collection.insert_one(m)


class Hdf5Writer():
    def __init__(self, out_fp, table_name, row_class, postprocessor):
        self.out_fp = out_fp
        self.table_name = table_name
        self.row_class = row_class
        self.postprocessor = postprocessor

    def __enter__(self):
        self.file = open_file(self.out_fp, 'w')
        self.group = self.file.create_group("/", 'experiments', '')
        self.table = self.file.create_table(self.group, self.table_name, self.row_class, "")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.file.close()

    def write(self, m):
        exp = self.table.row
        m = self.postprocessor(m)
        for k, v in m.items():
            exp[k] = v
        exp.append()
        self.table.flush()


class ListWriter():
    def __init__(self, ll):
        self.ll = ll

    def write(self, m):
        for l in self.ll:
            l.write(m)


def create_folds_batch(df, target, preprocessor, folds_num, runs_num, seed, dump_fp=None, return_res=True):
    folds_batch = []
    for run_id in range(runs_num):
        print run_id
        folds = []
        random_state = seed + run_id
        skf = StratifiedKFold(n_splits=folds_num, shuffle=True, random_state=random_state)
        for fold_id, (big_ind, small_ind) in enumerate(skf.split(np.zeros(len(df)), df[target])):
            train = df.iloc[big_ind]
            test = df.iloc[small_ind]
            train, test = preprocessor(train, test)
            if return_res:
                folds.append((fold_id, train, test))
            if dump_fp is not None:
                __dump_one_cv_chunk__(train, test, run_id, fold_id, dump_fp)
        if return_res:
            folds_batch.append((run_id, df, folds))

    if return_res:
        return folds_batch

def load_fold_batch(fp, index_label=None):
    tmp_index_label = 'tmp_index_label'
    if index_label is None:
        index_label = tmp_index_label
    tmp_train_test_label = 'tmp_train_test_label'
    tmp_run_id_label = 'tmp_run_id_label'
    tmp_fold_id_label = 'tmp_fold_id_label'

    bl = pd.read_hdf(fp, key='folds')

    run_ids = list(bl[tmp_run_id_label].unique())
    run_ids.sort()

    fold_ids = list(bl[tmp_fold_id_label].unique())
    fold_ids.sort()

    res = []
    for run_id in run_ids:
        res_local = []
        df = bl[bl[tmp_run_id_label]==run_id]
        for fold_id in fold_ids:
            test = df[(df[tmp_fold_id_label]==fold_id)&(df[tmp_train_test_label]=='test')]
            train = df[(df[tmp_fold_id_label]==fold_id)&(df[tmp_train_test_label]=='train')]

            del test[tmp_fold_id_label]
            del train[tmp_fold_id_label]

            train.index = train[tmp_index_label]
            test.index = test[tmp_index_label]

            del test[tmp_index_label]
            del train[tmp_index_label]

            del test[tmp_train_test_label]
            del train[tmp_train_test_label]

            res_local.append((fold_id, train, test))

            train.index.name = index_label
            test.index.name = index_label

        df = df[df[tmp_fold_id_label]==fold_id][[tmp_run_id_label]]
        df.index.name = index_label
        res.append((run_id, df, res_local))

    return res


def dump_folds_batch(folds_batch, fp):
    cols = None
    for run_id, df, folds in folds_batch:
        for fold_id, train, test in folds:
            if cols is None:
                cols = set(train.columns)

            assert set(train.columns)==set(test.columns)
            assert set(train.columns) == cols

            __dump_one_cv_chunk__(train, test, run_id, fold_id, fp)


def __dump_one_cv_chunk__(train, test, run_id, fold_id, fp):
    tmp_index_label = 'tmp_index_label'
    tmp_train_test_label = 'tmp_train_test_label'
    tmp_run_id_label = 'tmp_run_id_label'
    tmp_fold_id_label = 'tmp_fold_id_label'

    train[tmp_train_test_label] = 'train'
    test[tmp_train_test_label] = 'test'
    train[tmp_run_id_label] = run_id
    test[tmp_run_id_label] = run_id
    train[tmp_fold_id_label] = fold_id
    test[tmp_fold_id_label] = fold_id
    train[tmp_index_label] = train.index
    test[tmp_index_label] = test.index
    bl = pd.concat([train, test])
    bl.to_hdf(fp, 'folds', mode='a', format='table', append=True)


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

    train, test = train.copy(), test.copy()

    train_target = train[target]
    test_target = test[target]
    for col in [target]:
        del train[col], test[col]

    d_train = xgb.DMatrix(train, train_target)
    d_test = xgb.DMatrix(test, test_target)
    evals = [(d_test, 'test')]
    booster = xgb.train(params, d_train, evals=evals, num_boost_round=2000, early_stopping_rounds=100,
                        verbose_eval=False)

    probs = booster.predict(d_test, ntree_limit=booster.best_ntree_limit)
    importance = booster.get_fscore()
    additional_info = {'best_ntree_limit': booster.best_ntree_limit, 'importance': importance}

    return probs, additional_info


def NativeXgbEst(train, test, target, params):
    params = params.copy()
    num_boost_round = params['num_boost_round']
    del params['num_boost_round']

    train, test = train.copy(), test.copy()

    train_target, test_target = train[target], test[target]

    del train[target], test[target]

    d_train = xgb.DMatrix(train, train_target)
    d_test = xgb.DMatrix(test, test_target)
    booster = xgb.train(params, d_train, num_boost_round=num_boost_round)

    probs = booster.predict(d_test)

    importance = booster.get_fscore()
    additional_info = {'best_ntree_limit': booster.best_ntree_limit, 'importance': importance}

    return probs, additional_info


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


class FromListParamsGenerator():
    def __init__(self, params_list):
        self.params_list = params_list
        self.counter = -1

    def generate(self, previous_performances):
        self.counter += 1
        if self.counter < len(self.params_list):
            return self.counter, self.params_list[self.counter]
        else:
            return None, None


class GreedyPerParamParamsGenerator():
    def __init__(self, grid, repetitions):
        self.grid = grid
        self.repetitions = repetitions
        self.pos_x = -1
        self.pos_y = 0
        self.scores = []
        self.results = []
        self.counter = -1
        self.done_cycles = -1
        self.previous_params = None
        self.cache = {}
        self.init_index_to_p_name()

    def init_index_to_p_name(self):
        self.index_to_p_name = {}
        for i, (name, v) in enumerate(self.grid.items()):
            self.index_to_p_name[i] = name

    def start_new_cycle(self):
        self.done_cycles += 1
        self.current_params = {}
        for k, v in self.grid.items():
            self.current_params[k] = np.random.choice(v)

    def start_new_y(self):
        param_name = self.index_to_p_name[self.pos_y]
        self.current_params[param_name] = self.grid[param_name][0]

    def start_new_x(self):
        param_name = self.index_to_p_name[self.pos_y]
        self.current_params[param_name] = self.grid[param_name][self.pos_x]

    def should_start_new_y(self):
        param_name = self.index_to_p_name[self.pos_y]
        vals = self.grid[param_name]
        return self.pos_x >= len(vals)

    def select_best_param(self):
        index = np.argmax(self.scores)
        name = self.index_to_p_name[self.pos_y]
        self.current_params[name] = self.grid[name][index]

    def dict_to_key(self, d):
        keys = list(d.keys())
        keys.sort()
        ll = []
        for k in keys:
            ll += [k, d[k]]

        return tuple(ll)

    def add_cache_item(self, previous_performances):
        key = self.dict_to_key(self.previous_params)
        self.cache[key] = previous_performances

    def generate_inner(self, previous_performances):
        self.pos_x += 1
        if previous_performances is None:
            self.start_new_cycle()
            self.start_new_y()
            return self.current_params

        self.scores.append(np.mean(previous_performances))
        should_start_new_y = self.should_start_new_y()
        if should_start_new_y:
            self.select_best_param()
            self.scores = []
            self.pos_x = 0
            self.pos_y += 1
            if self.pos_y >= len(self.grid):
                self.pos_y = 0
                self.results.append(self.current_params)
                self.start_new_cycle()
                self.start_new_y()
                if self.done_cycles >= self.repetitions:
                    return None
                else:
                    return self.current_params
            else:
                self.start_new_y()
                return self.current_params

        else:
            self.start_new_x()
            return self.current_params

    def generate(self, previous_performances):
        self.counter += 1
        while True:
            if previous_performances is not None:
                self.add_cache_item(previous_performances)
            next_params = self.generate_inner(previous_performances)
            if next_params is None:
                return None

            self.previous_params = next_params
            key = self.dict_to_key(next_params)
            if key in self.cache:
                previous_performances = self.cache[key]
                log('From cache {} {}', key, previous_performances)
                continue
            else:
                return next_params


class ParamsOptimizerController():
    def __init__(self, optimizer, folds_batch):
        self.optimizer = optimizer
        self.folds_batch = folds_batch

    def run(self, previous_performances, previous_run_info):
        return self.folds_batch, self.optimizer.generate(previous_performances), {}

class RunSingleParamController():
    def __init__(self, params, folds_batch):
        self.params = params
        self.folds_batch = folds_batch

    def run(self, previous_performances, previous_run_info):
        return self.folds_batch, self.params, {}


class RandomGridSearchController():
    def __init__(self, grid, folds_batch, shuffle_grid=True):
        self.grid = grid
        self.folds_batch = folds_batch
        if shuffle_grid:
            self.params_list = np.random.permutation(walk_through_grid(grid))
        else:
            self.params_list = walk_through_grid(grid)
        self.counter = -1

    def run(self, previous_performances, previous_run_info):
        self.counter += 1
        if self.counter < len(self.params_list):
            return self.folds_batch, self.params_list[self.counter], {}
        else:
            return None, None, None


class SimpleFeatureEliminationController():
    def __init__(self, folds_batch, params, target, threshold):
        self.folds_batch = folds_batch
        self.params = params
        self.threshold = threshold
        self.target = target
        self.cols = self.init_cols()
        self.baseline = None
        self.victim = None

    def get_next_victim(self):
        i = self.cols.index(self.victim)
        i = (i + 1) % (len(self.cols))
        return self.cols[i]

    def init_cols(self):
        cols = list(self.folds_batch[0][2][0][1].columns)
        cols.remove(self.target)
        cols.sort()
        return cols

    def get_next_folds_batch(self, cols):
        cols = cols + [self.target]
        res = []
        for run_id, df, folds in self.folds_batch:
            res_local = []
            for fold_id, train, test in folds:
                res_local.append((fold_id, train[cols], test[cols]))
            res.append((run_id, df, res_local))

        return res

    def run(self, previous_performances, previous_run_info):
        if previous_performances is None:
            cols = list(self.cols)
            return self.get_next_folds_batch(cols), self.params, {'columns': cols}

        performance = np.mean(previous_performances)

        if self.baseline is None:
            self.baseline = performance
            next_victim = self.cols[0]
            self.victim = next_victim
            cols = list(self.cols)
            cols.remove(next_victim)
            folds_batch = self.get_next_folds_batch(cols)
            return folds_batch, self.params, {'columns': cols}

        next_victim = self.get_next_victim()

        log('Comparing prev={} VS current={}', self.baseline, performance)
        if self.baseline - performance <= self.threshold:
            self.baseline = performance
            log('Removing column {}', self.victim)
            self.cols.remove(self.victim)

        self.victim = next_victim
        cols = list(self.cols)
        cols.remove(next_victim)
        folds_batch = self.get_next_folds_batch(cols)
        return folds_batch, self.params, {'columns': cols}


def perform_cv_experiments(controller,
                           est,
                           metric,
                           writer,
                           target,
                           exiter):
    counter = 0
    previous_performances = None
    previous_run_info = None
    while True:
        folds_batch, params, controller_info = controller.run(previous_performances, previous_run_info)
        if params is None:
            break
        previous_performances = []
        previous_run_info = []
        for run_id, df, folds in folds_batch:
            t = time()
            log((run_id, params))
            folds_info = []
            performance = []
            for fold_id, train, test in folds:
                if exiter.must_we_exit():
                    exiter.clear()
                    sys.exit(1)
                probs, info = est(train, test, target, params)
                df.loc[test.index, 'probs'] = probs
                df.loc[test.index, 'fold'] = fold_id
                folds_info.append(info)
                g = metric(test[target], probs)
                performance.append(g)
                log('#{} {}={}', fold_id, metric.__name__, g)
            log('Avg {}={}', metric.__name__, np.mean(performance))
            previous_performances.append(performance)
            previous_run_info.append(folds_info)
            global_id = counter
            counter += 1
            w = {
                'indexes': df.index,
                'probs': df['probs'],
                'folds': df['fold'],
                'performance': performance,
                'params': params,
                'folds_info': folds_info,
                'run_id': run_id,
                'global_id': global_id,
                'controller_info': controller_info
            }
            writer.write(w)
            log('time={}', time() - t)


def test_GreadyPerParamParamsGenerator():
    np.random.seed(114)
    grid = OrderedDict([('a', [3, 4, 5, 6]), ('b', [1, 7, 8]), ('c', [0.2, 0.5, 0.7, 0.8, 0.9])])
    gen = GreedyPerParamParamsGenerator(grid, 3)

    print gen.generate(None)
    print gen.generate([0.1])
    print gen.generate([0.2])
    print gen.generate([0.6])
    print gen.generate([0.5])

    print gen.generate([0.6])
    print gen.generate([0.4])
    # print gen.generate([0.5])

    print gen.generate([0.3])
    print gen.generate([0.9])
    # print gen.generate([0.8])
    print gen.generate([0.7])
    print gen.generate([0.6])

    assert len(gen.results) == 1
    assert gen.results[0] == {'a': 5, 'c': 0.5, 'b': 1}

    print gen.generate([0.1])
    print gen.generate([0.7])
    # print gen.generate([0.2])
    print gen.generate([0.5])

    # print gen.generate([0.5])
    print gen.generate([0.6])
    # print gen.generate([0.4])

    print gen.generate([0.3])
    print gen.generate([0.7])
    # print gen.generate([0.9])
    print gen.generate([0.6])
    print gen.generate([0.8])

    assert len(gen.results) == 2
    assert gen.results[1] == {'a': 4, 'c': 0.9, 'b': 1}

    print gen.generate([0.1])
    print gen.generate([0.2])
    print gen.generate([0.5])
    print gen.generate([0.6])

    print gen.generate([0.5])
    # print gen.generate([0.4])
    print gen.generate([0.6])

    print gen.generate([0.3])
    print gen.generate([0.7])
    print gen.generate([0.6])
    # print gen.generate([0.8])
    last_id, last = gen.generate([0.9])

    assert last_id is None
    assert last is None
    assert len(gen.results) == 3
    assert gen.results[2] == {'a': 6, 'c': 0.9, 'b': 7}


def add_interaction_col(df, features):
    col_name = 'int_' + '**'.join(features)
    df[col_name] = ''
    for f in features:
        df[col_name] += "*"
        df[col_name] += df[f].apply(str)

    # log((col_name, len(df[col_name].unique())))

    return col_name
