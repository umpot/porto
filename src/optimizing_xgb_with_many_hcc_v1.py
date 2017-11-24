from collections import OrderedDict
from loading_v1 import *
from common_kg import *
from commons import *
import numpy as np


def try_it(grid, experiment_name, folds_fp):
    out_fp = '{}.hdf5'.format(experiment_name)
    seed = 110
    np.random.seed(seed)
    folds_batch = load_fold_batch(folds_fp, index_label='id')
    est = NativeXgbEstWithEarlyStop
    exiter = Exiter('exit.txt')
    controller = ParamsOptimizerController(GreedyPerParamParamsGenerator(grid, 1), folds_batch)

    mongo_writer = MongoWriter(user, password, gc_host, db_name, experiment_name, MongoPostprocessor)
    with Hdf5Writer(out_fp, 'experiments', Experiment, Hdf5Postprocessor) as hdf5writer:
        writer = ListWriter([mongo_writer, hdf5writer])
        perform_cv_experiments(controller, est, gini_normalized, writer, target, exiter)

    print 'Done!'


grid = OrderedDict([
    ('tree_method', ['hist']),
    ('objective', ['binary:logistic']),
    ('eval_metric', ['auc']),
    ('max_depth', [3]),
    ('subsample', [0.8]),
    ('eta', [0.1]),
    ('colsample_bytree',[0.8]),
    ('lambda', [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]),
    ('alpha', [0, 5,  10, 20, 30, 40, 50]),
    ('min_child_weight', [0, 1, 2, 5, 10, 20, 50]),
    ('scale_pos_weight',[1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7])

])

name = 'optimizing_xgb_many_hcc_seed110'
folds_fp = '../../../data/porto/many_hcc_seed_110.hdf5'

try_it(grid, name, folds_fp)
