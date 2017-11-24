from collections import OrderedDict
from loading_v1 import *
from common_kg import *
from commons import *
import numpy as np


def try_it(params, experiment_name, folds_fp):
    out_fp = '{}.hdf5'.format(experiment_name)
    seed = 110
    np.random.seed(seed)
    folds_batch = load_fold_batch(folds_fp, index_label='id')
    est = NativeXgbEstWithEarlyStop
    exiter = Exiter('exit.txt')
    controller = RunSingleParamController(params, folds_batch)

    mongo_writer = MongoWriter(user, password, gc_host, db_name, experiment_name, MongoPostprocessor)
    with Hdf5Writer(out_fp, 'experiments', Experiment, Hdf5Postprocessor) as hdf5writer:
        writer = ListWriter([mongo_writer, hdf5writer])
        perform_cv_experiments(controller, est, gini_normalized, writer, target, exiter)

    print 'Done!'


grid = OrderedDict([
    ('tree_method', 'hist'),
    ('objective', 'binary:logistic'),
    ('eval_metric', 'auc'),
    ('max_depth', 3),
    ('subsample', 0.8),
    ('eta', 0.1),
    ('colsample_bytree',0.8),

])

name = 'trash_21_11_2017'
folds_fp = '../../../data/porto/many_hcc_seed_110.hdf5'

try_it(grid, name, folds_fp)
