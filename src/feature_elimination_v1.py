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
from pymongo import MongoClient
from commons import *
from common_kg import *
from loading_v1 import *


def try_it(params, experiment_name, folds_fp):
    out_fp = os.path.join('../out', '{}.hdf5'.format(experiment_name))
    seed = 110
    np.random.seed(seed)
    folds_batch = load_fold_batch(folds_fp, 'id')
    est = NativeXgbEst
    exiter = Exiter('exit.txt')
    controller = SimpleFeatureEliminationController(folds_batch, params, target, 0.0001)

    mongo_writer = MongoWriter(user, password, gc_host, db_name, experiment_name, MongoPostprocessor)
    with Hdf5Writer(out_fp, 'experiments', Experiment, Hdf5Postprocessor) as hdf5writer:
        writer = ListWriter([mongo_writer, hdf5writer])
        perform_cv_experiments(
            controller,
            est,
            gini_normalized,
            writer,
            target,
            exiter)

    print 'Done!'


params =   {
    "colsample_bytree": 0.8,
    "eval_metric": "auc",
    # "scale_pos_weight": 1,
    # "min_child_weight": 20,
    "subsample": 0.8,
    "eta": 0.1,
    "objective": "binary:logistic",
    # "alpha": 10,
    "num_boost_round": 100,
    "tree_method": "hist",
    "max_depth": 3
    # "lambda": 70
}

name = 'feature_elimination_many_hcc_boost_rounds_100_v1'
folds_fp = '../../../data/porto/many_hcc_5_run_5_seed_110.hdf5'
try_it(params, name, folds_fp)