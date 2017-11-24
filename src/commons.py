import os
import json
import sys
from time import time
from collections import OrderedDict

import numpy as np
import pandas as pd
import xgboost as xgb
from pandas.core.common import SettingWithCopyWarning

from pymongo import MongoClient
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from tables import *
from itertools import combinations
import logging
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=SettingWithCopyWarning)

pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 10000)
pd.set_option('display.max_rows', 5000)
pd.set_option('display.max_colwidth', 5000)

#################################################################################
#################################################################################


categoricals = [
    'ps_car_01_cat', #13
    'ps_car_02_cat', #3
    'ps_car_03_cat', #3
    'ps_car_04_cat', #10
    'ps_car_05_cat', #3
    'ps_car_06_cat', #18
    'ps_car_07_cat', #3
    'ps_car_08_cat', #2
    'ps_car_09_cat', #6
    'ps_car_10_cat', #3
    'ps_car_11_cat', #104
    'ps_ind_02_cat', #5
    'ps_ind_04_cat', #3
    'ps_ind_05_cat', #8,

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
    'ps_car_11_cat', #104,
]

additional_categoticals =[
    'ps_car_11', # 5,
    'ps_car_12', # 184,
    'ps_car_14', # 850,
    'ps_car_15', # 15,
    'ps_ind_01', # 8,
    'ps_ind_02_cat', # 5,
    'ps_ind_03', # 12,
    'ps_ind_04_cat', # 3,
    'ps_ind_05_cat', # 8,
    'ps_ind_14', # 5,
    'ps_ind_15', # 14,
    'ps_reg_01', # 10,
    'ps_reg_02', # 19,
    'ps_reg_03', # 5013
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

fp_train = '../../../data/porto/train.csv'
fp_test = '../../../data/porto/test.csv'

target = 'target'

sizes = {'train': 595212, 'test': 892816}

gc_host='35.203.180.79'
local_host = '10.20.0.144'
user='ubik'
password='nfrf[eqyz'#nfrf[eqyz
db_name = 'porto'

def add_combs(train, test, combs):
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

class Experiment(IsDescription):
    param_id = Int32Col()
    run_id = Int32Col()
    global_id = Int32Col()

    avg_performance = Float32Col()
    performance = Float32Col(shape=(5,))

    probs = Float32Col(shape=(sizes['train'],))
    indexes = Int32Col(shape=(sizes['train'],))
    folds = Int32Col(shape=(sizes['train'],))

    params = StringCol(1000)
    folds_info = StringCol(1000)
    controller_info = StringCol(10000)

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

def Hdf5Postprocessor(w):
    w = w.copy()
    for x in w['folds_info']:
        del x['importance']
    for k in ['params', 'folds_info', 'controller_info']:
        w[k] = json.dumps(w[k])

    return w


def MongoPostprocessor(w):
    w = w.copy()
    del w['probs']
    del w['folds']
    del w['indexes']
    for fi in w['folds_info']:
        fi['importance'] = {k.replace('.', ','): v for k, v in fi['importance'].items()}

    w['avg_performance'] = np.mean(w['performance'])

    return w