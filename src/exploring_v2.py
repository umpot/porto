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
    'ps_ind_05_cat', #8
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

    return pd.DataFrame({
        'col': cols,
        'auc': aucs,
        'coverage': coverages,
        'not_zero':not_zero})[['col', 'auc', 'coverage', 'not_zero']].sort_values(by='auc')

def explore_categ_badrate(df, col, target):
    return df.groupby(col)[target].agg(['mean', 'count'])


train_df = pd.read_csv(fp_train, index_col='id')
test_df = pd.read_csv(fp_test, index_col='id')