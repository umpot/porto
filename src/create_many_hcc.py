from tqdm import tqdm

from common_kg import *
from commons import *
from loading_v1 import *
from itertools import combinations

class Preprocessor():
    def __init__(self, to_comb):
        self.to_comb = to_comb

    def preprocess(self, train, test):
        categoticals = hi_categoricals + comb_cols + lo_categoricals + additional_categoticals
        for col in tqdm(categoticals):
            if col not in train.columns:
                log('Skipped {}', col)
                continue
            log(col)
            train, test, tmp = hcc_encode(train, test, col, target)

        for features in tqdm(self.to_comb):
            col = add_interaction_col(train, features)
            col = add_interaction_col(test, features)
            train, test, tmp = hcc_encode(train, test, col, target)
            del train[col], test[col]

        return train, test


to_comb_list = ['ps_car_11_cat',
           'ps_car_01_cat',
           'ps_car_06_cat',
           'ps_car_04_cat',
           'ps_reg_03',
           'ps_car_14']

def get_to_comb():
    res = []
    for i in range(2, 6):
        res+=list(combinations(to_comb_list, i))

    return res

def save_folds(fp, seed):
    train_df, test_df = load_train_test()
    df = train_df
    to_comb = get_to_comb()
    p = Preprocessor(to_comb)
    print p.to_comb
    folds_batch = create_folds_batch(df, target, p.preprocess, 5, 5, seed, fp, return_res=False)


fp = '../../../data/porto/many_hcc_5_run_5_seed_110.hdf5'
seed = 110
save_folds(fp, seed)