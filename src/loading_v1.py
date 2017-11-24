from commons import *

combs = [
    ('ps_reg_01', 'ps_car_02_cat'),
    ('ps_reg_01', 'ps_car_04_cat'),
]

comb_cols = ['comb_{}'.format(i) for i in range(len(combs))]

def load_train_test():
    train_df = pd.read_csv(fp_train, index_col='id')
    test_df = pd.read_csv(fp_test, index_col='id')

    add_combs(train_df, test_df, combs)
    train_df = preprocess_df_for_xgb(train_df)
    test_df = preprocess_df_for_xgb(test_df)

    print 'Loaded!'

    return train_df, test_df

def preprocess_df_for_xgb(df):
    df = df.replace(-1, np.nan)
    df['ps_car_13_x_ps_reg_03'] = df['ps_car_13'] * df['ps_reg_03']
    good_cols = oliver_features+[target]+comb_cols+['ps_car_13_x_ps_reg_03']
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