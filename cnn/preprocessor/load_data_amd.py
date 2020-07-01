from pathlib import Path
import os

from sklearn.model_selection import StratifiedShuffleSplit

import cnn.preprocessor.load_data as ld    # preprocess X-Ray dataset
import cnn.preprocessor.load_data_mura as ldm    # preprocess MURA dataset
import pandas as pd





def get_train_test_1fold(df):
    train_inds, test_inds = next(StratifiedShuffleSplit(
                                n_splits     = 1,
                                test_size    = 0.2,
                                random_state = 0).split(df, df['Label']))
    
    assert train_inds.all() != test_inds.all(), "overlapp occur"
    
    return df.iloc[train_inds], df.iloc[test_inds]


def split_train_val_test(df):
    train_val, test = get_train_test_1fold(df)
    train, val = get_train_test_1fold(train_val)

    df_train = ld.keep_index_and_1diagnose_columns(train, 'Instance labels')
    df_test  = ld.keep_index_and_1diagnose_columns(test, 'Instance labels')
    df_val   = ld.keep_index_and_1diagnose_columns(val, 'Instance labels')
    return df_train, df_val, df_test


def split_data_cv(df, nr_cv):
    sss = StratifiedShuffleSplit(n_splits=nr_cv, test_size=0.2, random_state=0)
    train_ind_col = []
    test_ind_col = []
    for train_inds, test_inds in sss.split(df, df['Label']):
        assert train_inds.all() != test_inds.all(), "overlapp occur"
        train_ind_col.append(train_inds)
        test_ind_col.append(test_inds)
    return train_ind_col, test_ind_col


def construct_train_test_cv(df, nr_cv, split):
    train_val_ind_col, test_ind_col = split_data_cv(df, nr_cv)
    df_train_val, df_test = ld.get_rows_from_indices(df, train_val_ind_col[split], test_ind_col[split])
    train_ind_col, val_ind_col = split_data_cv(df_train_val, nr_cv)
    df_train, df_val = ld.get_rows_from_indices(df_train_val, train_ind_col[split], val_ind_col[split])

    train_set = ld.keep_index_and_1diagnose_columns(df_train, 'Instance labels')
    val_set   = ld.keep_index_and_1diagnose_columns(df_val, 'Instance labels')
    test_set  = ld.keep_index_and_1diagnose_columns(df_test, 'Instance labels')

    return train_set, val_set, test_set