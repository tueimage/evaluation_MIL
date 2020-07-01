# Prepare the dataset.

import numpy as np
import pandas as pd
import cnn.preprocessor.load_data as ld


def create_subset(df_class_train, df_bbox_train, seed, ratio):
    ''''''

    # If there is a separate training bbox (e.g. X-ray dataset)
    if bool(df_bbox_train.size):
        obs_total = len(df_class_train) + len(df_bbox_train)
        # Total observations to be drawn from the classification training set
        obs_to_withdraw = np.ceil(ratio * total_train_obs) - len(df_bbox_train)

    # If there is no separate training bbox (e.g. MURA dataset)
    else:
        obs_to_withdraw = np.ceil(ratio * len(df_class_train))


    if obs_to_withdraw > 0:
        np.random.seed(seed = seed)

        idx_train_set = np.random.choice(df_class_train.index, int(obs_to_withdraw), replace=False)

        train_subset = df_class_train.loc[idx_train_set]
        return train_subset

    else:
        return df_class_train


def prepare_dataset(config, df_labels, df_labels_test=None):
    ''''''

    dataset     = config['dataset']
    class_name  = config['class_name']
    nr_splits   = config['number_of_subsets']
    subset_size = config['subset_size']

    if dataset == 'xray':
        col_patches = class_name + '_loc'
    if dataset == 'pascal':
        col_patches = 'Instance labels'
        groupby = ''
    if dataset == 'mura':
        col_patches = 'instance labels'
        groupby = ''
        df_labels = df_labels[(df_labels['class'] == class_name)]
        df_labels_test = df_labels_test[(df_labels_test['class'] == class_name)]

    # If test is not yet split from train/val (e.g. X-ray dataset)
    # Split the database into classification bags and localization instances.
    if dataset == 'xray':
        df_class, df_bbox = ld.separate_localization_classification_labels(df_labels, class_name)
    else:
        df_class = df_labels
        df_bbox  = pd.DataFrame()

    # Split the classification dataset into training and testing/val
    if not df_labels_test:
        if dataset == 'xray':
            groupby = df_class['Patient ID']
        _, _, \
            df_class_train, df_class_test = ld.split_train_test(
                                                df_class,
                                                nr_splits,
                                                test_ratio   = 0.2,
                                                random_state = 1,
                                                groupby = groupby)
    else:
        [df_class_train, df_class_test] = [[df_labels], [df_labels_test]]


    # Split the classification training/val dataset into training and validation
    col_df_class_train = []
    col_df_class_val   = []

    for df_class_train_split in df_class_train:
        if dataset == 'xray':
            groupby = df_class_train_split['Patient ID']
        _, _, \
            split_class_train, split_class_val = ld.split_train_test(
                                               df_class_train_split,
                                               1,
                                               test_ratio   = 0.2,
                                               random_state = 1,
                                               groupby = groupby)

        col_df_class_train.extend(split_class_train)
        col_df_class_val.extend(split_class_val)

    df_class_train = col_df_class_train
    df_class_val = col_df_class_val


    # Split the localization dataset into training and testing
    if bool(df_bbox.size):
        if dataset == 'xray':
            groupby = df_bbox['Patient ID']
        _, _, \
            df_bbox_train, df_bbox_test   = ld.split_train_test(
                                               df_bbox,
                                               nr_splits,
                                               test_ratio   = 0.2,
                                               random_state = 1,
                                               groupby = groupby)
    else:
        df_bbox_train, df_bbox_test = [[pd.DataFrame()],[pd.DataFrame()]]

    df_val    = df_class_val
    df_train  = []
    df_test   = []

    # Create subsets
    for idx in range(nr_splits):
        df_class_train[idx] = create_subset(df_class_train[idx], df_bbox_train[idx], 1, subset_size)

    # Combine bags and instances to form a single dataset
    for idx in range(nr_splits):
        df_train.append(pd.concat([df_class_train[idx], df_bbox_train[idx]]))
        df_test.append( pd.concat([df_class_test[idx],  df_bbox_test[idx]]))

    # Remove redundant columns
    for idx in range(nr_splits):
        df_train[idx] = ld.keep_index_and_1diagnose_columns(df_train[idx], col_patches)
        df_val[idx]   = ld.keep_index_and_1diagnose_columns(df_val[idx],   col_patches)
        df_test[idx]  = ld.keep_index_and_1diagnose_columns(df_test[idx],  col_patches)

    return df_train, df_val, df_test
