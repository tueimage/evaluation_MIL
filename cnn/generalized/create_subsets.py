import numpy as np
import pandas as pd
import cnn.preprocessor.load_data as ld
import cnn.preprocessor.load_data_mura as ldm    # preprocess MURA dataset

# def construct_train_test_CV(df_class, df_class_train, df_class_test,
#                             df_bbox, df_bbox_train, df_bbox_test,
#                             val_ratio,
#                             split, random_state, col_patches, train_subset_ratio=None):
    
#     df_class_train, df_class_test = get_rows_from_indices(
#                                         df_class,
#                                         df_class_train[split],
#                                         df_class_test[split])
    
#     df_bbox_train, df_bbox_test   = get_rows_from_indices(
#                                         df_bbox,
#                                         df_bbox_train[split],
#                                         df_bbox_test[split])

#     print("BBOx TRAIN vs ")
#     print(df_bbox_train.shape)
#     print(df_bbox_test.shape)
#     #     TODO: RENAMED DF_CLASS+TRAIN TO OLD_DF_CLASS_TRAIN
#     _, _, \
#         df_class_train, df_class_val = ld.split_test_train_v3(
#                                            df_class_train,
#                                            1,
#                                            test_ratio   = val_ratio,
#                                            random_state = random_state)
        
#     df_train = pd.concat([df_class_train, df_bbox_train])


#     df_val = df_class_val
#     df_test = pd.concat([df_class_test, df_bbox_test])

#     df_train       = ld.keep_index_and_1diagnose_columns(df_train,       col_patches)
#     df_val         = ld.keep_index_and_1diagnose_columns(df_val,         col_patches)
#     df_test        = ld.keep_index_and_1diagnose_columns(df_test,        col_patches)
    
#     df_bbox_train  = ld.keep_index_and_1diagnose_columns(df_bbox_train,  col_patches)
#     df_bbox_test   = ld.keep_index_and_1diagnose_columns(df_bbox_test,   col_patches)
#     df_class_train = ld.keep_index_and_1diagnose_columns(df_class_train, col_patches)
    
#     return df_train, df_val, df_test, df_bbox_train, df_bbox_test, df_class_train



# def get_train_test_CV(df_labels, splits_nr, current_split, random_seed,  class_name, ratio_to_keep=None):
#     '''
#     Returns a train-test separation in cross validation settings, for a specific split #
#     If ratio_to_keep is specified, then (1-ratio) observations from the TRAIN set are dropped
#     :param Y:
#     :param splits_nr: total folds
#     :param split: current fold
#     :param random_state:
#     :param label_col:
#     :param ratio_to_keep: If None
#     :return:
#     '''
    
#     df_class, df_bbox = ld.separate_localization_classification_labels(df_labels, class_name)
    
    
#     df_class_train, df_class_test = ld.split_test_train_v3(
#                                          df_class,
#                                          splits_nr,
#                                          test_ratio   = 0.2,
#                                          random_state = random_seed)
    
    
#     df_bbox_train, df_bbox_test   = ld.split_test_train_v3(
#                                          df_bbox,
#                                          splits_nr,
#                                          test_ratio   = 0.2,
#                                          random_state = random_seed)
    
    
    
    
#     # Here also train test is divided into train and validation
#     train_set, val_set, test_set, df_bbox_train, \
#         df_bbox_test, train_only_class = construct_train_test_CV(
#                                              df_class, df_class_train, df_class_test,
#                                              df_bbox,  df_bbox_train,  df_bbox_test,
#                                              random_state = random_seed,
#                                              diagnose_col = class_name + '_loc',
#                                              split        = current_split,
#                                              val_ratio    = 0.2,
#                                              train_subset_ratio = ratio_to_keep)
        

#     return train_set, val_set, test_set, df_bbox_train, df_bbox_test, train_only_class

# def split_xray_cv(xray_df, cv_splits, split, class_name):
#     df_train, df_val, df_test, \
#     df_bbox_train, df_bbox_test, train_only_class = get_train_test_CV(xray_df, cv_splits, split, random_seed=1,
#                                                                          class_name=class_name, ratio_to_keep=None)

#     print('Training set: ' + str(df_train.shape))
#     print('Validation set: ' + str(df_val.shape))
#     print('Localization testing set: ' + str(df_test.shape))

#     return df_train, df_val, df_test, df_bbox_train, df_bbox_test, train_only_class


def create_subset(df_class_train, df_bbox_train, seed, ratio):
    ''''''
    # Total observations that need to be drawn from the classification training set.
    # If there is a separate training bbox (e.g. X-ray dataset)
    if bool(df_bbox_train.size):
        obs_to_keep = np.ceil(ratio * (len(df_class_train) + len(df_bbox_train))) - len(df_bbox_train)
   
    # If there is no separate training bbox (e.g. MURA dataset)
    else:
        obs_to_keep = np.ceil(ratio * len(df_class_train))


    if obs_to_keep > 0:
        # np.random.seed(seed = seed)
        
        idx_train_set = np.random.choice(df_class_train.index, int(obs_to_keep), replace=False)
        
        train_subset = df_class_train.loc[idx_train_set]
        return train_subset
    
    else:
        return df_class_train
    
    
    

def prepare_subsets(config, df_labels, df_labels_test=None):
    
    class_name  = config['class_name']
    nr_splits   = config['number_of_subsets']
    subset_size = config['subset_size']
    
    # If test is not yet split from train/val (e.g. X-ray dataset)
    if not df_labels_test:
        
        # Split the database into classification bags and localization instances.
        df_class, df_bbox = ld.separate_localization_classification_labels(df_labels, class_name)
        
        col_patches = class_name + '_loc'
        
        # Split the classification dataset into training and testing
        _, _, \
            df_class_train, df_class_test = ld.split_train_test(
                                                df_class,
                                                nr_splits,
                                                test_ratio   = 0.2,
                                                random_state = 1)
        
        # Split the classification training dataset into training and validation
        col_df_class_train = []
        col_df_class_val   = []
        
        for df_class_train_split in df_class_train:
            _, _, \
                split_class_train, split_class_val = ld.split_train_test(
                                                   df_class_train_split,
                                                   1,
                                                   test_ratio   = 0.2,
                                                   random_state = 1)
                
            col_df_class_train.extend(split_class_train)
            col_df_class_val.extend(split_class_val)

        df_class_train = col_df_class_train
        df_class_val = col_df_class_val
    
        # Split the localization dataset into training and testing
        if bool(df_bbox.size):
            _, _, \
                df_bbox_train, df_bbox_test   = ld.split_train_test(
                                                   df_bbox,
                                                   nr_splits,
                                                   test_ratio   = 0.2,
                                                   random_state = 1)        
        else:
            df_bbox_train, df_bbox_test = [[pd.DataFrame()],[pd.DataFrame()]]
        
        df_val    = df_class_val
        df_train = []
        df_test = []
        
        for idx in range(nr_splits):
            print('From ' + str(df_class_train[idx].shape))
            df_class_train[idx] = create_subset(df_class_train[idx], df_bbox_train[idx], 1, subset_size)
            print('to ' + str(df_class_train[idx].shape))
            print('with bbox ' + str(df_bbox_train[idx].shape) + '\n')
        
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
        
    
# IMAGE_SIZE = 512
# BATCH_SIZE = 1
# BATCH_SIZE_TEST = 1
# BOX_SIZE = 16

# overlap_ratio = 0.95
# CV_SPLITS = 5
# number_classifiers = 5

# # this should have the same length as the number of classifiers
# subset_seeds = [1234, 5678, 9012, 3456, 7890]

# for split in range(0, CV_SPLITS):

#         # Split the relevant dataset
#         # TODO: Generalize this part to more datasets
#         if use_xray_dataset:
#             df_train, df_val, df_test, \
#             df_bbox_train, df_bbox_test, \
#             class_train = split_xray_cv(
#                               df_labels, CV_SPLITS, split, class_name)
                
                
#         df_train, df_val, df_test
                
                
#         # elif use_pascal_dataset:
#         #     df_train, df_val, df_test = construct_train_test_cv(pascal_df, CV_SPLITS, split)

#         # else:
#         #     df_train, df_val = split_data_cv(df_train_val, CV_SPLITS, split, random_seed=1, diagnose_col=class_name,
#         #                                      ratio_to_keep=None)
#         #     df_test = filter_rows_and_columns(test_df_all_classes, class_name)


#         # For every classifier
#         for curr_classifier in range(0, number_classifiers):
#             if split == 1:
                
#                 # Initialize the relevant data subset
#                 if use_xray_dataset:
                    
#                     # Form a classification training SUBSET
#                     df_class_train = create_subset(
#                                          class_train, df_bbox_train,
#                                          seed = subset_seeds[curr_classifier],
#                                          ratio = overlap_ratio)
                    
#                     print("Classification training subset:" + str(df_class_train.shape))
                    
                    
#                     df_train        = pd.concat([df_class_train, df_bbox_train])
                    
#                     print(df_bbox_train.shape)
#                     print(df_class_train.shape)
                    
#                 else:
                    
#                     df_train_subset = create_subset(
#                                           df_train, None,
#                                           seed  = subset_seeds[curr_classifier],
#                                           ratio = overlap_ratio)