# Prepare the dataset.

import pandas as pd
import cnn.preprocessor.load_data as ld
import cnn.preprocessor.load_data_mura as ldm    # preprocess MURA dataset


def prepare_dataset(config, df_labels, df_labels_test=None):
    
    class_name = config['class_name']
    
    # If test is not yet split from train/val (e.g. X-ray dataset)
    if not df_labels_test:
        
        df_class, df_bbox = ld.separate_localization_classification_labels(df_labels, class_name)

        # Split the classification dataset into training and testing
        _, _, \
            df_class_train, df_class_test = ld.split_test_train_v2(
                                                 df_class,
                                                 test_ratio   = 0.2,
                                                 random_state = 1)
            
        # Split the classification training dataset into training and validation
        _, _, \
            df_class_train, df_class_val  = ld.split_test_train_v2(
                                                df_class_train,
                                                test_ratio   = 0.2,
                                                random_state = 1)
    
        # Split the localization dataset into training and testing
        _, _, \
            df_bbox_train, df_bbox_test   = ld.split_test_train_v2(
                                                 df_bbox,
                                                 test_ratio   = 0.2,
                                                 random_state = 1)
        
        df_train  = pd.concat([df_class_train, df_bbox_train])
        df_test   = pd.concat([df_class_test,  df_bbox_test])
        df_val    = df_class_val
        
        col_patches = class_name + '_loc'
    
    
    # If test is already split from train/val (e.g. MURA dataset)
    if df_labels_test:
        # Split dataset into training and validation
        _, _, \
            df_labels, df_labels_val = ldm.split_train_val_set(df_labels)
        
        df_train   = ldm.filter_rows_on_class(df_labels,      class_name)
        df_val     = ldm.filter_rows_on_class(df_labels_val,  class_name)
        df_test    = ldm.filter_rows_on_class(df_labels_test, class_name)
        
        col_patches = 'instance labels'
    
    
    df_train   = ld.keep_index_and_1diagnose_columns(df_train, col_patches)
    df_val     = ld.keep_index_and_1diagnose_columns(df_val,   col_patches)
    df_test    = ld.keep_index_and_1diagnose_columns(df_test,  col_patches)
    
    return df_train, df_val, df_test