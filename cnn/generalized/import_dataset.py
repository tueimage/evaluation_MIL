# Load or create an image database.

import pandas as pd                        # to import .csv files as a database
import cnn.preprocessor.load_data as ld    # preprocess X-Ray dataset
import cnn.preprocessor.load_data_mura as ldm    # preprocess MURA dataset

PATCH_SIZE = 16


def import_dataset(config):
    '''
    Load an already prepared database, or create and prepare an image database.
    Input: Global configuration file.
    Output: df_labels, df_labels_test 
    '''
    
    dataset           = config['dataset']
    skip_processing   = config['skip_processing_labels']
    path_image        = config['image_path']
    path_image_test   = config['test_image_path']
    path_labels       = config['labels_path']
    path_labels_test  = config['test_labels_path']
    path_labels_loc   = config['localization_labels_path']
    path_results      = config['results_path']
    class_name        = config['class_name']
    
    
    # No processing: load already processed database --------------------------
    
    if skip_processing:
        print('Loading data ...')
        
        df_labels = pd.read_csv(path_labels).dropna(axis=1)
        df_labels_test = None
        
        if path_labels_test:
            df_labels_test = pd.read_csv(path_labels_test).dropna(axis=1)
        
        return df_labels, df_labels_test
    
    
    # Processing: create a new database ---------------------------------------
    
    else:
        print('Processing database ...')
        
        if dataset == 'xray':
            
            # Categorize the database entry labels
            df_labels_class = ld.get_classification_labels(path_labels, False)
            
            ## TODO: Only when input length != database length
            # Match the database to the dataset images.
            print('Match the database to the dataset images.')
            df_labels_adj = ld.preprocess_labels(df_labels_class, path_image)
            
            # Filter patients with >= 1 postitive image for the class category
            print('Filter patients with >= 1 postitive image for the class category')
            df_labels_pos = ld.keep_observations_of_positive_patients(df_labels_adj, path_results, class_name)
            
            # Add location annotations
            print('Add location annotations')
            df_labels = ld.couple_location_labels(path_labels_loc, df_labels_pos, PATCH_SIZE, path_results)
            
            return df_labels, None
        
            
        if dataset == 'mura':
                
            end_class = path_image.find('MURA-v1.1')            
            mura_folder_root = path_image[0:end_class]
            
            df_labels = ldm.get_save_processed_df(path_labels, path_image, mura_folder_root, "train_mura")
            
            df_labels_test = ldm.get_save_processed_df(path_labels_test, path_image_test, mura_folder_root,
                                                        "test_mura")

            return df_labels, df_labels_test