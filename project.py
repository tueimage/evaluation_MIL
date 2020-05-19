import argparse
import yaml
import pandas as pd

from cnn.generalized.import_dataset import import_dataset
from cnn.generalized.prepare_dataset import prepare_dataset
from cnn.generalized.train_model import train_model
from cnn.generalized.load_model import load_model
from cnn.generalized.predict_patches import predict_patches
from cnn.generalized.evaluate_performance import evaluate_performance


## Add handler for the global configuration file ==============================

# def load_config(path):
#     with open(path, 'r') as ymlfile:
#         return yaml.load(ymlfile)

# parser = argparse.ArgumentParser()
# parser.add_argument('-c', '--config_path', type=str,
#                     help='Provide the file path to the configuration')

# args   = parser.parse_args()
# config = load_config(args.config_path)


## ============================================================================

with open('config_new.yml', 'r') as ymlfile:
    config = yaml.load(ymlfile)

## ============================================================================

dataset = config['dataset']
mode = config['mode']


# # Initialize datasets for training, validation and testing
# df_labels, df_labels_test = import_dataset(config)

# # Generate separate datasets for training, validation and testing
# df_train, df_val, df_test = prepare_dataset(config, df_labels, df_labels_test)

# # Save the databases to a file
# df_train.to_hdf( str(dataset)+'.hdf5', 'df_train', 'w')
# df_val.to_hdf(   str(dataset)+'.hdf5', 'df_val',   'a')
# df_test.to_hdf(  str(dataset)+'.hdf5', 'df_test',  'a')

# del df_labels, df_labels_test


## ============================================================================

if mode == 'train':
    
    # Load the databases
    df_train = pd.read_hdf(str(dataset)+'.hdf5', 'df_train')
    df_val   = pd.read_hdf(str(dataset)+'.hdf5', 'df_val')
    df_test  = pd.read_hdf(str(dataset)+'.hdf5', 'df_test')
    
    train_model(config, df_train, df_val, df_test)

## ============================================================================

if mode == 'test':
    
    # Load the model.
    model = load_model(config)
    
    # predict_patches(config, model, df_test)

    evaluate_performance(config)