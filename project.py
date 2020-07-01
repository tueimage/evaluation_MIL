import argparse
import yaml
import pandas as pd
import os

import random as rn
from numpy.random import seed

from cnn.generalized.verify_input import verify_input
from cnn.generalized.import_dataset import import_dataset
from cnn.generalized.prepare_dataset import prepare_dataset
from cnn.generalized.train_model import train_model
from cnn.generalized.load_model import load_model
from cnn.generalized.predict_patches import predict_patches
from cnn.generalized.evaluate_performance import evaluate_performance


## Add handler for the global configuration file ==============================

def load_config(path):
    if path == None:
        with open('config_new.yml', 'r') as ymlfile:
            return yaml.load(ymlfile)
    else:
        with open(path, 'r') as ymlfile:
            return yaml.load(ymlfile)

parser = argparse.ArgumentParser()
parser.add_argument('-c', '--config_path', type=str,
                    help='Provide the file path to the configuration')

args   = parser.parse_args()
config = load_config(args.config_path)

## ============================================================================

## i. Verify the configuration file
verify_input(config)

## ii. Select a custom GPU (Tensorflow v1)
if config['gpu']:
    os.environ["CUDA_VISIBLE_DEVICES"] = config['gpu'][0]

np.random.seed(1)
rn.seed(1)
seed(1)
os.environ['PYTHONHASHSEED'] = '1'

dataset   = config['dataset']
mode      = config['mode']
path_base = config['project_path']


## ----------------------------------------------------------------------------

if mode == 'prepare':

    # Initialize datasets for training, validation and testing
    df_labels, df_labels_test = import_dataset(config)

    # Generate separate datasets for training, validation and testing
    list_df_train, list_df_val, list_df_test = prepare_dataset(config, df_labels, df_labels_test)

    # Save the databases to a file
    z = 0
    for df_train, df_val, df_test in zip(list_df_train, list_df_val, list_df_test):
        df_train.to_hdf(f'{path_base+dataset}_labels_{z}.hdf5', 'df_train', 'w')
        df_val.to_hdf(  f'{path_base+dataset}_labels_{z}.hdf5', 'df_val',   'a')
        df_test.to_hdf( f'{path_base+dataset}_labels_{z}.hdf5', 'df_test',  'a')
        z += 1

    del df_labels, df_labels_test, z, df_train, df_val, df_test


## ----------------------------------------------------------------------------

# Load the databases
if mode == 'train' or mode == 'test':

    nr_subset = config['subset_number']
    if nr_subset == '':
        nr_subset = 0
        print('\nUsing subset 0')

    df_test  = pd.read_hdf(f'{path_base+dataset}_labels_{nr_subset}.hdf5', 'df_test')

if mode == 'train':
    df_train = pd.read_hdf(f'{path_base+dataset}_labels_{nr_subset}.hdf5', 'df_train')
    df_val   = pd.read_hdf(f'{path_base+dataset}_labels_{nr_subset}.hdf5', 'df_val')

    train_model(config, df_train, df_val, df_test)


## ============================================================================

if mode == 'test':

    # Load the model.
    model = load_model(config)

    predict_patches(config, model, df_test)

    evaluate_performance(config)
