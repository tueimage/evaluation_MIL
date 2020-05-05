import argparse
import os
import yaml

from cnn.preprocessor import import_dataset, prepare_dataset


## Select a custom GPU (Tensorflow v1) ----------------------------------------
#os.environ["CUDA_VISIBLE_DEVICES"] = "2"


## Add handler for the global configuration file ==============================

def load_config(path):
    with open(path, 'r') as ymlfile:
        return yaml.load(ymlfile)

parser = argparse.ArgumentParser()
parser.add_argument('-c', '--config_path', type=str,
                    help='Provide the file path to the configuration')

args   = parser.parse_args()
config = load_config(args.config_path)


## Initialize datasets for training, validation and testing ===================

print()

# TODO: Add validation step?

df_labels, df_labels_test = import_dataset(config)

df_train, df_val, df_test = prepare_dataset(config, df_labels, df_labels_test)

del df_labels, df_labels_test