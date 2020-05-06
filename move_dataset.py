import argparse
import yaml
from shutil import copy2

from cnn.generalized.import_dataset import import_dataset


## Add handler for the global configuration file ==============================

def load_config(path):
    with open(path, 'r') as ymlfile:
        return yaml.load(ymlfile)

parser = argparse.ArgumentParser()
parser.add_argument('-c', '--config_path', type=str,
                    help='Provide the file path to the configuration')
parser.add_argument('-n', '--new_path', type=str,
                    help='Provide the file path to the new folder')

args   = parser.parse_args()
config = load_config(args.config_path)

## ============================================================================


def move_dataset(df, path_new):
    for x in df['Dir Path']:
        copy2(x, path_new)


df_labels, df_labels_test = import_dataset(config)
move_dataset(df_labels, args.new_path)
