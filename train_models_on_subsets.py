import yaml
import argparse
#import os

from cnn.subsets_training import train_on_subsets


## Select a custom GPU (Tensorflow v1) ----------------------------------------
#os.environ["CUDA_VISIBLE_DEVICES"] = "1"


## Add handler for the global configuration file ==============================

def load_config(path):
    with open(path, 'r') as ymlfile:
        return yaml.load(ymlfile)


parser = argparse.ArgumentParser()
parser.add_argument('-c', '--config_path', type=str,
                    help='Provide the file path to the configuration')

args = parser.parse_args()
config = load_config(args.config_path)


## Train on subsets ===========================================================

train_on_subsets(config)
