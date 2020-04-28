# Computes the performance of the model.

import argparse
import os
import tensorflow as tf         # for machine learning
import yaml                     # to import global parameters
from cnn import keras_preds   


## Select a custom GPU (Tensorflow v1) ----------------------------------------

# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
# sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))


## Definitions ----------------------------------------------------------------

def load_config(path):
    '''Load the global configuration file.'''
    with open(path, 'r') as ymlfile:
        return yaml.load(ymlfile)

parser = argparse.ArgumentParser()
parser.add_argument('-c', '--config_path', type=str,
                    help='Provide the file path to the configuration')
args   = parser.parse_args()
config = load_config(args.config_path)

# import global variables
predict_res_path  = config['prediction_results_path']
use_xray          = config['use_xray_dataset']
class_name        = config['class_name']

# set manual variables
image_prediction_method = 'as_production'
dataset_name = 'subset_test_set_CV0_4_0.95'
pool_method  = 'nor'            # lse / nor / mean
r            = 0.1


## Generate supportive files --------------------------------------------------

# generate the files
image_labels, image_predictions, \
has_bbox, accurate_localizations, dice_scores, \
inst_auc = keras_preds.process_prediction(dataset_name,
                                          predict_res_path,
                                          r=r,
                                          pool_method=pool_method,
                                          img_pred_method=image_prediction_method,
                                          threshold_binarization=0.5,
                                          iou_threshold=0.1)

# save the files
keras_preds.save_generated_files(predict_res_path, dataset_name, image_labels, image_predictions,
                                 has_bbox, accurate_localizations, dice_scores)


## Compute the performance of the model ---------------------------------------

if use_xray:
    keras_preds.compute_save_accuracy_results(dataset_name, predict_res_path, has_bbox, accurate_localizations)
    keras_preds.compute_save_dice_results(dataset_name, predict_res_path, has_bbox, dice_scores)
    keras_preds.compute_save_inst_auc_results(dataset_name, predict_res_path, inst_auc)
    keras_preds.compute_save_auc(dataset_name, image_prediction_method, predict_res_path,
                                 image_labels, image_predictions, class_name)
else:
    keras_preds.compute_save_auc(dataset_name, image_prediction_method, predict_res_path,
                                 image_labels, image_predictions, class_name)
