from cnn.keras_preds import predict_patch_and_save_results

# extra imports for choosing GPU
import os
import tensorflow as tf

IMAGE_SIZE = 512
BATCH_SIZE = 2
BOX_SIZE = 16

def predict_patches(config, model, df):
    
    predict_patch_and_save_results(
            model, mode+'_set', df, True,
            BATCH_SIZE, BOX_SIZE, IMAGE_SIZE, path_pred,
            True)