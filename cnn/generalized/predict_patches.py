from cnn.keras_preds import predict_patch_and_save_results

# extra imports for choosing GPU
import os
import tensorflow as tf

IMAGE_SIZE = 512
BATCH_SIZE = 2
BOX_SIZE = 16

def predict_patches(config, model, df):
    
    ## Select a custom GPU (Tensorflow v1) ====================================
    
    if config['gpu']:
        os.environ["CUDA_VISIBLE_DEVICES"] = config['gpu'][0]
        tfconfig = tf.compat.v1.ConfigProto()
        tfconfig.gpu_options.per_process_gpu_memory_fraction = config['gpu'][1]
        tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=tfconfig))
    
    path_pred = config['prediction_results_path']
    mode      = config['mode']
    
    ## ========================================================================
    
    predict_patch_and_save_results(
            model, mode+'_set', df, True,
            BATCH_SIZE, BOX_SIZE, IMAGE_SIZE, path_pred,
            True)