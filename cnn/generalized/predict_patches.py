from cnn.keras_preds import predict_patch_and_save_results

IMAGE_SIZE = 512
BATCH_SIZE = 2
BOX_SIZE = 16

def predict_patches(config, model, df):
    
    path_pred = config['prediction_results_path']
    mode         = config['mode']
    
    predict_patch_and_save_results(
            model, mode+'_set', df, True,
            BATCH_SIZE, BOX_SIZE, IMAGE_SIZE, path_pred,
            True)