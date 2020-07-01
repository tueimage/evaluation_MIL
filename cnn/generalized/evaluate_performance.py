import pandas as pd
import numpy as np

import cnn.keras_preds as kp

def save_results(path, *args):
    df = pd.DataFrame()
    for (col_name, col_value) in args:
        df[col_name] = pd.Series(col_value)    
    df.to_csv(path)

def compute_accuracy(acc_localization, has_bbox):
    '''Sum of localization accuracy divided by the total number of bboxes.'''
    
    total_accurate_segmentations = np.sum(acc_localization, axis=0)
    total_segmentations          = np.sum(has_bbox,         axis=0)
    
    with np.errstate(divide='ignore', invalid='ignore'):
        acc_class = total_accurate_segmentations / total_segmentations

    return acc_class

def compute_dice(dice_scores):
    '''Computes Dice'''    
    dice_score_ma = np.ma.masked_array(dice_scores, mask=np.equal(dice_scores, -1))    
    return np.mean(dice_score_ma, axis=0)

def compute_instance_AUC(inst_auc):
    '''Computes instance AUC'''
    return np.mean(inst_auc, axis=0)


def evaluate_performance(config):
    
    # import global variables
    predict_res_path  = config['prediction_results_path']
    path_results      = config['results_path']
    dataset           = config['dataset']
    class_name        = config['class_name']
    dataset_name      = config['model_name']
    pool_method       = config['pooling_operator']
    mode              = config['mode']
    
    # set manual variables
    image_prediction_method = 'as_production'
    r                       = 0.1
    
    
    ## Generate supportive files --------------------------------------------------
    
    # generate the files
    image_labels, image_predictions, \
    has_bbox, accurate_localizations, dice_scores, \
    inst_auc = kp.process_prediction(mode+'_set',
                                     predict_res_path,
                                     pool_method = pool_method,
                                     img_pred_method = image_prediction_method,
                                     r = r,
                                     threshold_binarization = 0.5,
                                     iou_threshold = 0.1)
    
    # save the files
    path = path_results
    np.save(path + '_image_labels',          image_labels)
    np.save(path + '_image_predictions',     image_predictions)
    np.save(path + '_bbox_present',          has_bbox)
    np.save(path + '_accurate_localization', accurate_localizations)
    np.save(path + '_dice',                  dice_scores)
    
    
    ## Compute the performance of the model ---------------------------------------
    
    path = path_results + dataset + '.csv'
    
    if dataset == 'xray':
        accuracy      = compute_accuracy(accurate_localizations, has_bbox)
        dice          = compute_dice(dice_scores)
        instance_AUC  = compute_instance_AUC(inst_auc)
        
        # Save the results
        save_results(path, ['accuracy', accuracy],
                           ['dice',     dice],
                           ['inst_auc', instance_AUC])
        
    kp.compute_save_auc(dataset_name, image_prediction_method, predict_res_path,
                        image_labels, image_predictions, class_name)
