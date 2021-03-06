import numpy as np
from pathlib import Path
from keras_preprocessing.image import load_img, img_to_array
from sklearn.metrics import roc_auc_score, average_precision_score

from cnn.keras_utils import calculate_scale_ratio, image_larger_input
from cnn.preprocessor.load_data_mura import pad_image, padding_needed
from stability.preprocessor.preprocessing import load_predictions_v2, indices_segmentation_images_v2, \
    filter_predictions_files_on_indeces, indices_positive_images
from stability.stability_2classifiers.stability_2classifiers import compute_correlation_scores_v2, \
    get_binary_scores_forthreshold_v2
from stability.utils import get_image_index, save_additional_kappa_scores_forthreshold, save_mean_stability, \
    save_mean_stability_auc, save_mean_dice
from stability.visualizations.visualization_utils import visualize_single_image_1class_5classifiers, \
    visualize_correlation_heatmap, combine_correlation_heatmaps_next_to_each_other, \
    make_scatterplot_with_errorbar, make_scatterplot_with_errorbar_v2, \
    visualize_5_classifiers_mura, visualize_5_classifiers


def get_matrix_total_nans_stability_score(stab_index_collection, total_images_collection, normalize):
    nan_matrix = np.count_nonzero(np.isnan(np.array(stab_index_collection).
                                           reshape(5, 5, len(total_images_collection[0]))), axis=-1)
    if normalize:
        return nan_matrix / len(total_images_collection[0])
    else:
        return nan_matrix


def compute_auc_classifiers(all_bag_predictions, all_bag_labels):
    auc_scores = []
    for ind in range(0, len(all_bag_predictions)):
        auc = roc_auc_score(all_bag_labels[ind], all_bag_predictions[ind])
        auc_scores.append(auc)
    print("all auc scores")
    print(auc_scores)
    return auc_scores


def visualize_bag_vs_stability(bag_auc_list, stability, y_axis_el1, y_axis_el2, x_axis_title, res_path):
    means = []
    stdevs = []
    stabilities = []
    for ind in range(0, len(bag_auc_list)):
        for ind2 in range(0, len(bag_auc_list)):
            if ind != ind2:
                observations = [bag_auc_list[ind], bag_auc_list[ind2]]
                mean_point = np.mean(observations, axis=0)
                stand_point = np.std(observations, axis=0)
                means.append(mean_point)
                stdevs.append(stand_point)
                stabilities.append(stability[ind, ind2])
    # visualize_scatter_bag_auc_stability(bag_auc_list, stability, y_axis_el1, y_axis_el2)
    make_scatterplot_with_errorbar(means, 'mean_' + y_axis_el1 + '_' + y_axis_el2 + '_error',
                                   stabilities, x_axis_title,
                                   res_path, fitting_curve=False, y_errors=stdevs, error_bar=True,
                                   bin_threshold_prefix=None)


def compute_and_visualize_average_instance_stability(reshaped_stability, res_path, identifier, index_name,
                                                     xy_axis_title):
    average_index_all_images = np.average(reshaped_stability, axis=-1)
    visualize_correlation_heatmap(average_index_all_images, res_path,
                                  '_avg_inst_' + str(index_name) + '_' + identifier,
                                  xy_axis_title, dropDuplicates=True)


def ensure_file_contain_same(files, file_nr):
    for nr in range(0, file_nr):
        assert (np.asarray(files[0]) == np.asarray(files[nr])).all(), "files provided are not the same"


def get_value_unique_combinations(stability_coll):
    pairwise_stability_all_images = []
    all_stab_scores = []
    classifiers_stab_score = []
    for classifier_ind in range(0, 5):
        stab_scores = stability_coll[classifier_ind, classifier_ind + 1:, :]
        all_stab_scores = np.concatenate((all_stab_scores, stab_scores.reshape(-1)), axis=0)
        stab_score_per_classifier = np.delete(stability_coll[classifier_ind], classifier_ind, axis=0)
        classifiers_stab_score.append(stab_score_per_classifier.reshape(-1))
    return all_stab_scores, classifiers_stab_score


def compute_intersection_union_patches(binary_patch_predictions, patch_labels, threshold_binarization):
    correct_prediction = np.equal(binary_patch_predictions, patch_labels, dtype=float)
    # check only active patches from the labels and see if the prediction there agrees with the labels
    intersection = np.where(np.greater(patch_labels, 0), correct_prediction, 0)
    total_patch_intersection = np.sum(intersection)
    union = np.sum(binary_patch_predictions) + np.sum(patch_labels) - total_patch_intersection
    return total_patch_intersection, union


def compute_iou(predictions, patch_labels, threshold_binarization):
    intersection, union = compute_intersection_union_patches(predictions, patch_labels, threshold_binarization)
    return intersection / union


def compute_dice(predictions, patch_labels, th_binarization):
    intersection, union = compute_intersection_union_patches(predictions, patch_labels, th_binarization)
    return (2 * intersection) / (union + intersection)


def compute_accuracy_on_segmentation(predictions, patch_labels, th_binarization, th_iou):
    iou_score = compute_iou(predictions, patch_labels, th_binarization)
    return np.greater_equal(iou_score, th_iou, dtype=float)


def get_dice_and_accuracy_pascal(inst_labels, inst_pred):
    dice_collection_all_images = []
    accuracy_all_images = []
    total_images = inst_labels[0].shape[0]
    for image_ind in range(0, total_images):
        all_instances_labels = inst_labels[0].reshape(total_images, -1)

        dice_collection = []
        accuracy_collection = []

        for classifier_ind in range(0, 5):
            inst_predictions_classifier = inst_pred[classifier_ind].reshape(total_images, -1)

            binary_instance_labels = np.array(inst_predictions_classifier[image_ind] >= 0.5, dtype=bool)
            dice = compute_dice(binary_instance_labels, all_instances_labels[image_ind], th_binarization=0.5)
            acc = compute_accuracy_on_segmentation(binary_instance_labels, all_instances_labels[image_ind], 0.5, 0.1)
            dice_collection.append(dice)
            accuracy_collection.append(acc)

        dice_collection_all_images.append(dice_collection)
        accuracy_all_images.append(accuracy_collection)
    return np.array(dice_collection_all_images), np.array(accuracy_all_images)


def get_instance_auc_stability_score_all_classifiers(inst_labels, inst_pred, stability_coll):
    inst_labels = np.array(inst_labels)
    ensure_file_contain_same(inst_labels, len(inst_labels))

    image_auc_collection_all_classifiers = []
    pairwise_stability_all_images = []
    total_images = inst_labels[0].shape[0]
    # instance auc
    # FOR EACH IMAGE, THE PREDICTIONS OF EACH CLASSIFIERS ARE COMPARED WITH THE WHOLE BAG AND AUC IS COMPUTED
    for image_ind in range(0, total_images):
        all_instances_labels = inst_labels[0].reshape(total_images, -1)
        image_auc_collection = []
        image_stab_scores = []

        for classifier_ind in range(0, 5):
            inst_predictions_classifier = inst_pred[classifier_ind].reshape(total_images, -1)
            inst_auc_classifiers = roc_auc_score(all_instances_labels[image_ind],
                                                 inst_predictions_classifier[image_ind])
            image_auc_collection.append(inst_auc_classifiers)
            stab_scores = stability_coll[classifier_ind, classifier_ind + 1:, image_ind]
            image_stab_scores = np.concatenate((image_stab_scores, stab_scores))

        pairwise_stability_all_images.append(image_stab_scores)
        image_auc_collection_all_classifiers.append(image_auc_collection)

    # TOTAL_IMAGES x TOTAL_CLASSIFIERS
    auc_res = np.array(image_auc_collection_all_classifiers)
    # TOTAL_IMAGES x 10 combinations of stability
    stability_res = np.array(pairwise_stability_all_images)

    return auc_res, stability_res


def compute_ap(inst_labels, inst_pred):
    image_ap_collection_all_classifiers = []
    # pairwise_stability_all_images = []
    total_images = inst_labels[0].shape[0]
    # instance auc
    # FOR EACH IMAGE, THE PREDICTIONS OF EACH CLASSIFIERS ARE COMPARED WITH THE WHOLE BAG AND AUC IS COMPUTED
    for image_ind in range(0, total_images):
        all_instances_labels = inst_labels[0].reshape(total_images, -1)
        ap_collection = []

        for classifier_ind in range(0, 5):
            inst_predictions_classifier = inst_pred[classifier_ind].reshape(total_images, -1)
            ap_classifiers = average_precision_score(all_instances_labels[image_ind],
                                                     inst_predictions_classifier[image_ind])
            ap_collection.append(ap_classifiers)

        image_ap_collection_all_classifiers.append(ap_collection)

    # TOTAL_IMAGES x TOTAL_CLASSIFIERS
    ap_res = np.array(image_ap_collection_all_classifiers)
    return ap_res


def stability_all_classifiers(config, classifiers, only_segmentation_images,
                              only_positive_images, visualize_per_image):
    '''

    :param config:
    :param classifiers: list of the unique names of all classifiers
    :param only_segmentation_images: True: only images with available segmentation to consider
    :param only_positive_images: True: only images with positive labelto consider
    :param visualize_per_image: True: visualize predictions for each image and
                                visualize stability scores correlation per image
    :return: saves .csv files of stability for each image across classifiers,
            visualizations for each stability score across classifiers and per image,
             visualizations of nan values of the stability scores
    '''
    image_path = config['image_path']
    stability_res_path = config['stability_results']
    xray_dataset = config['use_xray_dataset']
    use_pascal_dataset = config['use_pascal_dataset']
    class_name = config['class_name']

    if xray_dataset:
        dataset_identifier = 'xray'
    elif use_pascal_dataset:
        dataset_identifier = 'pascal'
    else:
        dataset_identifier = 'mura'

    image_labels_collection, image_index_collection, raw_predictions_collection, \
    bag_labels_collection, bag_predictions_collection, identifier = get_analysis_data_subset(config,
                                                                                             classifiers,
                                                                                             only_segmentation_images,
                                                                                             only_positive_images)

    dataset_identifier += identifier
    jacc_coll, corr_jacc_coll, _, _, _, corr_iou = get_binary_scores_forthreshold_v2(0.5, raw_predictions_collection)

    pearson_corr_collection, spearman_rank_corr_collection = compute_correlation_scores_v2(raw_predictions_collection)

    reshaped_jacc_coll = np.asarray(jacc_coll).reshape(5, 5, len(image_index_collection[0]))
    reshaped_corr_jacc_coll = np.asarray(corr_jacc_coll).reshape(5, 5, len(image_index_collection[0]))
    reshaped_spearman_coll = np.asarray(spearman_rank_corr_collection).reshape(5, 5, len(image_index_collection[0]))
    reshaped_corr_iou = np.asarray(corr_iou).reshape(5, 5, len(image_index_collection[0]))

    xyaxis = ['classifier1', 'classifier2', 'classifier3', 'classifier4', 'classifier5']
    xyaxis_short = ['Cl.1', 'Cl. 2', 'Cl. 3', 'Cl. 4', 'Cl. 5']
    if visualize_per_image:
        # the for loop shows stability per image - which may be interesting only in specific cases
        for idx in range(0, len(image_index_collection[0])):
            img_ind = get_image_index(xray_dataset, image_index_collection[0], idx)

            print("index " + str(idx))
            if not np.isnan(reshaped_corr_jacc_coll[:, :, idx]).all():
                visualize_correlation_heatmap(reshaped_corr_jacc_coll[:, :, idx], stability_res_path,
                                              '_classifiers_jaccard_' + str(img_ind),
                                              xyaxis, dropDuplicates=True)

                combine_correlation_heatmaps_next_to_each_other(reshaped_corr_jacc_coll[:, :, idx],
                                                                reshaped_spearman_coll[:, :, idx],
                                                                "Corrected Jaccard", "Spearman rank",
                                                                xyaxis_short, stability_res_path, str(img_ind),
                                                                drop_duplicates=True)
            visualize_correlation_heatmap(reshaped_spearman_coll[:, :, idx], stability_res_path,
                                          '_classifiers_spearman_' + str(img_ind),
                                          xyaxis, dropDuplicates=True)
        visualize_5_classifiers(xray_dataset, use_pascal_dataset, image_index_collection, image_labels_collection,
                                raw_predictions_collection, image_path, stability_res_path, class_name, '_test_5_class')
    ## ADD inst AUC vs score
    ma_corr_jaccard_images = np.ma.masked_array(reshaped_corr_jacc_coll, np.isnan(reshaped_corr_jacc_coll))
    ma_jaccard_images = np.ma.masked_array(reshaped_jacc_coll, np.isnan(reshaped_jacc_coll))
    ma_corr_iou = np.ma.masked_array(reshaped_corr_iou, np.isnan(reshaped_corr_iou))
    ma_spearman = np.ma.masked_array(reshaped_spearman_coll, np.isnan(reshaped_spearman_coll))

    ############ visualizing NANs of corrected jaccard ###################
    nan_matrix_norm = get_matrix_total_nans_stability_score(corr_jacc_coll, image_index_collection, normalize=True)
    visualize_correlation_heatmap(nan_matrix_norm, stability_res_path, '_corr_jacc_nan_norm' + identifier, xyaxis,
                                  dropDuplicates=True)
    nan_matrix = get_matrix_total_nans_stability_score(corr_jacc_coll, image_index_collection, normalize=False)
    visualize_correlation_heatmap(nan_matrix, stability_res_path, '_corr_jacc_nan' + identifier, xyaxis,
                                  dropDuplicates=True)

    nan_matrix_jacc_norm = get_matrix_total_nans_stability_score(jacc_coll, image_index_collection, normalize=True)
    visualize_correlation_heatmap(nan_matrix_jacc_norm, stability_res_path, '_jacc_nan_norm' + identifier, xyaxis,
                                  dropDuplicates=True)
    nan_matrix_jacc = get_matrix_total_nans_stability_score(jacc_coll, image_index_collection, normalize=False)
    visualize_correlation_heatmap(nan_matrix_jacc, stability_res_path, '_jacc_nan' + identifier, xyaxis,
                                  dropDuplicates=True)
    nan_matrix_spearman = get_matrix_total_nans_stability_score(spearman_rank_corr_collection,
                                                                image_index_collection, normalize=False)
    visualize_correlation_heatmap(nan_matrix_spearman, stability_res_path, '_spearman_nan' + identifier, xyaxis,
                                  dropDuplicates=True)
    nan_matrix_spearman_norm = get_matrix_total_nans_stability_score(spearman_rank_corr_collection,
                                                                     image_index_collection, normalize=True)
    visualize_correlation_heatmap(nan_matrix_spearman_norm, stability_res_path, '_spearman_nan_norm' + identifier,
                                  xyaxis,
                                  dropDuplicates=True)

    ##### AVERAGE STABILITY ACROSS ALL IMAGES ##############
    average_corr_jacc_index_inst = np.average(ma_corr_jaccard_images, axis=-1)
    visualize_correlation_heatmap(average_corr_jacc_index_inst, stability_res_path,
                                  '_mean_stability_' + str("corr_jaccard") + '_' + identifier,
                                  xyaxis, dropDuplicates=True)
    average_jacc_inst = np.average(ma_jaccard_images, axis=-1)
    visualize_correlation_heatmap(average_jacc_inst, stability_res_path,
                                  '_mean_stability_' + str("jaccard") + '_' + identifier,
                                  xyaxis, dropDuplicates=True)

    average_iou_inst = np.average(ma_corr_iou, axis=-1)
    visualize_correlation_heatmap(average_iou_inst, stability_res_path,
                                  '_mean_stability_' + str("iou") + '_' + identifier,
                                  xyaxis, dropDuplicates=True)

    save_additional_kappa_scores_forthreshold(0.5, raw_predictions_collection, image_index_collection,
                                              reshaped_corr_iou,
                                              reshaped_corr_jacc_coll, stability_res_path)

    avg_abs_sprearman = np.average(abs(ma_spearman), axis=-1)
    visualize_correlation_heatmap(avg_abs_sprearman, stability_res_path,
                                  '_mean_stability_' + str("abs_spearman") + '_' + identifier,
                                  xyaxis, dropDuplicates=True)
    avg_spearman = np.average(ma_spearman, axis=-1)
    visualize_correlation_heatmap(avg_spearman, stability_res_path,
                                  '_mean_stability_' + str("spearman") + '_' + identifier,
                                  xyaxis, dropDuplicates=True)


    #### AVERAGE ACROSS ALL CLASSIFIERS - PER IMAGE #######
    mask_repetition = np.ones((ma_corr_jaccard_images.shape), dtype=bool)
    for i in range(0, 4):
        for j in range(i+1, 5):
            mask_repetition[i, j, :] = False
    mean_all_classifiers_corr_jacc = np.mean(np.ma.masked_array(ma_corr_jaccard_images,
                                                                mask=mask_repetition), axis=(0, 1))
    mean_all_classifiers_jacc = np.mean(np.ma.masked_array(ma_jaccard_images, mask=mask_repetition), axis=(0, 1))

    mean_all_classifiers_iou = np.mean(np.ma.masked_array(ma_corr_iou, mask=mask_repetition), axis=(0, 1))
    mean_all_classifiers_spearman = np.mean(np.ma.masked_array(ma_spearman, mask=mask_repetition), axis=(0, 1))

    save_mean_stability(image_index_collection[0], mean_all_classifiers_jacc, mean_all_classifiers_corr_jacc,
                        mean_all_classifiers_iou,mean_all_classifiers_spearman, stability_res_path, dataset_identifier)


def get_analysis_data_subset(config, classifiers, only_segmentation_images, only_positive_images):
    '''

    :param config: config file
    :param classifiers: list with all classifier names
    :param only_segmentation_images: True: analysis is done only on images with segmentation,
                                    False: analysis is done on all images
    :param only_positive_images: this parameter is considered only if only_segmentation_images is FALSE
                True: analysis done on image with positive label
                False: analysis is done on all images
    :return: Returns the suitable rows of the subset desired
    '''

    prediction_results_path = config['prediction_results_path']

    all_img_labels, all_img_image_ind, all_img_raw_predictions, all_bag_labels, all_bag_predictions = \
        load_predictions_v2(classifiers, prediction_results_path)

    if only_segmentation_images:
        bbox_ind_collection = indices_segmentation_images_v2(all_img_labels)
        image_labels_collection, image_index_collection, raw_predictions_collection, \
        bag_labels_collection, bag_predictions_collection = \
            filter_predictions_files_on_indeces(all_img_labels, all_img_image_ind,
                                                all_img_raw_predictions,
                                                all_bag_predictions, all_bag_labels, bbox_ind_collection)
        identifier = "_bbox"
    else:
        if only_positive_images:
            positive_ind_collection = indices_positive_images(all_bag_labels)
            image_labels_collection, image_index_collection, raw_predictions_collection, \
            bag_labels_collection, bag_predictions_collection = \
                filter_predictions_files_on_indeces(all_img_labels, all_img_image_ind,
                                                    all_img_raw_predictions,
                                                    all_bag_predictions, all_bag_labels, positive_ind_collection)
            identifier = "_pos_img"
            print("finished")
        else:
            image_labels_collection, image_index_collection, raw_predictions_collection, \
            bag_labels_collection, bag_predictions_collection = \
                all_img_labels, all_img_image_ind, all_img_raw_predictions, all_bag_labels, all_bag_predictions
            identifier = "_all_img"

    return image_labels_collection, image_index_collection, raw_predictions_collection, \
           bag_labels_collection, bag_predictions_collection, identifier


def stability_all_classifiers_instance_level(config, classifiers, only_segmentation_images, only_positive_images):
    '''

    :param config:
    :param classifiers: list of results names
    :param only_segmentation_images: True: only images with available segmentation to consider. This should be True
    :param only_positive_images: True: only images with positive label to consider
    :return: saves .csv files of stability for each image across classifiers,
            visualizations for each stability score across classifiers and per image,
             visualizations of nan values of the stability scores
    '''
    image_path = config['image_path']
    stability_res_path = config['stability_results']

    image_labels_collection, image_index_collection, raw_predictions_collection, \
    bag_labels_collection, bag_predictions_collection, identifier = get_analysis_data_subset(config,
                                                                                             classifiers,
                                                                                             only_segmentation_images,
                                                                                             only_positive_images)

    jacc_coll, corr_jacc_coll, _, _, _, corr_iou = get_binary_scores_forthreshold_v2(0.5, raw_predictions_collection)

    pearson_corr_collection, spearman_rank_corr_collection = compute_correlation_scores_v2(raw_predictions_collection)

    reshaped_jacc_coll = np.asarray(jacc_coll).reshape(5, 5, len(image_index_collection[0]))
    reshaped_corr_jacc_coll = np.asarray(corr_jacc_coll).reshape(5, 5, len(image_index_collection[0]))
    reshaped_spearman_coll = np.asarray(spearman_rank_corr_collection).reshape(5, 5, len(image_index_collection[0]))
    reshaped_corr_iou = np.asarray(corr_iou).reshape(5, 5, len(image_index_collection[0]))

    auc_res, stability_res_corr_jacc = get_instance_auc_stability_score_all_classifiers(image_labels_collection,
                                                                                        raw_predictions_collection,
                                                                                        reshaped_corr_jacc_coll)

    ap_res =  compute_ap(image_labels_collection,  raw_predictions_collection)

    avg_auc = np.mean(auc_res, axis=1)
    stand_dev_auc = np.std(auc_res, axis=1)

    avg_stability_jacc = np.mean(np.ma.masked_array(stability_res_corr_jacc, np.isnan(stability_res_corr_jacc)), axis=1)
    stand_dev_stability_jacc = np.std(np.ma.masked_array(stability_res_corr_jacc, np.isnan(stability_res_corr_jacc)),
                                      axis=1)

    make_scatterplot_with_errorbar(avg_auc, 'mean instance AUC', avg_stability_jacc, 'mean adjusted Positive0 Jaccard',
                                   stability_res_path, fitting_curve=False, y_errors=stand_dev_auc,
                                   x_errors=stand_dev_stability_jacc,
                                   error_bar=True, bin_threshold_prefix=0)
    make_scatterplot_with_errorbar(avg_auc, 'mean instance AUC', avg_stability_jacc, 'mean adjusted positive Jaccard',
                                   stability_res_path, fitting_curve=True, y_errors=stand_dev_auc, x_errors=None,
                                   error_bar=True, bin_threshold_prefix=0)
    make_scatterplot_with_errorbar(avg_auc, 'mean instance AUC', avg_stability_jacc, 'mean Aajusted Positive1 Jaccard',
                                   stability_res_path, fitting_curve=False, y_errors=None, x_errors=stand_dev_stability_jacc,
                                   error_bar=True, bin_threshold_prefix=0)

    _, stability_res_spear = get_instance_auc_stability_score_all_classifiers(image_labels_collection,
                                                                              raw_predictions_collection,
                                                                              reshaped_spearman_coll)

    avg_stability_spear = np.mean(stability_res_spear, axis=1)
    stand_dev_stability_spear = np.std(stability_res_spear, axis=1)

    make_scatterplot_with_errorbar(avg_auc, 'mean instance AUC', avg_stability_spear, 'mean Spearman0 correlation',
                                   stability_res_path, fitting_curve=False,
                                   y_errors=stand_dev_auc, x_errors=stand_dev_stability_spear,
                                   error_bar=True, bin_threshold_prefix=0)
    make_scatterplot_with_errorbar(avg_auc, 'mean instance AUC', avg_stability_spear, 'mean Spearman correlation',
                                   stability_res_path, fitting_curve=True, y_errors=stand_dev_auc, x_errors=None,
                                   error_bar=True, bin_threshold_prefix=0)
    make_scatterplot_with_errorbar(avg_auc, 'mean instance AUC', avg_stability_spear, 'mean Spearman1 correlation',
                                   stability_res_path, fitting_curve=False, y_errors=None, x_errors=stand_dev_stability_spear,
                                   error_bar=True, bin_threshold_prefix=0)
    make_scatterplot_with_errorbar_v2(avg_stability_jacc, avg_stability_spear, 'stability_score', avg_auc,
                                      "auc", stability_res_path, y_errors=stand_dev_stability_jacc,
                                      y_errors2=stand_dev_stability_spear, error_bar=False,
                                      bin_threshold_prefix=None, x_errors=None)

    mask_repetition = np.ones((reshaped_corr_jacc_coll.shape), dtype=bool)
    for i in range(0, 4):
        for j in range(i+1, 5):
            mask_repetition[i, j, :] = False
    diag_classifiers_corr_jacc = np.ma.masked_array(reshaped_corr_jacc_coll, mask=mask_repetition)
    diag_masked_corr_jacc = np.ma.masked_array(diag_classifiers_corr_jacc, mask=np.isnan(diag_classifiers_corr_jacc))
    mean_corr_jacc = np.mean(diag_masked_corr_jacc, axis=(0,1))

    diag_classifiers_spearman = np.ma.masked_array(reshaped_spearman_coll, mask=mask_repetition)
    diag_masked_spearman = np.ma.masked_array(diag_classifiers_spearman, mask=np.isnan(diag_classifiers_spearman))
    mean_spearman = np.mean(diag_masked_spearman, axis=(0,1))
    assert mean_corr_jacc.all()==avg_stability_jacc.all(), "error"
    assert mean_spearman.all() ==avg_stability_spear.all(), "error"
    save_mean_stability_auc(image_index_collection[0], auc_res, avg_stability_jacc,
                            avg_stability_spear, stability_res_path, 'bbox', ap_res)


def do_transformation_masks_pascal(image_dir):
    img_width, img_height = load_img(image_dir, target_size=None, color_mode='rgb').size
    decrease_needed = image_larger_input(img_width, img_height, 512, 512)

    # IF one or both sides have bigger size than the input, then decrease is needed
    if decrease_needed:
        ratio = calculate_scale_ratio(img_width, img_height, 512, 512)
        assert ratio >= 1.00, "wrong ratio - it will increase image size"
        assert int(img_height / ratio) == 512 or int(img_width / ratio) == 512, \
            "error in computation"
        image = img_to_array(load_img(image_dir, target_size=(int(img_height / ratio), int(img_width / ratio)),
                                      color_mode='rgb'))
    else:
        # ELSE just open image in its original form
        image = img_to_array(load_img(image_dir, target_size=None, color_mode='rgb'))
    ### PADDING
    pad_needed = padding_needed(image)

    if pad_needed:
        image = pad_image(image, final_size_x=512, final_size_y=512)

    return image


def get_analysis_data_masks_pascal(config, classifiers, indices_to_keep):
        prediction_results_path = config['prediction_results_path']

        all_img_labels, all_img_image_ind, all_img_raw_predictions, all_bag_labels, all_bag_predictions = \
            load_predictions_v2(classifiers, prediction_results_path)

        indices_to_keep_coll = [indices_to_keep, indices_to_keep, indices_to_keep, indices_to_keep, indices_to_keep]
        # bbox_ind_collection = indices_segmentation_images_v2(all_img_labels)
        image_labels_collection, image_index_collection, raw_predictions_collection, \
        bag_labels_collection, bag_predictions_collection = \
            filter_predictions_files_on_indeces(all_img_labels, all_img_image_ind,
                                                all_img_raw_predictions,
                                                all_bag_predictions, all_bag_labels, indices_to_keep_coll)
        identifier = "_bbox"
        return image_labels_collection, image_index_collection, raw_predictions_collection, \
        bag_labels_collection, bag_predictions_collection, identifier


def filter_images_with_masks(mask_path1, mask_path2, image_indices):
    masks = []
    images_ind = []
    mask_parent_path1 = mask_path1.split('/')[-1]
    mask_parent_path2 = mask_path2.split('/')[-1]
    indices = []
    parent_paths = []
    for img_ind in range(0, image_indices.shape[0]):
        parent_path = image_indices[img_ind].split('/')[-2]

        if parent_path == mask_parent_path1 or parent_path == mask_parent_path2:
            if parent_path  == mask_parent_path1:
                masks_path = mask_path1
            else:
                masks_path = mask_path2
            try:
                image_mask = do_transformation_masks_pascal(str(masks_path+"/"+image_indices[img_ind].split('/')[-1]))
                # predictions_with_mask.append(predictions[img_ind])
                masks.append(image_mask)
                images_ind.append(image_indices[img_ind].split('/')[-1])
                indices.append(img_ind)
                parent_paths.append(parent_path)
            except:
                print("Image was not found: "+ str(masks_path+"/"+image_indices[img_ind].split('/')[-1]) )
    return masks, images_ind, indices, parent_paths


def transform_pixels_to_patches(binary_masked, patch_pixels):
    annotation = np.zeros((16, 16))
    for height in range(0, 16):
        for width in range(0, 16):
            max_pixel = binary_masked[height * patch_pixels:(height + 1) * patch_pixels,
                        width * patch_pixels:(width + 1) * patch_pixels].max()

            annotation[height, width] = max_pixel
    return annotation


def convert_mask_image_to_binary_matrix(mask_parent_folder, masks):
    patch_pixels = 32
    all_classifiers_annotations = []
    for classifier_ind in range(5):
        annotations_coll = []
        for ind in range(masks.shape[1]):
            if mask_parent_folder[ind].lower() == 'tugraz_cars':
                ## BLUE CHANNEL is larger than 0
                red_green_channel = np.add(masks[classifier_ind, ind][:, :, 0],  masks[classifier_ind, ind][:, :, 1])
                background_masked = np.ma.masked_where(red_green_channel > 0, red_green_channel).mask


            elif mask_parent_folder[ind].lower() == 'ethz_sideviews_cars':
                ## BLUE CHANNEL is 0
                white_color = masks[classifier_ind, ind][:, :, 0]
                background_masked = np.ma.masked_where(white_color > 0, white_color).mask
            annotation_image = transform_pixels_to_patches(background_masked, patch_pixels)
            annotations_coll.append(annotation_image)
        all_classifiers_annotations.append(annotations_coll)
    return all_classifiers_annotations


def stability_all_classifiers_instance_level_pascal(config, classifiers):
    '''
    This functions measures the instance performance only on the Pascal dataset
    :param config: configurations
    :param classifiers: list of classifier names
    :return: evaluation of the stability score against the instance performance.
             instance performance is measured with the dice score betweeen predictions and available segmentations.
             Saves .csv files for dice score across classifiers for each image and visualizations of stability
             against instance performance.
    '''
    stability_res_path = config['stability_results']
    pascal_image_path = config['pascal_image_path']
    pascal_dir  = str(Path(pascal_image_path).parent).replace("\\", "/")
    masks_path1 = pascal_dir + "/GTMasks/ETHZ_sideviews_cars"

    masks_path_2 =  pascal_dir + "/GTMasks/TUGraz_cars"

    prediction_results_path = config['prediction_results_path']

    masks_labels_coll= []
    image_name_coll = []
    indices_to_keep_coll= []
    for ind in range(5):
        img_ind = np.load(prediction_results_path+'image_indices_'+classifiers[ind], allow_pickle=True)
        gt_masks, image_name_to_keep, indices_to_keep, parents_folder =filter_images_with_masks(masks_path1, masks_path_2, img_ind)
        masks_labels_coll.append(gt_masks)
        image_name_coll.append(image_name_to_keep)
        indices_to_keep_coll.append(indices_to_keep)


    image_labels_collection, image_index_collection, raw_predictions_collection, \
    bag_labels_collection, bag_predictions_collection, identifier = get_analysis_data_masks_pascal(config, classifiers, indices_to_keep)

    jacc_coll, corr_jacc_coll, _, _, _, corr_iou = get_binary_scores_forthreshold_v2(0.5, raw_predictions_collection)

    pearson_corr_collection, spearman_rank_corr_collection = compute_correlation_scores_v2(raw_predictions_collection)

    reshaped_jacc_coll = np.asarray(jacc_coll).reshape(5, 5, len(image_index_collection[0]))
    reshaped_corr_jacc_coll = np.asarray(corr_jacc_coll).reshape(5, 5, len(image_index_collection[0]))
    reshaped_spearman_coll = np.asarray(spearman_rank_corr_collection).reshape(5, 5, len(image_index_collection[0]))
    reshaped_corr_iou = np.asarray(corr_iou).reshape(5, 5, len(image_index_collection[0]))

    masks_labels_coll = np.asarray(masks_labels_coll)
    annotations_coll = convert_mask_image_to_binary_matrix(parents_folder, masks_labels_coll)
    annotations_coll = np.array(annotations_coll)
    auc_res, stability_res_corr_jacc = get_instance_auc_stability_score_all_classifiers(annotations_coll,
                                                                                        raw_predictions_collection,
                                                                                        reshaped_corr_jacc_coll)

    dice, accuracy_iou =  get_dice_and_accuracy_pascal(annotations_coll, raw_predictions_collection)

    ap_res =  compute_ap(annotations_coll,  raw_predictions_collection)

    avg_auc = np.mean(auc_res, axis=1)
    stand_dev_auc = np.std(auc_res, axis=1)

    avg_stability_jacc = np.mean(np.ma.masked_array(stability_res_corr_jacc, np.isnan(stability_res_corr_jacc)), axis=1)
    stand_dev_stability_jacc = np.std(np.ma.masked_array(stability_res_corr_jacc, np.isnan(stability_res_corr_jacc)),
                                      axis=1)

    make_scatterplot_with_errorbar(np.mean(dice, axis=1), 'mean DICE', avg_stability_jacc, 'DICE_mean adjusted Positive0 Jaccard',
                                   stability_res_path, fitting_curve=False, y_errors=np.std(dice, axis=1),
                                   x_errors=None, error_bar=True, bin_threshold_prefix=0)

    make_scatterplot_with_errorbar(avg_auc, 'mean instance AUC', avg_stability_jacc, 'mean adjusted positive Jaccard',
                                   stability_res_path, fitting_curve=False, y_errors=stand_dev_auc, x_errors=None,
                                   error_bar=True, bin_threshold_prefix=0)

    _, stability_res_spear = get_instance_auc_stability_score_all_classifiers(annotations_coll,
                                                                              raw_predictions_collection,
                                                                              reshaped_spearman_coll)

    avg_stability_spear = np.mean(stability_res_spear, axis=1)
    stand_dev_stability_spear = np.std(stability_res_spear, axis=1)

    make_scatterplot_with_errorbar(np.mean(dice, axis=1), 'mean instance AUC', avg_stability_spear, 'dice_mean Spearman0 correlation',
                                   stability_res_path, fitting_curve=False,
                                   y_errors=stand_dev_auc, x_errors=stand_dev_stability_spear,
                                   error_bar=True, bin_threshold_prefix=0)
    make_scatterplot_with_errorbar(np.mean(dice, axis=1), 'mean DICE', avg_stability_spear, 'DICE_mean spear',
                                   stability_res_path, fitting_curve=False, y_errors=np.std(dice, axis=1),
                                   x_errors=None, error_bar=True, bin_threshold_prefix=0)

    make_scatterplot_with_errorbar(avg_auc, 'mean instance AUC', avg_stability_spear, 'mean Spearman correlation',
                                   stability_res_path, fitting_curve=False, y_errors=stand_dev_auc, x_errors=None,
                                   error_bar=True, bin_threshold_prefix=0)
    make_scatterplot_with_errorbar(np.mean(dice, axis=1), 'mean DICE', avg_stability_spear,
                                   'DICE_mean_spearman',
                                   stability_res_path, fitting_curve=True, y_errors=np.std(dice, axis=1),
                                   x_errors=None, error_bar=True, bin_threshold_prefix=0)
    mask_repetition = np.ones(reshaped_corr_jacc_coll.shape, dtype=bool)
    for i in range(0, 4):
        for j in range(i+1, 5):
            mask_repetition[i, j, :] = False
    diag_classifiers_corr_jacc = np.ma.masked_array(reshaped_corr_jacc_coll, mask=mask_repetition)
    diag_masked_corr_jacc = np.ma.masked_array(diag_classifiers_corr_jacc, mask=np.isnan(diag_classifiers_corr_jacc))
    mean_corr_jacc = np.mean(diag_masked_corr_jacc, axis=(0,1))

    diag_classifiers_spearman = np.ma.masked_array(reshaped_spearman_coll, mask=mask_repetition)
    diag_masked_spearman = np.ma.masked_array(diag_classifiers_spearman, mask=np.isnan(diag_classifiers_spearman))
    mean_spearman = np.mean(diag_masked_spearman, axis=(0,1))
    assert mean_corr_jacc.all()==avg_stability_jacc.all(), "error"
    assert mean_spearman.all() ==avg_stability_spear.all(), "error"
    save_mean_stability_auc(image_index_collection[0], auc_res, avg_stability_jacc,
                            avg_stability_spear, stability_res_path, 'bbox', ap_res)

    save_mean_dice(image_index_collection[0],dice, accuracy_iou, stability_res_path, 'bbox')
