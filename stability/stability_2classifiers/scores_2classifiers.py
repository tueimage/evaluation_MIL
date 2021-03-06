import numpy as np
from scipy.stats import rankdata, spearmanr, kendalltau


from stability.preprocessor.preprocessing import calculate_subsets_between_two_classifiers


def positive_Jaccard_index_batch(bin_pred1, bin_pred2, P):
    # jaccard_coll = []
    # assert bin_pred1.shape == bin_pred2.shape, "Predictions don't have same shapes"
    #
    # for obs in range(0, bin_pred1.shape[0]):

    sum_preds = bin_pred1 + bin_pred2
    n11_mask = np.array(sum_preds > 1, dtype=int)
    n10_n01_mask = np.array(sum_preds ==1, dtype=int)

    n11 = np.sum(n11_mask.reshape((n11_mask.shape[0], P*P)), axis=1)
    n10_n01 = np.sum(n10_n01_mask.reshape((n10_n01_mask.shape[0], P*P)), axis=1)
    np.seterr(divide='ignore', invalid='ignore')
    pos_jaccard_dist = n11/(n11+n10_n01)

    # n00, n10, n01, n11
    d, c, b, a  = calculate_subsets_between_two_classifiers(bin_pred1, bin_pred2)
    # N = a + d  + b + c
    # expected_positive_overlap = (n11 + n01) * (n11 + n10) / N
    corrected_score2 = a / (a + b + c)
    assert ((np.ma.masked_array(pos_jaccard_dist, np.isnan(pos_jaccard_dist)) ==
             np.ma.masked_array(corrected_score2,
                                np.isnan(corrected_score2)))).all(), "Error in computing some of the index! "
        # jaccard_coll.append(pos_jaccard_dist)

    return pos_jaccard_dist


def calculate_spearman_rank_coefficient(pred1, pred2):
    spearman_corr_coll = []
    assert pred1.shape[0] == pred2.shape[0], "Ensure the predictions have same shape!"
    for obs in range(0, pred1.shape[0]):
        rank_image1 = rankdata(pred1.reshape(-1, 16*16*1)[obs])
        rank_image2 = rankdata(pred2.reshape(-1, 16*16*1)[obs])
        rho, pval = spearmanr(rank_image1, rank_image2)
        spearman_corr_coll.append(rho)
    return spearman_corr_coll


def calculate_IoU(bin_pred1, bin_pred2):
    sum_preds = bin_pred1 + bin_pred2
    n11_mask = np.array(sum_preds > 1, dtype=int)
    n00_mask = np.array(sum_preds == 0, dtype=int)
    n10_n01_mask = np.array(sum_preds == 1, dtype=int)

    n11 = np.sum(n11_mask.reshape((n11_mask.shape[0], 16 * 16 * 1)), axis=1)
    n00 = np.sum(n00_mask.reshape((n00_mask.shape[0], 16*16*1)), axis=1)
    n10_n01 = np.sum(n10_n01_mask.reshape((n10_n01_mask.shape[0], 16 * 16 * 1)), axis=1)
    np.seterr(divide='ignore', invalid='ignore')

    iou_score = (n11+n00) / (n11 + n10_n01+n00)
    return iou_score

#
# def calculate_pearson_coefficient_batch(raw_pred1, raw_pred2):
#     correlation_coll = []
#     assert raw_pred1.shape == raw_pred2.shape, "Predictions don't have same shapes"
#
#     for ind in range(0, raw_pred1.shape[0]):
#         corr_coef = np.corrcoef(raw_pred1.reshape((raw_pred1.shape[0], 16*16*1))[ind],
#                     raw_pred2.reshape((raw_pred2.shape[0], 16*16*1))[ind])
#         correlation_coll.append(corr_coef[0,1])
#         corr_coef2 = np.corrcoef(raw_pred1.reshape((raw_pred1.shape[0], 16*16*1))[ind],
#                     raw_pred2.reshape((raw_pred2.shape[0], 16*16*1))[ind], rowvar=False)
#         assert corr_coef[0, 1]==corr_coef2[0, 1], "think on the dimensions of the correlation computed "
#
#     return correlation_coll


def calculate_pearson_coefficient_batch(raw_pred1, raw_pred2):
    correlation_coll = []
    assert raw_pred1.shape == raw_pred2.shape, "Predictions don't have same shapes"

    for ind in range(0, raw_pred1.shape[0]):
        corr_coef = np.corrcoef(raw_pred1.reshape((raw_pred1.shape[0], 16*16*1))[ind],
                    raw_pred2.reshape((raw_pred2.shape[0], 16*16*1))[ind])
        correlation_coll.append(corr_coef[0,1])
        corr_coef2 = np.corrcoef(raw_pred1.reshape((raw_pred1.shape[0], 16*16*1))[ind],
                    raw_pred2.reshape((raw_pred2.shape[0], 16*16*1))[ind], rowvar=False)
        # assert corr_coef[0, 1]==corr_coef2[0, 1], "think on the dimensions of the correlation computed "

    return correlation_coll


def calculate_kendallstau_coefficient_batch(raw_pred1, raw_pred2):
    correlation_coll = []
    assert raw_pred1.shape == raw_pred2.shape, "Predictions don't have same shapes"

    for ind in range(0, raw_pred1.shape[0]):
        corr_coef  = kendalltau(raw_pred1.reshape((raw_pred1.shape[0], 16*16*1))[ind],
                    raw_pred2.reshape((raw_pred2.shape[0], 16*16*1))[ind])
        correlation_coll.append(corr_coef[0])
        corr_coef2 = kendalltau(raw_pred1.reshape((raw_pred1.shape[0], 16*16*1))[ind],
                    raw_pred2.reshape((raw_pred2.shape[0], 16*16*1))[ind])
        assert corr_coef[0]==corr_coef2[0], "think on the dimensions of the correlation computed "

    return correlation_coll


def corrected_IOU(bin_pred1, bin_pred2):
    n00, n10, n01, n11 = calculate_subsets_between_two_classifiers(bin_pred1, bin_pred2)

    N = n00 + n11 + n10 + n01
    # expected_positive_overlap = ((n11 + n01) * (n11 + n10)) / N
    # expected_negative_overlap = ((n00 + n01) * (n00 + n10)) / N
    expected_positive_overlap = (((n11 + n01)/N) * ((n11 + n10)/N))* N
    expected_negative_overlap = (((n00 + n01)/N) * ((n00 + n10)/N)) * N

    corrected_score = ((n11 + n00 - expected_positive_overlap - expected_negative_overlap) /
                       (n10 + n11 + n01 + n00 - expected_positive_overlap - expected_negative_overlap))
    corrected_score2 = (2*n00 * n11 - 2*n10 * n01) / (2*(n00 * n11) - 2*(n01 * n10) + ((n10 + n01) * N))
    simplf_div = (n11*n10 + n11*n01 + 2*n11*n00 + n10*n10 + n10*n00 + n01*n01 + n01*n00)
    corrected_score3 = (2*n00 * n11 - 2*n10 * n01) /(simplf_div)
    assert ((np.ma.masked_array(corrected_score, np.isnan(corrected_score)) ==
             np.ma.masked_array(corrected_score2,
                                np.isnan(corrected_score2)))).all(), "Error in computing some of the index! "
    assert ((np.ma.masked_array(corrected_score, np.isnan(corrected_score)) ==
             np.ma.masked_array(corrected_score3,
                                np.isnan(corrected_score3)))).all(), "Error in computing some of the index! "
    # return np.ma.masked_array(corrected_score, np.isnan(corrected_score))
    return corrected_score


#### SOURCE: "High agreement but low kappa: II. Resolving the paradoxes" Cicchetti, Feinstein
def compute_positive_agreement_ratio(positive_overlap, observer1_positive, observer2_positive):
    return (2*positive_overlap)/(observer1_positive + observer2_positive)


def compute_negative_agreement_ratio(negative_overlap, observer1_negative, observer2_negative):
    return (2*negative_overlap)/(observer1_negative + observer2_negative)


def compute_total_agreement_ratio(neg_overlap, pos_overlap, N):
    return (pos_overlap+neg_overlap)/N


def compute_f1_f2(observer1_pos, observer1_neg):
    return observer1_pos - observer1_neg


def compute_g1_g2(observer2_pos, observer2_neg):
    return observer2_pos - observer2_neg


def compute_additional_scores_kappa(bin_pred1, bin_pred2):
    n00, n10, n01, n11 = calculate_subsets_between_two_classifiers(bin_pred1,bin_pred2)
    observer1_positive = n11 + n10
    observer2_positive = n11 + n01
    p_pos = compute_positive_agreement_ratio(n11, observer1_positive, observer2_positive)
    p_neg = compute_negative_agreement_ratio(n00, observer1_negative=n00+n01, observer2_negative=n00+n10)
    po= compute_total_agreement_ratio(n00, n11, n11+n00+n10+n01)
    f1_f2 = compute_f1_f2(n11+n10, n00+n01)
    g1_g2 = compute_g1_g2(n11 + n01, n00 + n10)
    return po, p_pos, p_neg, p_pos-p_neg, f1_f2, g1_g2


def corrected_positive_Jaccard(bin_pred1, bin_pred2):
    n00, n10, n01, n11 = calculate_subsets_between_two_classifiers(bin_pred1,bin_pred2)

    N = n00 + n11 + n10 + n01
    expected_positive_overlap = (n11+n01)*(n11+n10)/N

    corrected_score = ((n11 - expected_positive_overlap)/(n10 + n11 +n01 - expected_positive_overlap))
    corrected_score2 = (n00*n11 - n10*n01)/((n00*n11) - (n01*n10) + ((n10+n01)*N))
    assert ((np.ma.masked_array(corrected_score, np.isnan(corrected_score)) ==
             np.ma.masked_array(corrected_score2, np.isnan(corrected_score2)))).all(), "Error in computing some of the index! "
    return np.ma.masked_array(corrected_score, np.isnan(corrected_score))
    # return corrected_score


def corrected_Jaccard_pigeonhole(bin_pred1, bin_pred2):
    n00, n10, n01, n11 = calculate_subsets_between_two_classifiers(bin_pred1, bin_pred2)

    N = n00 + n11 + n10 + n01
    pigeonhole_positive_correction = (2*n11 + n01 + n10) - N
    max_overlap = np.maximum(pigeonhole_positive_correction, 0)

    corrected_score = ((n11 - max_overlap) /
                       (n10 + n11 + n01 - max_overlap))
    return np.ma.masked_array(corrected_score, np.isnan(corrected_score))
    # return corrected_score


def overlap_coefficient(bin_pred1, bin_pred2, P):
    n00, n10, n01, n11 = calculate_subsets_between_two_classifiers(bin_pred1, bin_pred2, P)
    min_n01_n10 = np.minimum(n10, n01)
    return n11/(min_n01_n10 + n11)


def corrected_overlap_coefficient(bin_pred1, bin_pred2):
    n00, n10, n01, n11 = calculate_subsets_between_two_classifiers(bin_pred1, bin_pred2)

    min_n01_n10 = np.minimum(n10, n01)
    # assert (n11+n01) == np.sum(np.ma.masked_equal(bin_pred2, 0).reshape(-1, 16*16*1), axis=1),\
    #     "Error with computing the positive instances "
    # assert (n11+n10) == np.sum(np.ma.masked_equal(bin_pred1, 0).reshape(-1, 16*16*1), axis=1),\
    #     "Error with computing the positive instances "
    N = n00 + n11 + n10 + n01
    expected_overlap = (n11+n01)*(n11+n10)/N
    # 0/0 -> nan so convert nan to 0
    # corrected_score = np.nan_to_num((n11 - expected_overlap)/(np.minimum((n11+n01), (n11+n10)) - expected_overlap))
    # corrected_score2 = np.nan_to_num((n00*n11 - n10*n01)/((min_n01_n10+n11)*(min_n01_n10 + n00)))
    corrected_score = ((n11 - expected_overlap)/(np.minimum((n11+n01), (n11+n10)) - expected_overlap))
    corrected_score2 = ((n00*n11 - n10*n01)/((min_n01_n10+n11)*(min_n01_n10 + n00)))

    assert ((np.ma.masked_array(corrected_score, np.isnan(corrected_score)) ==
             np.ma.masked_array(corrected_score2, np.isnan(corrected_score2)))).all(), "Error in computing some of the index! "
    return np.ma.masked_array(corrected_score, np.isnan(corrected_score))


