import torchio as tio
from pathlib import Path
import torch
import numpy as np
import copy


def all_metric(gt, wt_pred, et_pred, tc_pred):
    wt_dice, wt_recall, wt_specificity, wt_hs95 = metric(gt[0], wt_pred)
    et_dice, et_recall, et_specificity, et_hs95 = metric(gt[1], et_pred)
    tc_dice, tc_recall, tc_specificity, tc_hs95 = metric(gt[2], tc_pred)
    return [wt_dice, wt_recall, wt_specificity, wt_hs95], [et_dice, et_recall, et_specificity, et_hs95], [tc_dice, tc_recall, tc_specificity, tc_hs95]


def metric(gt, pred):
    #* input shape: (batch, channel, height, width)
    gt = gt.squeeze()  # (240,240)
    pred = pred.squeeze()  # (240,240)

    preds = pred.detach().numpy()
    gts = gt.detach().numpy()

    pred = preds.astype(int)  # float data does not support bit_and and bit_or
    gdth = gts.astype(int)  # float data does not support bit_and and bit_or
    fp_array = copy.deepcopy(pred)  # keep pred unchanged
    fn_array = copy.deepcopy(gdth)
    gdth_sum = np.sum(gdth)
    pred_sum = np.sum(pred)
    intersection = gdth & pred
    union = gdth | pred
    intersection_sum = np.count_nonzero(intersection)
    union_sum = np.count_nonzero(union)

    tp_array = intersection

    tmp = pred - gdth
    fp_array[tmp < 1] = 0

    tmp2 = gdth - pred
    fn_array[tmp2 < 1] = 0

    tn_array = np.ones(gdth.shape) - union

    tp, fp, fn, tn = np.sum(tp_array), np.sum(fp_array), np.sum(fn_array), np.sum(tn_array)

    smooth = 0.001
    precision = tp / (pred_sum + smooth)
    recall = tp / (gdth_sum + smooth)
    specificity = tn / (tn + fp + smooth)

    false_positive_rate = fp / (fp + tn + smooth)
    false_negtive_rate = fn / (fn + tp + smooth)

    jaccard = intersection_sum / (union_sum + smooth)
    dice = 2 * intersection_sum / (gdth_sum + pred_sum + smooth)

    # hs95 = 0
    # hs95 = hausdorff_95(gdth, pred, (1, 1))
    # hs95 = hausdorff_95(preds, gts, (1, 1, 1))

    return jaccard, dice


# def hausdorff_95(gt_array, pred_array, spacing):
#     '''
#     :params gt_array: ground true mask
#     :params pred_array: the result of segmentation
#     :params num_class: label number
#     :params spacing: spacing of the image
#     '''
#     Hausdorff_95_score = []
#     gt_array = gt_array.astype(bool)
#     pred_array = pred_array.astype(bool)

#     # compute Hausdorff_95 score
#     surface_distances = surface_distance.compute_surface_distances(pred_array, gt_array, spacing)
#     hs95 = surface_distance.compute_robust_hausdorff(surface_distances, 95)
#     return hs95