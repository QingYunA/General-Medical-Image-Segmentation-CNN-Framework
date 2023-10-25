import torchio as tio
import SimpleITK as sitk
from pathlib import Path
import argparse
import torch
import numpy as np
import copy
import os
import json
import pandas as pd
from logger import create_logger, metrics_logger
from tqdm import tqdm
from hparam import hparams as hp
from cl_dice import clDice_metric,clDice
from soft_skeleton import soft_skel
BLUE = '\033[94m'
END = '\033[0m'

pred_dir = './results/dice_loss/pred_file'
gt_dir= '/data/cc/Ying-TOF/test/label1'
metric_save_path = './logs'

def parse_training_args(parser):
    """
    Parse commandline arguments.
    """
    parser.add_argument('-o', '--output_dir', type=str, default=hp.output_dir, help='Directory to save checkpoints')
    parser.add_argument('--latest-checkpoint-file',
                        type=str,
                        default=hp.latest_checkpoint_file,
                        help='Store the latest checkpoint in each epoch')
    parser.add_argument('--gpus', type=str, default='0', help='use which gpu')

    parser.add_argument('--epochs', type=int, default=hp.total_epochs, help='Number of total epochs to run')
    parser.add_argument('--epochs-per-checkpoint',
                        type=int,
                        default=hp.epochs_per_checkpoint,
                        help='Number of epochs per checkpoint')
    parser.add_argument('--batch_size', type=int, default=hp.batch_size, help='batch-size')
    parser.add_argument('--patch_size', type=str, default=hp.patch_size, help='patch-size')
    parser.add_argument('--swap_size', type=int, default=hp.swap_size, help='patch-size used for swap')
    parser.add_argument('--swap_iters', type=int, default=hp.swap_iters, help='swap iterations')
    parser.add_argument('--in_classes', type=int, default=hp.in_class, help='input channel')
    parser.add_argument('--out_classes', type=int, default=hp.out_class, help='output channel')
    parser.add_argument('--data_path', type=str, default=hp.data_path, help='source data_dir')
    parser.add_argument('-g','--gt_path', type=str, default=hp.gt_path, help='gt_data_dir')
    parser.add_argument('-d', "--dataset", type=str, choices=['brats', 'else'], default=hp.dataset, help='dataset type')
    parser.add_argument("--init_lr", type=float, default=hp.init_lr, help="learning rate")
    parser.add_argument("--load_mode", type=int, default=hp.load_mode, help="decide how to load model")
    parser.add_argument("--init_type", type=str, default=hp.init_type, help="init type")
    parser.add_argument("-p", "--pred_path", type=str, default=hp.pred_path, help="pred_path")

    parser.add_argument('--fold_arch', type=str, default=hp.fold_arch, help='fold_arch')
    parser.add_argument('-k', "--ckpt", type=str, default=hp.ckpt, help="path to the checkpoints to resume training")
    parser.add_argument("--use_scheduler", action="store_true", help="use scheduler")
    parser.add_argument('--aug', action='store_true', help='data augmentation')
    parser.add_argument('--save_arch', type=str, default=hp.save_arch, help="save arch")
    parser.add_argument('--file_name', type=str, default=os.path.basename(__file__).split('.')[0], help='file name')

    parser.add_argument('--cudnn-enabled', default=True, help='Enable cudnn')
    parser.add_argument('--cudnn-benchmark', default=True, help='Run cudnn benchmark')

    return parser


def make_subject(image_paths, label_paths, suffixes):
    images_dir = Path(image_paths)
    labels_dir = Path(label_paths)

    image_paths = sorted(images_dir.glob(suffixes))
    label_paths = sorted(labels_dir.glob(suffixes))
    subjects = []
    for (image_path, label_path) in zip(image_paths, label_paths):
        subject = tio.Subject(
            pred=tio.ScalarImage(image_path),
            gt=tio.LabelMap(label_path),
        )
        subjects.append(subject)
    return subjects


def save_csv(data, path):
    df = pd.DataFrame(data)
    # 添加列名
    df.columns = ["jaccard", "dice","cl_dice","score"]
    df.loc[len(df)] = [df.iloc[:, 0].mean(), df.iloc[:, 1].mean(), df.iloc[:, 2].mean(), df.iloc[:, 3].mean()]
    # 保存为csv文件
    df.to_csv(path, encoding='utf-8')


def metric_evaluation(predict_dir, labels_dir, logger,skel_path, suffixes="*.mhd"):

    subjects = make_subject(predict_dir, labels_dir, suffixes)

    training_set = tio.SubjectsDataset(subjects)

    dice_arr = []
    Data = []

    for i, subj in tqdm(enumerate(training_set._subjects), total=len(training_set._subjects),
                        desc="Generating metrics"):
        data = []
        gt = subj['gt'][tio.DATA]

        # subj = toc(subj)
        pred = subj['pred'][tio.DATA]  # .permute(0,1,3,2)

        # preds.append(pred)
        # gts.append(gt)

        preds = pred.numpy()
        gts = gt.numpy()

        pred = preds.astype(int)  # float data does not support bit_and and bit_or
        gdth = gts.astype(int)  # float data does not support bit_and and bit_or
        fp_array = copy.deepcopy(pred)  # keep pred unchanged
        fn_array = copy.deepcopy(gdth)
        gdth_sum = np.sum(gdth)
        pred_sum = np.sum(pred)

        intersection = gdth & pred  # only both 1 will be 1
        union = gdth | pred
        intersection_sum = np.count_nonzero(intersection)  # sum of nonzero elements
        union_sum = np.count_nonzero(union)  #

        tp_array = intersection

        tmp = pred - gdth
        fp_array[tmp < 1] = 0  # fp : false positive

        tmp2 = gdth - pred
        fn_array[tmp2 < 1] = 0

        tn_array = np.ones(gdth.shape) - union

        tp, fp, fn, tn = np.sum(tp_array), np.sum(fp_array), np.sum(fn_array), np.sum(tn_array)

        smooth = 0.001
        # precision = tp / (pred_sum + smooth)
        # recall = tp / (gdth_sum + smooth)
        cl_dice,skel_pred,skel_gt = clDice_metric(preds, gts)


        #! save skel
        # img_skel_pred=sitk.GetImageFromArray(skel_pred)
        # img_skel_gt=sitk.GetImageFromArray(skel_gt)

        # img_skel_pred.SetSpacing([1.0, 1.0, 1.0])
        # img_skel_gt.SetSpacing([1.0, 1.0, 1.0])

        # img_skel_pred.SetOrigin([0.0, 0.0, 0.0])
        # img_skel_gt.SetOrigin([0.0, 0.0, 0.0])

        # sitk.WriteImage(img_skel_pred,os.path.join(skel_path,f"skel_pred_{i}.mhd"))
        # sitk.WriteImage(img_skel_gt,os.path.join(skel_path,f"skel_gt_{i}.mhd"))

        # img_skel_pred.SetPixelID(sitk.sitkFloat64)
        # img_skel_gt.SetPixelID(sitk.sitkFloat64)

        # img_skel_pred.SetRegions([skel_pred.shape[2], skel_pred.shape[1], skel_pred.shape[0]])
        # img_skel_gt.SetRegions([skel_gt.shape[2], skel_gt.shape[1], skel_gt.shape[0]])



        # false_positive_rate = fp / (fp + tn + smooth)
        # false_negative_rate = fn / (fn + tp + smooth)

        jaccard = intersection_sum / (union_sum + smooth)
        dice = 2 * intersection_sum / (gdth_sum + pred_sum + smooth)
        logger.info(# f'\nprecision: {precision}\n'
                    # f'recall: {recall}\n'
                    # f'false_positive_rate: {false_positive_rate}\n'
                    # f'false_negative_rate: {false_negative_rate}\n'
                    f'\njaccard: {jaccard}\n'
                    f'cl_dice: {cl_dice}\n'
                    f'dice: {dice}\n')
        dice_arr.append(dice)
        # data.append(precision)
        # data.append(recall)
        # data.append(false_positive_rate)
        # data.append(false_negative_rate)
        data.append(jaccard)
        data.append(dice)
        data.append(cl_dice)
        data.append((jaccard+dice+cl_dice)/3)
        Data.append(data)
    return Data, np.mean(dice_arr)

def soft_cldice(y_true,y_pred,iters=2,alpha=0.5,smooth=1.):

    skel_pred = soft_skel(y_pred, iters)
    skel_true = soft_skel(y_true, iters)
    tprec = (torch.sum(torch.multiply(skel_pred, y_true)[:, 1:, ...]) +
                smooth) / (torch.sum(skel_pred[:, 1:, ...]) + smooth)
    tsens = (torch.sum(torch.multiply(skel_true, y_pred)[:, 1:, ...]) +
                smooth) / (torch.sum(skel_true[:, 1:, ...]) + smooth)
    cl_dice = 1. - 2.0 * (tprec * tsens) / (tprec + tsens)
    return cl_dice


if __name__ == '__main__':
    hp = hp()
    parser = argparse.ArgumentParser(description='PyTorch Medical Segmentation Training')
    parser = parse_training_args(parser)
    args, _ = parser.parse_known_args()
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    logger = metrics_logger(output_dir=args.output_dir, name='Evaluate Metrics')

    metric_save_path = os.path.join(args.output_dir)
    gt_dir = args.gt_path
    pred_dir = args.pred_path
    skel_path=os.path.join(args.output_dir,"skel")
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(metric_save_path, exist_ok=True)
    os.makedirs(skel_path, exist_ok=True)

    logger.info(f'\nPrediction results: {pred_dir}\n'
                f'Ground truth: {gt_dir}\n'
                f'Metric saved in: {metric_save_path}\n')

    # metric and save csv
    Data, mean_dice = metric_evaluation(pred_dir, gt_dir, logger, skel_path,'*.mhd')
    csv_path = os.path.join(metric_save_path, 'metric_evaluation.csv')
    save_csv(Data, csv_path)
    logger.info(f'\n{BLUE}Mean dice{END}: {mean_dice}\n'
                f'{BLUE}Metric saved in{END}: {csv_path}\n')

    # save hparam to json
    # with open(os.path.join(metric_dir, 'hparam.json'), 'w') as f:
    #     json.dump(hp.__dict__, f, indent=4)
    # print("hparam.json saveed at", os.path.join(metric_dir, 'hparam.json'))

