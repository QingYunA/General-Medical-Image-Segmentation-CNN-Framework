import os
import argparse
import torch
from torch.utils.tensorboard import SummaryWriter
import torch.distributed as dist
import torchio as tio
from torchio.transforms import (
    ZNormalization, )
from tqdm import tqdm
from utils.metric import metric
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR, CosineAnnealingLR
from process_input import process_x, process_gt
import numpy as np
from logger import create_logger
from utils import yaml_read
from utils.conf_base import Default_Conf
from rich.progress import (
    BarColumn,
    Progress,
    TextColumn,
    MofNCompleteColumn,
    TimeRemainingColumn,
)

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # ! solve warning


def parse_training_args(parser):
    """
    Parse commandline arguments.
    """
    parser.add_argument('-o', '--output_dir', type=str, help='Directory to save checkpoints')
    parser.add_argument('--conf_path', type=str, help='conf_path')
    parser.add_argument('--gpus', type=str, help='use which gpu')
    parser.add_argument('--epochs', type=int, help='Number of total epochs to run')
    parser.add_argument('--batch_size', type=int, help='batch-size')
    parser.add_argument('--network', type=str, help='decide which network to use')
    parser.add_argument("--init_lr", type=float, help="learning rate")
    parser.add_argument("--load_mode", type=int, help="decide how to load model")
    parser.add_argument('-k', "--ckpt", type=str, help="path to the checkpoints to resume training")
    parser.add_argument("--use_scheduler", action="store_true", help="use scheduler")
    parser.add_argument('--aug', action='store_true', help='data augmentation')
    parser.add_argument('--save_arch', type=str, help="save arch")
    parser.add_argument('--file_name', type=str, default=os.path.basename(__file__).split('.')[0], help='file name')

    parser.add_argument('--cudnn-enabled', default=True, help='Enable cudnn')
    parser.add_argument('--cudnn-benchmark', default=True, help='Run cudnn benchmark')

    return parser


def predict(model, conf, logger):

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = conf.cudnn_enabled
    torch.backends.cudnn.benchmark = conf.cudnn_benchmark
    # init progress
    progress = Progress(
        TextColumn("[bold blue]{task.description}", justify="right"),
        MofNCompleteColumn(),
        BarColumn(bar_width=40),
        "[progress.percentage]{task.percentage:>3.1f}%",
        TimeRemainingColumn(),
    )

    # * load model
    assert type(conf.ckpt) == str, "You must specify the checkpoint path"
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[conf.rank], output_device=conf.rank)
    logger.info(f"load model from:{os.path.join(conf.ckpt, conf.latest_checkpoint_file)}")
    ckpt = torch.load(os.path.join(conf.ckpt, conf.latest_checkpoint_file), map_location=lambda storage, loc: storage)
    model.load_state_dict(ckpt['model'])
    model.eval()

    # * load datasetBs
    from dataloader import Dataset
    dataset = Dataset(conf).subjects  # ! notice in predict.py should use Dataset(conf).subjects
    znorm = ZNormalization()

    jaccard_ls, dice_ls = [], []

    file_tqdm = progress.add_task("[red]Predicting file", total=len(dataset))
    batch_tqdm = progress.add_task("[blue]file batch", total=len(dataset))

    for i, item in enumerate(dataset):
        progress.start()
        item = znorm(item)
        grid_sampler = tio.inference.GridSampler(item, patch_size=(conf.patch_size), patch_overlap=(4, 4, 36))
        affine = item['source']['affine']
        # * dist sampler
        # dist_sampler = torch.utils.data.distributed.DistributedSampler(grid_sampler, shuffle=True)

        # assert conf.batch_size == 1, 'batch_size must be 1 for inference'

        patch_loader = torch.utils.data.DataLoader(grid_sampler, batch_size=conf.batch_size, shuffle=False, num_workers=0, pin_memory=True)
        #    sampler=dist_sampler)

        pred_aggregator = tio.inference.GridAggregator(grid_sampler)
        gt_aggregator = tio.inference.GridAggregator(grid_sampler)
        with torch.no_grad():
            for batch in progress.track(patch_loader, total=len(patch_loader), description='Predicting file {}'.format(i)):
                progress.start()
                locations = batch[tio.LOCATION]

                x = process_x(conf, batch)
                x = x.type(torch.FloatTensor).cuda()
                gt = process_gt(conf, batch)
                gt = gt.type(torch.FloatTensor).cuda()

                pred = model(x)

                mask = torch.sigmoid(pred.clone())
                mask[mask > 0.5] = 1
                mask[mask <= 0.5] = 0

                pred_aggregator.add_batch(mask, locations)
                gt_aggregator.add_batch(gt, locations)
                progress.advance(batch_tqdm, advance=1)
            # reset batchtqdm
            progress.advance(file_tqdm, advance=1)
            progress.reset(batch_tqdm)
            pred_t = pred_aggregator.get_output_tensor()
            gt_t = gt_aggregator.get_output_tensor()

            # * save pred mhd file
            save_mhd(pred_t, affine, i)

            # * calculate metrics
            jaccard, dice = metric(gt_t, pred_t)
            jaccard_ls.append(jaccard)
            dice_ls.append(dice)
            logger.info(f'\njaccard:{jaccard}'
                        f'\ndice:{dice}')
    save_csv(jaccard_ls, dice_ls)
    jaccard_mean = np.mean(jaccard_ls)
    dice_mean = np.mean(dice_ls)
    # print('-' * 40)
    logger.info(f'\njaccard_mean:{jaccard_mean}'
                f'\ndice_mean:{dice_mean}')


def save_csv(jaccard_ls, dice_ls):
    import pandas as pd
    data = {'jaccard': jaccard_ls, 'dice': dice_ls}
    df = pd.DataFrame(data)
    df.loc[len(df)] = [df.iloc[:, 0].mean(), df.iloc[:, 1].mean()]
    save_path = os.path.join(conf.output_dir, 'metrics.csv')
    df.to_csv(save_path, index=False)


def save_mhd(pred, affine, index):
    save_base = os.path.join(conf.output_dir, 'pred_file')
    os.makedirs(save_base, exist_ok=True)
    pred_data = tio.ScalarImage(tensor=pred, affine=affine)
    pred_data.save(os.path.join(save_base, f'pred-{index:04d}.mhd'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Medical Segmentation Training')
    parser = parse_training_args(parser)
    args, _ = parser.parse_known_args()
    args = parser.parse_args()
    args_dict = vars(args)

    conf_path = args.conf_path
    conf = Default_Conf()
    conf.update(yaml_read(conf_path))
    conf.update_from_args(args_dict)

    os.makedirs(conf.output_dir, exist_ok=True)
    os.environ['CUDA_VISIBLE_DEVICES'] = conf.gpus

    # #* distributed training
    dist.init_process_group(backend='nccl')
    conf.rank = dist.get_rank()
    torch.cuda.set_device(conf.rank)
    device = torch.device('cuda', conf.rank)

    # * model selection
    if conf.network == 'res_unet':
        from models.three_d.residual_unet3d import UNet
        model = UNet(in_channels=conf.in_classes, n_classes=conf.out_classes, base_n_filter=32)
    elif conf.network == 'unet':
        from models.three_d.unet3d import UNet3D  # * 3d unet
        model = UNet3D(in_channels=conf.in_classes, out_channels=conf.out_classes, init_features=32)
    elif conf.network == 'er_net':
        from models.three_d.ER_net import ER_Net
        model = ER_Net(classes=conf.out_classes, channels=conf.in_classes)

    model.to(device)
    # * create logger
    logger = create_logger(output_dir=conf.output_dir, dist_rank=0 if torch.cuda.device_count() == 1 else dist.get_rank(), name='predict')
    info = '\nParameter Settings:\n'
    for k, v in conf.items():
        info += f"{k}: {v}\n"
    logger.info(info)

    predict(model, conf, logger)
    logger.info(f'tensorboard file saved in:{conf.output_dir}')
