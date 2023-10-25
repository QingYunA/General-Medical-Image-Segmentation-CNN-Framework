import os
import argparse
import torch
import time
from torch.utils.tensorboard import SummaryWriter
import torch.distributed as dist
from tqdm import tqdm
from utils.metric import metric
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR, CosineAnnealingLR
from process_input import process_x, process_gt
from logger import create_logger
from timm.utils import AverageMeter
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

BLUE = '\033[94m'
YELLOW = '\033[93m'
END = '\033[0m'


def weights_init_normal(init_type):

    def init_func(m):
        classname = m.__class__.__name__
        gain = 0.02
        # init_type = conf.init_type

        if classname.find('BatchNorm2d') != -1:
            if hasattr(m, 'weight') and m.weight is not None:
                torch.nn.init.normal_(m.weight.data, 1.0, gain)
            if hasattr(m, 'bias') and m.bias is not None:
                torch.nn.init.constant_(m.bias.data, 0.0)
        elif hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                torch.nn.init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                torch.nn.init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'xavier_uniform':
                torch.nn.init.xavier_uniform_(m.weight.data, gain=1.0)
            elif init_type == 'kaiming':
                torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                torch.nn.init.orthogonal_(m.weight.data, gain=gain)
            elif init_type == 'none':  # uses pytorch's default init method
                m.reset_parameters()
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                torch.nn.init.constant_(m.bias.data, 0.0)

    return init_func


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


def train(model, args, logger):

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = conf.cudnn_enabled
    torch.backends.cudnn.benchmark = conf.cudnn_benchmark

    # * init averageMeter
    loss_meter = AverageMeter()
    dice_meter = AverageMeter()

    # init rich progress
    progress = Progress(
        TextColumn("[bold blue]{task.description}", justify="right"),
        MofNCompleteColumn(),
        BarColumn(bar_width=40),
        "[progress.percentage]{task.percentage:>3.1f}%",
        TimeRemainingColumn(),
    )

    # * set optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=conf.init_lr)

    # * set loss function
    from loss_function import Binary_Loss, DiceLoss, cross_entropy_3D
    criterion = Binary_Loss().cuda()
    # dice_criterion = DiceLoss().cuda()

    # * set scheduler strategy
    if conf.use_scheduler:
        scheduler = StepLR(optimizer, step_size=conf.scheduler_step_size, gamma=conf.scheduler_gamma)

    # * load model
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[conf.rank], output_device=conf.rank)
    if conf.load_mode == 1:  # * load weights from checkpoint
        logger.info(f"load model from: {os.path.join(conf.ckpt, conf.latest_checkpoint_file)}")
        ckpt = torch.load(os.path.join(conf.ckpt, conf.latest_checkpoint_file), map_location=lambda storage, loc: storage)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optim'])
        for state in optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.cuda()

        if conf.use_scheduler:
            scheduler.load_state_dict(ckpt["scheduler"])
        elapsed_epochs = ckpt["epoch"]
        # elapsed_epochs = 0
    else:
        elapsed_epochs = 0

    model.train()

    # * tensorboard writer
    writer = SummaryWriter(conf.output_dir)

    # * load datasetBs
    from dataloader import Dataset
    train_dataset = Dataset(conf)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset.queue_dataset, shuffle=True)
    #! in distributed training, the 'shuffle' must be false!
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset.queue_dataset,
                                               batch_size=conf.batch_size,
                                               shuffle=False,
                                               num_workers=0,
                                               pin_memory=True,
                                               drop_last=True,
                                               sampler=train_sampler)

    epochs = conf.epochs - elapsed_epochs
    iteration = elapsed_epochs * len(train_loader)

    epoch_tqdm = progress.add_task(description='[red]epoch progress', total=epochs)
    batch_tqdm = progress.add_task(description='[green]batch progress', total=len(train_loader))

    for epoch in range(1, epochs + 1):
        progress.start()
        epoch += elapsed_epochs

        num_iters = 0

        load_meter = AverageMeter()
        train_time = AverageMeter()
        load_start = time.time()  # * initialize

        train_loader.sampler.set_epoch(epoch)  # ! must set epoch for DistributedSampler!
        for i, batch in enumerate(train_loader):
            with torch.autograd.set_detect_anomaly(True):
                train_start = time.time()
                load_time = time.time() - load_start
                optimizer.zero_grad()

                x = process_x(conf, batch)  # * from batch extract x:[bs,4 or 1,h,w,d]
                gt = process_gt(conf, batch)  # * from batch extract gt:[bs,4 or 1,h,w,d]

                x = x.type(torch.FloatTensor).cuda()
                gt = gt.type(torch.FloatTensor).cuda()

                pred = model(x)

                # *  pred -> mask (0 or 1)
                mask = torch.sigmoid(pred.clone())  # TODO should use softmax, because it returns two probability (sum = 1)
                mask[mask > 0.5] = 1
                mask[mask <= 0.5] = 0

                loss = criterion(pred, gt)
                loss.backward()

            optimizer.step()

            num_iters += 1
            iteration += 1
            progress.advance(batch_tqdm, advance=1)

            # * calculate metrics
            # TODO use reduce to sum up all rank's calculation results
            _, dice = metric(gt.cpu(), mask.cpu())
            # dice = dist.all_reduce(dice, op=dist.ReduceOp.SUM) / dist.get_world_size()
            # recall = dist.all_reduce(recall, op=dist.ReduceOp.SUM) / dist.get_world_size()
            # specificity = dist.all_reduce(specificity, op=dist.ReduceOp.SUM) / dist.get_world_size()

            writer.add_scalar('Training/Loss', loss.item(), iteration)
            # writer.add_scalar('Training/recall', recall, iteration)
            # writer.add_scalar('Training/specificity', specificity, iteration)
            writer.add_scalar('Training/dice', dice, iteration)

            # print('lr:' + str(scheduler._last_lr[0]))

            temp_file_base = os.path.join(conf.output_dir, 'train_temp')
            os.makedirs(temp_file_base, exist_ok=True)
            # if (i % 20 == 0):
            #     with torch.no_grad():
            #         #! if dataset is brats ,it will automatically save flair modality as nii.gz
            #         if (conf.dataset == 'brats'):
            #             affine = batch['flair']['affine'][0].numpy()
            #             flair_source = tio.ScalarImage(tensor=x[:, 0, :, :, :].cpu().detach().numpy(), affine=affine)
            #             flair_source.save(os.path.join(temp_file_base, f"epoch-{epoch:04d}-batch-{i:02d}-source" + conf.save_arch))
            #             flair_gt = tio.ScalarImage(tensor=gt[:, 0, :, :, :].cpu().detach().numpy(), affine=affine)
            #             flair_gt.save(os.path.join(temp_file_base, f"epoch-{epoch:04d}-batch-{i:02d}-gt" + conf.save_arch))
            #             flair_pred = tio.ScalarImage(tensor=pred[:, 0, :, :, :].cpu().detach().numpy(), affine=affine)
            #             flair_pred.save(os.path.join(temp_file_base, f"epoch-{epoch:04d}-batch-{i:02d}-pred" + conf.save_arch))
            #         else:
            #             affine = batch['source']['affine'][0].numpy()
            #             source = tio.ScalarImage(tensor=x[0, :, :, :, :].cpu().detach().numpy(), affine=affine)
            #             source.save(os.path.join(temp_file_base, f"epoch-{epoch:04d}-batch-{i:02d}-source" + conf.save_arch))
            #             gt_data = tio.ScalarImage(tensor=gt[0, :, :, :, :].cpu().detach().numpy(), affine=affine)
            #             gt_data.save(os.path.join(temp_file_base, f"epoch-{epoch:04d}-batch-{i:02d}-gt" + conf.save_arch))
            #             pred_data = tio.ScalarImage(tensor=pred[0, :, :, :, :].cpu().detach().numpy(), affine=affine)
            #             pred_data.save(os.path.join(temp_file_base, f"epoch-{epoch:04d}-batch-{i:02d}-pred" + conf.save_arch))
            # * record metris
            loss_meter.update(loss.item(), x.size(0))
            dice_meter.update(dice, x.size(0))
            # recall_meter.update(recall, x.size(0))
            # spe_meter.update(specificity, x.size(0))
            train_time.update(time.time() - train_start)
            load_meter.update(load_time)
            # logger.info('batch used time: {:.3f} s\n'.format(batch_time.val))
            logger.info(f'\nEpoch: {epoch} Batch: {i}, data load time: {load_meter.val:.3f}s , train time: {train_time.val:.3f}s\n'
                        f'Loss: {loss_meter.val}\n'
                        f'Dice: {dice_meter.val}\n')
            # f'{BLUE}Recall:{END} {recall_meter.val}\n'
            # f'{BLUE}Specificity:{END} {spe_meter.val}\n')

            load_start = time.time()
        # reset batchtqdm
        progress.advance(epoch_tqdm, advance=1)
        progress.reset(batch_tqdm)

        if (conf.use_scheduler):
            scheduler.step()
            logger.info(f'{BLUE}Learning rate:{END} {scheduler.get_last_lr()[0]}')

        # * one epoch logger
        logger.info(f'\n{BLUE}Epoch {epoch} used time:{END} {load_meter.sum+train_time.sum:.3f} s\n'
                    f'{BLUE}Loss Avg:{END} {loss_meter.avg}\n'
                    f'{BLUE}Dice Avg:{END} {dice_meter.avg}\n')
        # f'{BLUE}Recall Avg:{END} {recall_meter.avg}\n'
        # f'{BLUE}Specificity Avg:{END} {spe_meter.avg}\n')

        # Store latest checkpoint in each epoch
        scheduler_dict = scheduler.state_dict() if conf.use_scheduler else None
        torch.save(
            {
                "model": model.state_dict(),
                "optim": optimizer.state_dict(),
                "scheduler": scheduler_dict,
                "epoch": epoch,
            },
            os.path.join(conf.output_dir, conf.latest_checkpoint_file),
        )

        # Save checkpoint
        if epoch % conf.epochs_per_checkpoint == 0:

            torch.save(
                {
                    "model": model.state_dict(),
                    "optim": optimizer.state_dict(),
                    "scheduler": scheduler_dict,
                    "epoch": epoch,
                },
                os.path.join(conf.output_dir, f"checkpoint_{epoch:04d}.pt"),
            )
    writer.close()


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
    # print(conf)

    if type(conf.patch_size) == str:
        assert len(conf.patch_size.split(',')) <= 3, f'patch size can only be one str or three str but got {len(conf.patch_size.split(","))}'
        if len(conf.patch_size.split(',')) == 3:
            conf.patch_size = tuple(map(int, conf.patch_size.split(',')))
        else:
            conf.patch_size = int(conf.patch_size)

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
    elif conf.network == 're_net':
        from models.three_d.RE_net import RE_Net
        model = RE_Net(classes=conf.out_classes, channels=conf.in_classes)

    model.apply(weights_init_normal(conf.init_type))
    model = model.to(device)

    # * create logger
    logger = create_logger(output_dir=conf.output_dir, dist_rank=0 if torch.cuda.device_count() == 1 else dist.get_rank(), name='train')
    info = '\nParameter Settings:\n'
    for k, v in conf.items():
        info += f"{BLUE}{k}{END}: {v}\n"
    logger.info(info)

    train(model, conf, logger)
    logger.info(f'tensorboard file saved in:{conf.output_dir}')
