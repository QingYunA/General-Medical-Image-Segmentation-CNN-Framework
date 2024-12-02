import os
import argparse
import torch
import time
from torch.utils.tensorboard import SummaryWriter
import torch.distributed as dist
from tqdm import tqdm
from utils.metric import metric
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR, CosineAnnealingLR
from scipy.signal import filtfilt, butter

# from logger import create_logger
from timm.utils import AverageMeter
from accelerate import Accelerator

import torchio as tio

# from utils import yaml_read
# from utils.conf_base import Default_Conf
from rich.progress import (
    BarColumn,
    Progress,
    TextColumn,
    MofNCompleteColumn,
    TimeRemainingColumn,
)
import logging
from rich.logging import RichHandler
import hydra

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"  # ! solve warning


def weights_init_normal(init_type):
    def init_func(m):
        classname = m.__class__.__name__
        gain = 0.02

        if classname.find("BatchNorm2d") != -1:
            if hasattr(m, "weight") and m.weight is not None:
                torch.nn.init.normal_(m.weight.data, 1.0, gain)
            if hasattr(m, "bias") and m.bias is not None:
                torch.nn.init.constant_(m.bias.data, 0.0)
        elif hasattr(m, "weight") and (classname.find("Conv") != -1 or classname.find("Linear") != -1):
            if init_type == "normal":
                torch.nn.init.normal_(m.weight.data, 0.0, gain)
            elif init_type == "xavier":
                torch.nn.init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == "xavier_uniform":
                torch.nn.init.xavier_uniform_(m.weight.data, gain=1.0)
            elif init_type == "kaiming":
                torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode="fan_in")
            elif init_type == "orthogonal":
                torch.nn.init.orthogonal_(m.weight.data, gain=gain)
            elif init_type == "none":  # uses pytorch's default init method
                m.reset_parameters()
            else:
                raise NotImplementedError("initialization method [%s] is not implemented" % init_type)
            if hasattr(m, "bias") and m.bias is not None:
                torch.nn.init.constant_(m.bias.data, 0.0)

    return init_func


def get_logger(config):
    file_handler = logging.FileHandler(os.path.join(config.hydra_path, f"{config.job_name}.log"))
    rich_handler = RichHandler()

    log = logging.getLogger(__name__)
    log.setLevel(logging.DEBUG)
    log.addHandler(rich_handler)
    log.addHandler(file_handler)
    log.propagate = False
    log.info("Successfully create rich logger")

    return log


def train(config, model, logger):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = config.cudnn_enabled
    torch.backends.cudnn.benchmark = config.cudnn_benchmark

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
    optimizer = torch.optim.Adam(model.parameters(), lr=config.init_lr)

    # * set loss function
    from utils.loss_function import Binary_Loss, DiceLoss, cross_entropy_3D

    criterion = Binary_Loss()
    # dice_criterion = DiceLoss().cuda()

    # * set scheduler strategy
    if config.use_scheduler:
        scheduler = StepLR(optimizer, step_size=config.scheduler_step_size, gamma=config.scheduler_gamma)

    # * load model
    if config.load_mode == 1:  # * load weights from checkpoint
        logger.info(f"load model from: {os.path.join(config.ckpt, config.latest_checkpoint_file)}")
        ckpt = torch.load(
            os.path.join(config.ckpt, config.latest_checkpoint_file), map_location=lambda storage, loc: storage
        )
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optim"])
        for state in optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.cuda()

        if config.use_scheduler:
            scheduler.load_state_dict(ckpt["scheduler"])
        elapsed_epochs = ckpt["epoch"]
        # elapsed_epochs = 0
    else:
        elapsed_epochs = 0

    model.train()

    # * tensorboard writer
    writer = SummaryWriter(config.hydra_path)

    # * load datasetBs
    from dataloader import Dataset

    train_dataset = Dataset(config)
    #! in distributed training, the 'shuffle' must be false!
    # train_loader = torch.utils.data.DataLoader(
    #     dataset=train_dataset.queue_dataset,
    #     batch_size=config.batch_size,
    #     shuffle=False,
    #     num_workers=0,
    #     pin_memory=True,
    #     drop_last=True,
    # )
    train_loader = tio.SubjectsLoader(
        dataset = train_dataset.queue_dataset, 
        batch_size = config.batch_size,
        shuffle = False
    )

    epochs = config.epochs - elapsed_epochs
    iteration = elapsed_epochs * len(train_loader)

    epoch_tqdm = progress.add_task(description="[red]epoch progress", total=epochs)
    batch_tqdm = progress.add_task(description="[blue]batch progress", total=len(train_loader))

    accelerator = Accelerator()
    # * accelerate prepare
    train_loader, model, optimizer, scheduler = accelerator.prepare(train_loader, model, optimizer, scheduler)

    if (config.network == "IS"):
        from utils.Filter import low_pass_filter, high_pass_filter
        low_cutoff = 0.1
        high_cutoff = -0.1
        order = 4


    progress.start()
    for epoch in range(1, epochs + 1):
        progress.update(epoch_tqdm, completed=epoch)
        epoch += elapsed_epochs

        num_iters = 0

        load_meter = AverageMeter()
        train_time = AverageMeter()
        load_start = time.time()  # * initialize

        for i, batch in enumerate(train_loader):
            with torch.autograd.set_detect_anomaly(True):
                progress.update(batch_tqdm, completed=i + 1)
                train_start = time.time()
                load_time = time.time() - load_start
                optimizer.zero_grad()

                # x = batch["source"]["data"]  # * from batch extract x:[bs,4 or 1,h,w,d]
                # gt = batch["gt"]["data"]  # * from batch extract gt:[bs,4 or 1,h,w,d]
                x = batch["source"][tio.DATA]
                gt = batch["gt"][tio.DATA]  
                # print("x_shape, gt_shape=",x.shape, gt.shape)

                gt_back = torch.zeros_like(gt)
                gt_back[gt == 0] = 1
                gt = torch.cat([gt_back, gt], dim=1)  # * [bs,2,h,w,d]

                x = x.type(torch.FloatTensor).to(accelerator.device)
                gt = gt.type(torch.FloatTensor).to(accelerator.device)

                if (config.network == "IS"):
                    x_ = x.cpu().detach().numpy()
                    low_x = torch.tensor(low_pass_filter(x_, low_cutoff, order))
                    high_x = torch.tensor(high_pass_filter(x_, high_cutoff, order))


                    low_x = low_x.type(torch.FloatTensor).to(accelerator.device)
                    high_x = high_x.type(torch.FloatTensor).to(accelerator.device)

                    _, pred = model(x, low_x, high_x)
                else:
                    pred = model(x)

                mask = pred.argmax(dim=1, keepdim=True)  # * [bs,1,h,w,d]

                # *  pred -> mask (0 or 1)
                # mask[mask > 0.5] = 1
                # mask[mask <= 0.5] = 0

                loss = criterion(pred, gt)
                # loss.backward()
                accelerator.backward(loss)
                progress.refresh()

            optimizer.step()

            num_iters += 1
            iteration += 1

            # * calculate metrics
            # TODO use reduce to sum up all rank's calculation results
            _, dice = metric(gt.cpu().argmax(dim=1, keepdim=True), mask.cpu())
            # dice = dist.all_reduce(dice, op=dist.ReduceOp.SUM) / dist.get_world_size()
            # recall = dist.all_reduce(recall, op=dist.ReduceOp.SUM) / dist.get_world_size()
            # specificity = dist.all_reduce(specificity, op=dist.ReduceOp.SUM) / dist.get_world_size()

            writer.add_scalar("Training/Loss", loss.item(), iteration)
            # writer.add_scalar('Training/recall', recall, iteration)
            # writer.add_scalar('Training/specificity', specificity, iteration)
            writer.add_scalar("Training/dice", dice, iteration)

            temp_file_base = os.path.join(config.hydra_path, "train_temp")
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
            logger.info(
                f"\nEpoch: {epoch} Batch: {i}, data load time: {load_meter.val:.3f}s , train time: {train_time.val:.3f}s\n"
                f"Loss: {loss_meter.val}\n"
                f"Dice: {dice_meter.val}\n"
            )
            # f'Recall: {recall_meter.val}\n'
            # f'Specificity: {spe_meter.val}\n')

            load_start = time.time()
        # reset batchtqdm

        if config.use_scheduler:
            scheduler.step()
            logger.info(f"Learning rate:  {scheduler.get_last_lr()[0]}")

        # * one epoch logger
        logger.info(
            f"\nEpoch {epoch} used time:  {load_meter.sum+train_time.sum:.3f} s\n"
            f"Loss Avg:  {loss_meter.avg}\n"
            f"Dice Avg:  {dice_meter.avg}\n"
        )
        # f'Recall Avg: {recall_meter.avg}\n'
        # f'Specificity Avg: {spe_meter.avg}\n')

        # Store latest checkpoint in each epoch
        scheduler_dict = scheduler.state_dict() if config.use_scheduler else None
        torch.save(
            {
                "model": model.state_dict(),
                "optim": optimizer.state_dict(),
                "scheduler": scheduler_dict,
                "epoch": epoch,
            },
            os.path.join(config.hydra_path, config.latest_checkpoint_file),
        )

        # Save checkpoint
        if epoch % config.epochs_per_checkpoint == 0:
            torch.save(
                {
                    "model": model.state_dict(),
                    "optim": optimizer.state_dict(),
                    "scheduler": scheduler_dict,
                    "epoch": epoch,
                },
                os.path.join(config.hydra_path, f"checkpoint_{epoch:04d}.pt"),
            )
    writer.close()


@hydra.main(config_path="conf", config_name="config", version_base="1.3")
def main(config):
    config = config["config"]
    os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu
    if isinstance(config.patch_size, str):
        assert (
            len(config.patch_size.split(",")) <= 3
        ), f'patch size can only be one str or three str but got {len(config.patch_size.split(","))}'
        if len(config.patch_size.split(",")) == 3:
            config.patch_size = tuple(map(int, config.patch_size.split(",")))
        else:
            config.patch_size = int(config.patch_size)

    # * model selection
    if config.network == "res_unet":
        from models.three_d.residual_unet3d import UNet

        model = UNet(in_channels=config.in_classes, n_classes=config.out_classes, base_n_filter=32)
    elif config.network == "unet":
        from models.three_d.unet3d import UNet3D  # * 3d unet

        model = UNet3D(in_channels=config.in_classes, out_channels=config.out_classes, init_features=32)
    elif config.network == "er_net":
        from models.three_d.ER_net import ER_Net

        model = ER_Net(classes=config.out_classes, channels=config.in_classes)
    elif config.network == "re_net":
        from models.three_d.RE_net import RE_Net

        model = RE_Net(classes=config.out_classes, channels=config.in_classes)
    elif config.network == "IS":
        from models.three_d.IS import UNet3D

        model = UNet3D(in_channels=config.in_classes, out_channels=config.out_classes)

    elif config.network == "unetr":
        from models.three_d.unetr import UNETR

        model = UNETR()
    elif config.network == "densenet":
        from models.three_d.densenet3d import SkipDenseNet3D

        model = SkipDenseNet3D(in_channels=config.in_classes, classes=config.out_classes)

    elif config.network == "vtnet":
        from models.three_d.vtnet import VTUNet

        model = VTUNet(num_classes=config.out_classes, input_dim=config.in_classes)
    elif config.network == "Dunet_noSE":
        from models.three_d.Double_Unet_noSE import Double_Unet

        model = Double_Unet()
    elif config.network == "Dunet_noCrop":
        from models.three_d.Double_Unet_noCrop import Double_Unet

        model = Double_Unet()
    model.apply(weights_init_normal(config.init_type))

    # * create logger
    logger = get_logger(config)
    info = "\nParameter Settings:\n"
    for k, v in config.items():
        info += f"{k}: {v}\n"
    logger.info(info)

    train(config, model, logger)
    logger.info(f"tensorboard file saved in:{config.hydra_path}")


if __name__ == "__main__":
    main()
