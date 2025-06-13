import lightning as L
import torch
import torch.nn as nn
import torch.optim as optim
from thop import profile
import time
import torch.cuda as cuda
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from monai.losses import DiceLoss, DiceCELoss
from monai.data.utils import affine_to_spacing
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric, HausdorffDistanceMetric
from lightning.pytorch.utilities import measure_flops
from torchmetrics import Precision, Recall
import pandas as pd
import numpy as np
import os
import math


# def kaiming_init(model):
#     for name, param in model.named_parameters():
#         if name.endswith(".bias"):
#             param.data.fill_(0)
#         elif name.startswith(
#             "layers.0"
#         ):  # The first layer does not have ReLU applied on its input
#             param.data.normal_(0, 1 / math.sqrt(param.shape[1]))
#         else:
#             param.data.normal_(0, math.sqrt(2) / math.sqrt(param.shape[1]))


def kaiming_init(module):
    for name, param in module.named_parameters():
        if "weight" in name:
            if len(param.shape) > 1:  # 对于权重参数
                # 使用PyTorch内置的kaiming_normal_初始化
                torch.nn.init.kaiming_normal_(param, mode="fan_in", nonlinearity="relu")
            else:  # 对于1维参数（如bias）
                torch.nn.init.constant_(param, 0)
        elif "bias" in name:
            torch.nn.init.constant_(param, 0)


class BaseModel(L.LightningModule):
    def __init__(
        self,
        model=None,
        patch_size=(64, 64, 64),
        sw_batch_size=1,
        lr=0.001,
        save_path=None,
        show_model_stats=False,
        init_type=None,
        use_scheduler=True,
        lr_step_size=20,
        lr_gamma=0.8,
        *args,
        **kwargs,
    ):
        super(BaseModel, self).__init__()
        self.model = model
        if init_type == "kaiming":
            kaiming_init(self.model)
            print("Kaiming initialization done")
        if show_model_stats:
            self.cal_model_stats()

        self.lr = lr
        self.patch_size = patch_size
        self.sw_batch_size = sw_batch_size
        self.save_path = save_path
        self.use_scheduler = use_scheduler
        self.lr_step_size = lr_step_size
        self.lr_gamma = lr_gamma
        # self.dice_loss = DiceLoss(include_background=True, to_onehot_y=True, softmax=True)
        self.dice_ce_loss = DiceCELoss(
            include_background=True, to_onehot_y=True, softmax=True
        )
        # self.bce_loss = nn.BCEWithLogitsLoss()

        self.dice_metric = DiceMetric(include_background=True, reduction="mean")
        self.hs_distance = HausdorffDistanceMetric(
            include_background=True, percentile=95
        )
        self.precision = Precision(average="macro", num_classes=2, task="binary")
        self.recall = Recall(average="macro", num_classes=2, task="binary")

        self.metrics_l = []

    def init_from_ckpt(self, path, ignore_keys=list(), only_model=False):
        sd = torch.load(path, map_location="cpu")
        if "state_dict" in list(sd.keys()):
            sd = sd["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        missing, unexpected = (
            self.load_state_dict(sd, strict=False)
            if not only_model
            else self.model.load_state_dict(sd, strict=False)
        )
        print(
            f"Restored from {path} with {len(missing)} missing and {len(unexpected)} unexpected keys"
        )
        if len(missing) > 0:
            print(f"Missing Keys: {missing}")
        if len(unexpected) > 0:
            print(f"Unexpected Keys: {unexpected}")

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch["image"], batch["label"]
        pred = self(x)
        dice_ce_loss = self.dice_ce_loss(pred, y)
        # bce_loss = self.bce_loss(y_hat, y)
        # loss = dice_loss + bce_loss
        loss = dice_ce_loss
        self.log("train/loss", loss.item(), on_step=True, on_epoch=False, prog_bar=True)
        mask = torch.argmax(pred, dim=1, keepdim=True)
        metrics_dict = self.metrics(mask, y)
        self.log("train/dice", metrics_dict["dice"], prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        # lr_scheduler
        if self.use_scheduler:
            lr_scheduler = optim.lr_scheduler.StepLR(
                optimizer, step_size=self.lr_step_size, gamma=self.lr_gamma
            )
            return [optimizer], [{"scheduler": lr_scheduler, "interval": "epoch"}]
        else:
            return optimizer

    def validation_step(self, batch, batch_idx):
        pass

    def test_step(self, batch, batch_idx):
        x, y = batch["image"], batch["label"]
        patch_size = self.patch_size
        sw_batch_size = self.sw_batch_size

        outputs = sliding_window_inference(
            x, patch_size, sw_batch_size, self.forward, progress=True
        )
        spacing = affine_to_spacing(x.meta["affine"].squeeze()).tolist()

        mask = torch.argmax(outputs, dim=1, keepdim=True)
        metrics_dict = self.metrics(mask, y, spacing)
        self.metrics_l.append(metrics_dict)

    def on_test_epoch_end(self):
        df = pd.DataFrame(self.metrics_l)
        mean_dice = df["dice"].mean().item()
        dice_std = df["dice"].std().item()
        mean_hs_distance = df["hs_distance"].mean().item()
        hs_distance_std = df["hs_distance"].std().item()
        mean_precision = df["precision"].mean().item()
        precision_std = df["precision"].std().item()
        mean_recall = df["recall"].mean().item()
        recall_std = df["recall"].std().item()
        mean_df = pd.DataFrame(
            {
                "dice": f"{mean_dice:.4f}±{dice_std:.4f}",
                "hs_distance": f"{mean_hs_distance:.4f}±{hs_distance_std:.4f}",
                "precision": f"{mean_precision:.4f}±{precision_std:.4f}",
                "recall": f"{mean_recall:.4f}±{recall_std:.4f}",
            },
            index=["mean+std"],
        )
        print("=" * 50)
        print(f"Dice:     {mean_dice:.4f}±{dice_std:.4f}")
        print(f"Hausdorff95: {mean_hs_distance:.4f}±{hs_distance_std:.4f}")
        print(f"Precision:       {mean_precision:.4f}±{precision_std:.4f}")
        print(f"Recall:       {mean_recall:.4f}±{recall_std:.4f}")
        print("=" * 50)
        empty_df = pd.DataFrame({col: [np.nan] for col in df.columns})
        df = pd.concat([df, empty_df, mean_df], ignore_index=True)
        df.to_csv(os.path.join(self.save_path, "metrics.csv"), index=False)

    def cal_model_stats(self):
        """计算模型的FLOPs、参数量、推理时间等统计信息"""

        x = torch.randn(1, 1, 64, 64, 64)
        model_fwd = lambda: self.model(x)
        flops = measure_flops(self.model, model_fwd)
        # flops, params = profile(self.model, inputs=(x,))
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )

        # 转换为更易读的格式
        def format_num(num):
            if num >= 1e9:
                return f"{num / 1e9:.2f}B"
            elif num >= 1e6:
                return f"{num / 1e6:.2f}M"
            elif num >= 1e3:
                return f"{num / 1e3:.2f}K"
            return str(num)

        starter = cuda.Event(enable_timing=True)
        ender = cuda.Event(enable_timing=True)

        for _ in range(5):
            _ = self.model(x)

        times = []
        for _ in range(10):
            starter.record()
            _ = self.model(x)
            ender.record()
            cuda.synchronize()
            curr_time = starter.elapsed_time(ender)
            times.append(curr_time)

        mean_time = np.mean(times)
        std_time = np.std(times)

        # 计算模型大小(MB)
        param_size = 0
        for param in self.model.parameters():
            param_size += param.nelement() * param.element_size()
        buffer_size = 0
        for buffer in self.model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()

        size_all_mb = (param_size + buffer_size) / 1024**2

        print("=" * 50)
        print("模型统计信息(10次平均):")
        print(f"FLOPs: {flops / 1e9:.2f}G")
        print(f"参数量: {format_num(total_params)}")
        print(f"可训练参数量: {format_num(trainable_params)}")
        print(f"模型大小: {size_all_mb:.2f}MB")
        print(f"平均推理时间: {mean_time:.2f}±{std_time:.2f}ms")
        print("=" * 50)

    def metrics(self, pred, y, spacing=None):
        dice_score = self.dice_metric(pred, y).mean()
        hs_distance = self.hs_distance._compute_tensor(pred, y, spacing=spacing).mean()
        precision = self.precision(pred, y).mean()
        recall = self.recall(pred, y).mean()
        metrics_dict = {
            "dice": dice_score.item(),
            "hs_distance": hs_distance.item(),
            "precision": precision.item(),
            "recall": recall.item(),
        }
        return metrics_dict


class TriPlaneModel(BaseModel):
    def __init__(self, lr=0.001, model=None, upper_0=False, *args, **kwargs):
        super(TriPlaneModel, self).__init__(lr=lr, model=model, *args, **kwargs)
        self.upper_0 = upper_0
        self.mse_loss = nn.MSELoss()
        self.THRESHOLD = 0.5

    def training_step(self, batch, batch_idx):
        x, y = batch["image"], batch["label"]
        pred, seg_pred, m_outputs, seg_m_outputs, decouple_loss = self(x)
        m_dec4, m_dec3, m_dec2, m_dec1 = m_outputs
        seg_dec4, seg_dec3, seg_dec2, seg_dec1 = seg_m_outputs

        if self.upper_0:
            seg_dec4 = seg_dec4 * (m_dec4 > 0)
            seg_dec3 = seg_dec3 * (m_dec3 > 0)
            seg_dec2 = seg_dec2 * (m_dec2 > 0)
            seg_dec1 = seg_dec1 * (m_dec1 > 0)

        seg_2d_loss = self.dice_ce_loss(pred, y)
        seg_3d_loss = self.dice_ce_loss(seg_pred, y)
        dec4_loss = self.mse_loss(m_dec4, seg_dec4)
        dec3_loss = self.mse_loss(m_dec3, seg_dec3)
        dec2_loss = self.mse_loss(m_dec2, seg_dec2)
        dec1_loss = self.mse_loss(m_dec1, seg_dec1)
        loss = (
            seg_2d_loss
            + seg_3d_loss
            + dec4_loss
            + dec3_loss
            + dec2_loss
            + dec1_loss
            + decouple_loss
        )
        self.log("train/loss", loss.item(), on_step=True, on_epoch=False, prog_bar=True)

        # mask = torch.argmax(pred, dim=1, keepdim=True)
        mask = torch.sigmoid(pred.clone())
        mask[mask > self.THRESHOLD] = 1
        mask[mask <= self.THRESHOLD] = 0
        metrics_dict = self.metrics(mask, y)
        self.log("train/dice", metrics_dict["dice"], prog_bar=True)
        return loss
