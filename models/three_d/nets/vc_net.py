"""
Author: Yifan Wang
Date created: 11/09/2020
PyTorch implementation of VC-Net
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class MipArg(nn.Module):
    def __init__(self):
        super(MipArg, self).__init__()

    def forward(self, x):
        # 使用reshape替代view，并确保张量维度正确
        x = x.reshape(-1, 32, 3, 128, 2, 128)
        x = x.permute(0, 1, 2, 4, 3, 5)
        x = x.reshape(-1, 32, 6, 128, 128)
        # Stack 5 times for 5-sliced MIP
        x = torch.stack([x, x, x, x, x], dim=-1)
        return x


class Unproject(nn.Module):
    def __init__(self):
        super(Unproject, self).__init__()

    def forward(self, x):
        # Split along the 4th dimension
        a = torch.unbind(x, dim=-4)

        # Pad each slice
        a0 = F.pad(a[0], (0, 11, 0, 0, 0, 0, 0, 0))
        a1 = F.pad(a[1], (2, 9, 0, 0, 0, 0, 0, 0))
        a2 = F.pad(a[2], (4, 7, 0, 0, 0, 0, 0, 0))
        a3 = F.pad(a[3], (6, 5, 0, 0, 0, 0, 0, 0))
        a4 = F.pad(a[4], (8, 3, 0, 0, 0, 0, 0, 0))
        a5 = F.pad(a[5], (11, 0, 0, 0, 0, 0, 0, 0))

        # Stack and get maximum
        a_concat = torch.stack([a0, a1, a2, a3, a4, a5], dim=-1)
        a_max = torch.max(a_concat, dim=-1)[0]

        return a_max


class ConvBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, batch_norm=False):
        super(ConvBlock3D, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size, padding=1)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size, padding=1)
        self.bn1 = nn.BatchNorm3d(out_channels) if batch_norm else nn.Identity()
        self.bn2 = nn.BatchNorm3d(out_channels) if batch_norm else nn.Identity()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        return x


class UNet3D(nn.Module):
    def __init__(self, in_channels=1, base_filters=32, depth=4, batch_norm=False):
        super(UNet3D, self).__init__()
        self.depth = depth
        self.down_path = nn.ModuleList()
        self.up_path = nn.ModuleList()

        # Downsampling path
        in_ch = in_channels
        for i in range(depth):
            out_ch = base_filters * (2**i)
            self.down_path.append(ConvBlock3D(in_ch, out_ch, batch_norm=batch_norm))
            in_ch = out_ch

        # Upsampling path
        for i in range(depth - 1):
            in_ch = base_filters * (2 ** (depth - 1 - i))
            out_ch = base_filters * (2 ** (depth - 2 - i))
            self.up_path.append(
                nn.Sequential(
                    nn.ConvTranspose3d(in_ch, out_ch, kernel_size=2, stride=2),
                    ConvBlock3D(out_ch * 2, out_ch, batch_norm=batch_norm),
                )
            )

    def forward(self, x):
        # Downsampling
        features = []
        for i, down in enumerate(self.down_path):
            x = down(x)
            if i < self.depth - 1:
                features.append(x)
                x = F.max_pool3d(x, 2)

        # Upsampling
        for i, up in enumerate(self.up_path):
            x = up[0](x)  # 先进行转置卷积
            x = torch.cat([x, features[-(i + 1)]], dim=1)  # 拼接特征
            x = up[1](x)  # 再进行卷积块处理

        return x


class ConvBlock2D(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size=3, dropout=0.0, batch_norm=False
    ):
        super(ConvBlock2D, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels) if batch_norm else nn.Identity()
        self.bn2 = nn.BatchNorm2d(out_channels) if batch_norm else nn.Identity()
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.dropout(x)
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.dropout(x)
        return x


class UNet2D(nn.Module):
    def __init__(
        self, in_channels=1, base_filters=32, depth=4, dropout=0.0, batch_norm=False
    ):
        super(UNet2D, self).__init__()
        self.depth = depth
        self.down_path = nn.ModuleList()
        self.up_path = nn.ModuleList()

        # Downsampling path
        in_ch = in_channels
        for i in range(depth):
            out_ch = base_filters * (2**i)
            self.down_path.append(
                ConvBlock2D(in_ch, out_ch, dropout=dropout, batch_norm=batch_norm)
            )
            in_ch = out_ch

        # Upsampling path
        for i in range(depth - 1):
            in_ch = base_filters * (2 ** (depth - 1 - i))
            out_ch = base_filters * (2 ** (depth - 2 - i))
            self.up_path.append(
                nn.Sequential(
                    nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2),
                    ConvBlock2D(
                        out_ch * 2, out_ch, dropout=dropout, batch_norm=batch_norm
                    ),
                )
            )

    def forward(self, x):
        # Downsampling
        features = []
        for i, down in enumerate(self.down_path):
            x = down(x)
            if i < self.depth - 1:
                features.append(x)
                x = F.max_pool2d(x, 2)

        # Upsampling
        for i, up in enumerate(self.up_path):
            x = up[0](x)  # 先进行转置卷积
            x = torch.cat([x, features[-(i + 1)]], dim=1)  # 拼接特征
            x = up[1](x)  # 再进行卷积块处理

        return x


class VCNet(nn.Module):
    def __init__(
        self,
        cube_size=(1, 128, 128, 16),
        patch_size=(384, 256),
        num_channels_2d=1,
        dropout_2d=0.0,
        batch_norm=False,
    ):
        super(VCNet, self).__init__()

        # 3D UNet
        self.unet3d = UNet3D(in_channels=cube_size[0], batch_norm=batch_norm)

        # 2D UNet
        self.unet2d = UNet2D(
            in_channels=num_channels_2d, dropout=dropout_2d, batch_norm=batch_norm
        )

        # MIP and Unprojection layers
        self.mip_arg = MipArg()
        self.unproject = Unproject()

        # Final fusion layers
        self.fusion_conv1 = nn.Conv3d(64, 32, kernel_size=1)
        self.fusion_conv2 = nn.Conv3d(32, 1, kernel_size=1)
        self.final_conv2d = nn.Conv2d(32, 1, kernel_size=1)

    def forward(self, x_3d, x_2d, arg_2d):
        # 3D path
        fea_3d = self.unet3d(x_3d)

        # 2D path
        fea_2d = self.unet2d(x_2d)
        final_conv = torch.sigmoid(self.final_conv2d(fea_2d))

        # MIP and back-projection
        final_reshape = self.mip_arg(fea_2d)
        back3d = final_reshape * arg_2d
        fea_2d_3d = self.unproject(back3d)

        # Final fusion
        fea_fuse = torch.cat([fea_3d, fea_2d_3d], dim=1)
        fea_fuse = F.relu(self.fusion_conv1(fea_fuse))
        res_fuse = torch.sigmoid(self.fusion_conv2(fea_fuse))

        return res_fuse, final_conv


def get_vc_net(
    cube_size=(1, 128, 128, 16),
    patch_size=(384, 256),
    num_channels_2d=1,
    dropout_2d=0.0,
    batch_norm=False,
):
    """
    创建VC-Net模型

    参数:
        cube_size: 3D输入的大小 (channels, height, width, depth)
        patch_size: 2D输入的大小 (height, width)
        num_channels_2d: 2D输入的通道数
        dropout_2d: 2D UNet中的dropout率
        batch_norm: 是否使用批归一化
    """
    model = VCNet(
        cube_size=cube_size,
        patch_size=patch_size,
        num_channels_2d=num_channels_2d,
        dropout_2d=dropout_2d,
        batch_norm=batch_norm,
    )
    return model


if __name__ == "__main__":
    # 测试代码
    model = get_vc_net()
    x_3d = torch.randn(1, 1, 128, 128, 16)
    x_2d = torch.randn(1, 1, 384, 256)
    arg_2d = torch.randn(1, 1, 6, 128, 128, 5)

    output_3d, output_2d = model(x_3d, x_2d, arg_2d)
    print(f"3D output shape: {output_3d.shape}")
    print(f"2D output shape: {output_2d.shape}")
