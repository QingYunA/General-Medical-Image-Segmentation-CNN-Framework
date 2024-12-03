# 3D-UNet model.
# x: 128x128 resolution for 32 frames.
# https://github.com/huangzhii/FCN-3D-pytorch/blob/master/main3d.py
import torch
import torch.nn as nn
import os
import numpy as np
from collections import OrderedDict
from models.three_d.SE import SE_Inception, SE_Residual
# from SE import SE_Residual

def passthrough(x, **kwargs):
    return x


def ELUCons(elu, nchan):
    if elu:
        return nn.ELU(inplace=True)
    else:
        return nn.PReLU(nchan)


class LUConv(nn.Module):
    def __init__(self, nchan, elu):
        super(LUConv, self).__init__()
        self.relu1 = ELUCons(elu, nchan)
        self.conv1 = nn.Conv3d(nchan, nchan, kernel_size=5, padding=2)

        self.bn1 = torch.nn.BatchNorm3d(nchan)

    def forward(self, x):
        out = self.relu1(self.bn1(self.conv1(x)))
        return out


def _make_nConv(nchan, depth, elu):
    layers = []
    for _ in range(depth):
        layers.append(LUConv(nchan, elu))
    return nn.Sequential(*layers)


class InputTransition(nn.Module):
    def __init__(self, in_channels, features, elu):
        super(InputTransition, self).__init__()
        self.num_features = features
        self.in_channels = in_channels

        self.conv1 = nn.Conv3d(self.in_channels, self.num_features, kernel_size=5, padding=2)

        self.bn1 = torch.nn.BatchNorm3d(self.num_features)

        self.relu1 = ELUCons(elu, self.num_features)

    def forward(self, x):
        out = self.conv1(x)
        repeat_rate = int(self.num_features / self.in_channels)
        out = self.bn1(out)
        x16 = x.repeat(1, repeat_rate, 1, 1, 1)
        return self.relu1(torch.add(out, x16))


class DownTransition(nn.Module):
    def __init__(self, inChans, nConvs, elu, dropout=False):
        super(DownTransition, self).__init__()
        outChans = 2 * inChans
        self.down_conv = nn.Conv3d(inChans, outChans, kernel_size=2, stride=2)
        self.bn1 = torch.nn.BatchNorm3d(outChans)

        self.do1 = passthrough
        self.relu1 = ELUCons(elu, outChans)
        self.relu2 = ELUCons(elu, outChans)
        if dropout:
            self.do1 = nn.Dropout3d()
        self.ops = _make_nConv(outChans, nConvs, elu)

    def forward(self, x):
        down = self.relu1(self.bn1(self.down_conv(x)))
        out = self.do1(down)
        out = self.ops(out)
        out = self.relu2(torch.add(out, down))
        return out


class UpTransition(nn.Module):
    def __init__(self, inChans, outChans, nConvs, elu, dropout=False):
        super(UpTransition, self).__init__()
        self.up_conv = nn.ConvTranspose3d(inChans, outChans // 2, kernel_size=2, stride=2)

        self.bn1 = torch.nn.BatchNorm3d(outChans // 2)
        self.do1 = passthrough
        self.do2 = nn.Dropout3d()
        self.relu1 = ELUCons(elu, outChans // 2)
        self.relu2 = ELUCons(elu, outChans)
        if dropout:
            self.do1 = nn.Dropout3d()
        self.ops = _make_nConv(outChans, nConvs, elu)

    def forward(self, x, skipx):
        out = self.do1(x)
        skipxdo = self.do2(skipx)
        out = self.relu1(self.bn1(self.up_conv(out)))
        xcat = torch.cat((out, skipxdo), 1)
        out = self.ops(xcat)
        out = self.relu2(torch.add(out, xcat))
        return out


class OutputTransition(nn.Module):
    def __init__(self, in_channels, classes, elu):
        super(OutputTransition, self).__init__()
        self.classes = classes
        self.conv1 = nn.Conv3d(in_channels, classes, kernel_size=5, padding=2)
        self.bn1 = torch.nn.BatchNorm3d(classes)

        self.conv2 = nn.Conv3d(classes, classes, kernel_size=1)
        self.relu1 = ELUCons(elu, classes)

    def forward(self, x):
        # convolve 32 down to channels as the desired classes
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.conv2(out)
        return out


class VNet(nn.Module):
    """
    Implementations based on the Vnet paper: https://arxiv.org/abs/1606.04797
    """

    def __init__(self, elu=True, in_channels=1, classes=2):
        super(VNet, self).__init__()
        self.classes = classes
        self.in_channels = in_channels

        self.in_tr_1 = InputTransition(in_channels, 8,elu=elu)
        self.down_tr32_1 = DownTransition(8, 1, elu)
        self.down_tr64_1 = DownTransition(16, 2, elu)
        self.down_tr128_1 = DownTransition(32, 3, elu, dropout=False)
        self.down_tr256_1 = DownTransition(64, 2, elu, dropout=False)
        self.up_tr256_1 = UpTransition(128, 128, 2, elu, dropout=False)
        self.up_tr128_1 = UpTransition(128, 64, 2, elu, dropout=False)
        self.up_tr64_1 = UpTransition(64, 32, 1, elu)
        self.up_tr32_1 = UpTransition(32, 16, 1, elu)
        self.out_tr_1 = OutputTransition(16, classes, elu)


        in_channels = classes
        self.in_tr_2 = InputTransition(in_channels, 16, elu=elu)
        self.down_tr32_2 = DownTransition(16, 1, elu)
        self.down_tr64_2 = DownTransition(32, 2, elu)
        self.down_tr128_2 = DownTransition(64, 3, elu, dropout=False)
        self.down_tr256_2 = DownTransition(128, 2, elu, dropout=False)
        self.up_tr256_2 = UpTransition(256, 256, 2, elu, dropout=False)
        self.up_tr128_2 = UpTransition(256, 128, 2, elu, dropout=False)
        self.up_tr64_2 = UpTransition(128, 64, 1, elu)
        self.up_tr32_2 = UpTransition(64, 32, 1, elu)
        self.out_tr_2 = OutputTransition(32, classes, elu)

        self.SE = SE_Residual(256)


    def forward(self, x):
        out16 = self.in_tr_1(x)
        out32 = self.down_tr32_1(out16)
        out64 = self.down_tr64_1(out32)
        out128 = self.down_tr128_1(out64)
        out256 = self.down_tr256_1(out128)
        out = self.up_tr256_1(out256, out128)
        out = self.up_tr128_1(out, out64)
        out = self.up_tr64_1(out, out32)
        out = self.up_tr32_1(out, out16)
        out = self.out_tr_1(out)

        # print(out.shape)

        x_ = out

        # print(x_.shape)
        out16 = self.in_tr_2(x_)
        out32 = self.down_tr32_2(out16)
        out64 = self.down_tr64_2(out32)
        out128 = self.down_tr128_2(out64)
        out256 = self.down_tr256_2(out128)
        out256 = self.SE(out256)
        out = self.up_tr256_2(out256, out128)
        out = self.up_tr128_2(out, out64)
        out = self.up_tr64_2(out, out32)
        out = self.up_tr32_2(out, out16)
        out = self.out_tr_2(out)

        return out


if __name__ == "__main__":
    model = VNet()
    print(model)
    x = torch.randn(1, 1, 64, 64, 64)
    print(model(x).shape)

   