import torch
import torchvision
import numpy as np
from collections import OrderedDict
import torch.nn as nn
# from torchinfo import summary
from thop import profile

# from models.three_d.SE import SE_Inception, SE_Residual
from .SE import SE_Residual, SE_Inception


# --------------------------------------------------------------------------------

class Double_Unet(nn.Module):
    def __init__(self, in_channels=1, out_channels=2, unet_init_features=64, Cnn_init_features=64, elu=True):
        super(Double_Unet, self).__init__()

        # ? 1 Coarse Unet
        in_chan = in_channels
        features = unet_init_features // 2
        # * Encoder
        self.cu_encoder1 = self._block(in_chan, features, name="cu_enc1")
        self.cu_pool1 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.cu_encoder2 = self._block(features, features * 2, name="cu_enc2")
        self.cu_pool2 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.cu_encoder3 = self._block(features * 2, features * 4, name="cu_enc3")
        self.cu_pool3 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.cu_bottleneck = self._block(features * 4, features * 8, name="cu_bottleneck")

        # * Decoder
        self.cu_upconv3 = nn.ConvTranspose3d(features * 8, features * 8, kernel_size=2, stride=2)
        self.cu_decoder3 = self._block((features * 4) * 3, features * 4, name="cu_dec3")
        self.cu_upconv2 = nn.ConvTranspose3d(features * 4, features * 4, kernel_size=2, stride=2)
        self.cu_decoder2 = self._block((features * 2) * 3, features * 2, name="cu_dec2")
        self.cu_upconv1 = nn.ConvTranspose3d(features * 2, features * 2, kernel_size=2, stride=2)
        self.cu_decoder1 = self._block(features * 3, features, name="cu_dec1")
        self.cu_conv = nn.Conv3d(in_channels=features, out_channels=out_channels, kernel_size=1)


        # ? 1 Coarse Unet
        in_chan = in_channels + out_channels
        features = unet_init_features
        # * Encoder
        self.fu_encoder1 = self._block(in_chan, features, name="fu_enc1")
        self.fu_pool1 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.fu_encoder2 = self._block(features, features * 2, name="fu_enc2")
        self.fu_pool2 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.fu_encoder3 = self._block(features * 2, features * 4, name="fu_enc3")
        self.fu_pool3 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.fu_bottleneck = self._block(features * 4, features * 8, name="fu_bottleneck")

        # * Decoder
        self.fu_upconv3 = nn.ConvTranspose3d(features * 8, features * 8, kernel_size=2, stride=2)
        self.fu_decoder3 = self._block((features * 4) * 3, features * 4, name="fu_dec3")
        self.fu_upconv2 = nn.ConvTranspose3d(features * 4, features * 4, kernel_size=2, stride=2)
        self.fu_decoder2 = self._block((features * 2) * 3, features * 2, name="fu_dec2")
        self.fu_upconv1 = nn.ConvTranspose3d(features * 2, features * 2, kernel_size=2, stride=2)
        self.fu_decoder1 = self._block(features * 3, features, name="fu_dec1")
        self.fu_conv = nn.Conv3d(in_channels=features, out_channels=out_channels, kernel_size=1)

        self.SE3 = SE_Residual(4*features)
        self.SE2 = SE_Residual(2*features)
        self.SE1 = SE_Residual(features)

        # self.SE3 = SE_Inception(4*features)
        # self.SE2 = SE_Inception(2*features)
        # self.SE1 = SE_Inception(features)

    def forward(self, x):
        # * Coarse Unet
        enc1 = self.cu_encoder1(x)
        enc2 = self.cu_encoder2(self.cu_pool1(enc1))
        enc3 = self.cu_encoder3(self.cu_pool2(enc2))

        bottleneck = self.cu_bottleneck(self.cu_pool3(enc3))

        dec3 = self.cu_upconv3(bottleneck)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.cu_decoder3(dec3)
        dec2 = self.cu_upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.cu_decoder2(dec2)
        dec1 = self.cu_upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.cu_decoder1(dec1)
        cu_outputs = self.cu_conv(dec1)

        # * Fine Unet
        x_ = torch.cat((x, cu_outputs), dim=1)
        enc1 = self.fu_encoder1(x_)
        enc2 = self.fu_encoder2(self.fu_pool1(enc1))
        enc3 = self.fu_encoder3(self.fu_pool2(enc2))

        bottleneck = self.fu_bottleneck(self.fu_pool3(enc3))

        dec3 = self.fu_upconv3(bottleneck)
        enc3 = self.SE3(enc3)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.fu_decoder3(dec3)
        dec2 = self.fu_upconv2(dec3)
        enc2 = self.SE2(enc2)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.fu_decoder2(dec2)
        dec1 = self.fu_upconv1(dec2)
        enc1 = self.SE1(enc1)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.fu_decoder1(dec1)
        fu_outputs = self.fu_conv(dec1)
        return fu_outputs

    @staticmethod
    def _block(in_channels, features, name):
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "conv1",
                        nn.Conv3d(
                            in_channels=in_channels,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=True,
                        ),
                    ),
                    (name + "norm1", nn.BatchNorm3d(num_features=features)),
                    (name + "relu1", nn.ReLU(inplace=True)),
                    (
                        name + "conv2",
                        nn.Conv3d(
                            in_channels=features,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=True,
                        ),
                    ),
                    (name + "norm2", nn.BatchNorm3d(num_features=features)),
                    (name + "relu2", nn.ReLU(inplace=True)),
                ]
            )
        )

    def _block_CNN(self, in_channels, out_channels, name):
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "conv1",
                        nn.Conv3d(
                            in_channels=in_channels,
                            out_channels=out_channels,
                            kernel_size=3,
                            padding=1,
                            bias=True,
                        ),
                    ),
                    (name + "norm1", nn.BatchNorm3d(num_features=out_channels)),
                    (name + "relu1", nn.ReLU(inplace=True))
                ]
            )
        )

if __name__ == "__main__":
    model = Double_Unet()
    input = torch.randn(1, 1, 64, 64, 64)
    