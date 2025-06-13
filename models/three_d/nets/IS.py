import numpy as np
from collections import OrderedDict
import torch
import torch.nn as nn
# from torchsummary import summary


class UNet3D(nn.Module):
    def __init__(self, in_channels=1, out_channels=3, init_features=64):
        """
        Implementations based on the Unet3D paper: https://arxiv.org/abs/1606.06650
        """

        super(UNet3D, self).__init__()

        features = init_features
        self.encoder1 = UNet3D._block(in_channels, features, name="enc1")
        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.encoder2 = UNet3D._block(features, features * 2, name="enc2")
        self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.encoder3 = UNet3D._block(features * 2, features * 4, name="enc3")
        self.pool3 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.encoder4 = UNet3D._block(features * 4, features * 8, name="enc4")
        self.pool4 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.bottleneck = UNet3D._block(features * 8, features * 16, name="bottleneck")

        self.upconv4 = nn.ConvTranspose3d(
            features * 16, features * 8, kernel_size=2, stride=2
        )
        self.decoder4 = UNet3D._block((features * 8) * 2 , features * 8, name="dec4")
        self.upconv3 = nn.ConvTranspose3d(
            features * 8, features * 4, kernel_size=2, stride=2
        )
        self.decoder3 = UNet3D._block((features * 4) * 2, features * 4, name="dec3")
        self.upconv2 = nn.ConvTranspose3d(
            features * 4, features * 2, kernel_size=2, stride=2
        )
        self.decoder2 = UNet3D._block((features * 2) * 2, features * 2, name="dec2")
        self.upconv1 = nn.ConvTranspose3d(
            features * 2, features, kernel_size=2, stride=2
        )
        self.decoder1 = UNet3D._block(features * 2, features, name="dec1")




        self.encoder1_ = UNet3D._block(in_channels, features, name="enc1")
        self.pool1_ = nn.MaxPool3d(kernel_size=2, stride=2)
        self.encoder2_ = UNet3D._block(features, features * 2, name="enc2")
        self.pool2_ = nn.MaxPool3d(kernel_size=2, stride=2)
        self.encoder3_ = UNet3D._block(features * 2, features * 4, name="enc3")
        self.pool3_ = nn.MaxPool3d(kernel_size=2, stride=2)
        self.encoder4_ = UNet3D._block(features * 4, features * 8, name="enc4")
        self.pool4_ = nn.MaxPool3d(kernel_size=2, stride=2)

        self.bottleneck_ = UNet3D._block(features * 8, features * 16, name="bottleneck")

        self.upconv4_ = nn.ConvTranspose3d(
            features * 16, features * 8, kernel_size=2, stride=2
        )
        self.decoder4_ = UNet3D._block((features * 8) * 2 , features * 8, name="dec4")
        self.upconv3_ = nn.ConvTranspose3d(
            features * 8, features * 4, kernel_size=2, stride=2
        )
        self.decoder3_ = UNet3D._block((features * 4) * 2, features * 4, name="dec3")
        self.upconv2_ = nn.ConvTranspose3d(
            features * 4, features * 2, kernel_size=2, stride=2
        )
        self.decoder2_ = UNet3D._block((features * 2) * 2, features * 2, name="dec2")
        self.upconv1_ = nn.ConvTranspose3d(
            features * 2, features, kernel_size=2, stride=2
        )
        self.decoder1_ = UNet3D._block(features * 2, features, name="dec1")


        self.encoder1__= UNet3D._block(in_channels, features, name="enc1")
        self.pool1__ = nn.MaxPool3d(kernel_size=2, stride=2)
        self.encoder2__ = UNet3D._block(features, features * 2, name="enc2")
        self.pool2__ = nn.MaxPool3d(kernel_size=2, stride=2)
        self.encoder3__ = UNet3D._block(features * 2, features * 4, name="enc3")
        self.pool3__ = nn.MaxPool3d(kernel_size=2, stride=2)
        self.encoder4__ = UNet3D._block(features * 4, features * 8, name="enc4")
        self.pool4__ = nn.MaxPool3d(kernel_size=2, stride=2)

        self.bottleneck__ = UNet3D._block(features * 8, features * 16, name="bottleneck")

        self.upconv4__ = nn.ConvTranspose3d(
            features * 16, features * 8, kernel_size=2, stride=2
        )
        self.decoder4__ = UNet3D._block((features * 8) * 2 , features * 8, name="dec4")
        self.upconv3__ = nn.ConvTranspose3d(
            features * 8, features * 4, kernel_size=2, stride=2
        )
        self.decoder3__ = UNet3D._block((features * 4) * 2, features * 4, name="dec3")
        self.upconv2__ = nn.ConvTranspose3d(
            features * 4, features * 2, kernel_size=2, stride=2
        )
        self.decoder2__ = UNet3D._block((features * 2) * 2, features * 2, name="dec2")
        self.upconv1__ = nn.ConvTranspose3d(
            features * 2, features, kernel_size=2, stride=2
        )
        self.decoder1__ = UNet3D._block(features * 2, features, name="dec1")



        self.conv = nn.Conv3d(
            in_channels=features, out_channels=out_channels, kernel_size=1
        )
        self.conv_ = nn.Conv3d(
            in_channels=features, out_channels=out_channels, kernel_size=1
        )

    def forward(self, x, low_x, high_x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))
        bottleneck = self.bottleneck(self.pool4(enc4))
        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)

        enc1 = self.encoder1(low_x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))
        bottleneck = self.bottleneck(self.pool4_(enc4))
        dec4_ = self.upconv4_(bottleneck)
        dec4_ = torch.cat((dec4_, enc4), dim=1)
        dec4_ = self.decoder4_(dec4_)
        dec3_ = self.upconv3_(dec4_)
        dec3_ = torch.cat((dec3_, enc3), dim=1)
        dec3_ = self.decoder3_(dec3_)
        dec2_ = self.upconv2_(dec3_)
        dec2_ = torch.cat((dec2_, enc2), dim=1)
        dec2_ = self.decoder2_(dec2_)
        dec1_ = self.upconv1_(dec2_)
        dec1_ = torch.cat((dec1_, enc1), dim=1)
        dec1_ = self.decoder1_(dec1_)

        enc1 = self.encoder1(high_x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))
        bottleneck = self.bottleneck(self.pool4(enc4))
        dec4__ = self.upconv4__(bottleneck)
        dec4__ = torch.cat((dec4__, enc4), dim=1)
        dec4__ = self.decoder4__(dec4__)
        dec3__ = self.upconv3__(dec4__)
        dec3__ = torch.cat((dec3__, enc3), dim=1)
        dec3__ = self.decoder3__(dec3__)
        dec2__ = self.upconv2__(dec3__)
        dec2__ = torch.cat((dec2__, enc2), dim=1)
        dec2__ = self.decoder2__(dec2__)
        dec1__ = self.upconv1__(dec2__)
        dec1__ = torch.cat((dec1__, enc1), dim=1)
        dec1__ = self.decoder1__(dec1__)

        # enc1_ = self.encoder1_(low_x)
        # enc2_ = self.encoder2_(self.pool1_(enc1_))
        # enc3_ = self.encoder3_(self.pool2_(enc2_))
        # enc4_ = self.encoder4_(self.pool3_(enc3_))
        # bottleneck_ = self.bottleneck_(self.pool4_(enc4_))
        # dec4_ = self.upconv4_(bottleneck_)
        # dec4_ = torch.cat((dec4_, enc4_), dim=1)
        # dec4_ = self.decoder4_(dec4_)
        # dec3_ = self.upconv3_(dec4_)
        # dec3_ = torch.cat((dec3_, enc3_), dim=1)
        # dec3_ = self.decoder3_(dec3_)
        # dec2_ = self.upconv2_(dec3_)
        # dec2_ = torch.cat((dec2_, enc2_), dim=1)
        # dec2_ = self.decoder2_(dec2_)
        # dec1_ = self.upconv1_(dec2_)
        # dec1_ = torch.cat((dec1_, enc1_), dim=1)
        # dec1_ = self.decoder1_(dec1_)

        # enc1__ = self.encoder1__(high_x)
        # enc2__ = self.encoder2__(self.pool1__(enc1__))
        # enc3__ = self.encoder3__(self.pool2__(enc2__))
        # enc4__ = self.encoder4__(self.pool3__(enc3__))
        # bottleneck__ = self.bottleneck__(self.pool4__(enc4__))
        # dec4__ = self.upconv4__(bottleneck__)
        # dec4__ = torch.cat((dec4__, enc4__), dim=1)
        # dec4__ = self.decoder4__(dec4__)
        # dec3__ = self.upconv3__(dec4__)
        # dec3__ = torch.cat((dec3__, enc3__), dim=1)
        # dec3__ = self.decoder3__(dec3__)
        # dec2__ = self.upconv2__(dec3__)
        # dec2__ = torch.cat((dec2__, enc2__), dim=1)
        # dec2__ = self.decoder2__(dec2__)
        # dec1__ = self.upconv1__(dec2__)
        # dec1__ = torch.cat((dec1__, enc1__), dim=1)
        # dec1__ = self.decoder1__(dec1__)





        outputs1 = self.conv(dec1)
        outputs2 = self.conv_(dec1+dec1_+dec1__)
        # outputs2 = self.conv_(dec1+dec1__)  #x+ high
        # outputs2 = self.conv_(dec1+dec1_)  #x+low

        return outputs1, outputs2

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


