from collections import OrderedDict
import torch
import torch.nn as nn


class CSRNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=3, init_features=64):
        """
        Implementations based on the Unet3D paper: https://arxiv.org/abs/1606.06650
        """

        super(CSRNet, self).__init__()

        features = init_features
        self.encoder1 = CSRNet._block(in_channels, features, name="enc1")
        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.encoder2 = CSRNet._block(features, features * 2, name="enc2")
        self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.encoder3 = CSRNet._block(features * 2, features * 4, name="enc3")
        self.pool3 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.encoder4 = CSRNet._block(features * 4, features * 8, name="enc4")
        self.pool4 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.encoder_r_1 = CSRNet._block_r(features, features * 4, name="enc1_r")
        self.encoder_r_2 = CSRNet._block_r(features * 2, features * 8, name="enc2_r")
        self.encoder_r_3 = CSRNet._block_r(features * 4, features * 16, name="enc3_r")

        self.bottleneck = CSRNet._block(features * 8, features * 16, name="bottleneck")

        self.upconv4 = nn.ConvTranspose3d(
            features * 16, features * 8, kernel_size=2, stride=2
        )
        self.decoder4 = CSRNet._block((features * 8) * 2, features * 8, name="dec4")
        self.upconv3 = nn.ConvTranspose3d(
            features * 8, features * 4, kernel_size=2, stride=2
        )
        self.decoder3 = CSRNet._block((features * 4) * 2, features * 4, name="dec3")
        self.upconv2 = nn.ConvTranspose3d(
            features * 4, features * 2, kernel_size=2, stride=2
        )
        self.decoder2 = CSRNet._block((features * 2) * 2, features * 2, name="dec2")
        self.upconv1 = nn.ConvTranspose3d(
            features * 2, features, kernel_size=2, stride=2
        )
        self.decoder1 = CSRNet._block(features * 2, features, name="dec1")

        self.conv = nn.Conv3d(
            in_channels=features, out_channels=out_channels, kernel_size=1
        )

        self.dncoder_r_1 = CSRNet._block_rr(features*16, features * 4, name="dnc1_r")
        self.dncoder_r_2 = CSRNet._block_rr(features * 8, features * 2, name="dnc2_r")
        self.dncoder_r_3 = CSRNet._block_rr(features * 4, features * 1, name="dnc3_r")

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc3 = enc3 + self.encoder_r_1(enc1)
        enc4 = self.encoder4(self.pool3(enc3))
        enc4 = enc4 + self.encoder_r_2(enc2)

        bottleneck = self.bottleneck(self.pool4(enc4))
        bottleneck = bottleneck + self.encoder_r_3(enc3)

        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3 + self.dncoder_r_1(bottleneck), enc3), dim=1)
        dec3 = self.decoder3(dec3)
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2 + self.dncoder_r_2(dec4), enc2), dim=1)
        dec2 = self.decoder2(dec2)
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1 + self.dncoder_r_3(dec3), enc1), dim=1)
        dec1 = self.decoder1(dec1)
        outputs = self.conv(dec1)
        return outputs

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

    @staticmethod
    def _block_r(in_channels, features, name):
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "conv1",
                        nn.Conv3d(
                            in_channels=in_channels,
                            out_channels=features,
                            kernel_size=3,
                            stride=4,
                            bias=True,
                        ),
                    ),
                    (name + "norm1", nn.BatchNorm3d(num_features=features)),
                    (name + "relu1", nn.ReLU(inplace=True)),
                ]
            )
        )

    @staticmethod
    def _block_rr(in_channels, features, name):
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "conv1",
                        nn.ConvTranspose3d(
                            in_channels=in_channels,
                            out_channels=features,
                            kernel_size=4,
                            stride=4,
                            bias=True,
                        ),
                    ),
                    (name + "norm1", nn.BatchNorm3d(num_features=features)),
                    (name + "relu1", nn.ReLU(inplace=True)),
                ]
            )
        )


if __name__ == "__main__":

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    image_size = 128
    x = torch.Tensor(1, 1, image_size, image_size, image_size)
    x = x.to(device)
    print("x size: {}".format(x.size()))

    model = CSRNet(in_channels=1, out_channels=1, init_features=32).to(device)

    out = model(x)
    print("out size: {}".format(out.size()))
