import torch
import torch.nn as nn

# from torchsummary import summary


class ConvX(nn.Module):
    def __init__(
        self,
        in_planes,
        out_planes,
        kernel=3,
        stride=1,
        padding=None,
        dim="3d",
        use_LN=False,
    ):
        super(ConvX, self).__init__()
        # self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel, stride=stride, padding=kernel // 2, bias=False)
        if padding == None:
            padding = kernel // 2
        if dim == "2d":
            self.conv = nn.Conv2d(
                in_planes,
                out_planes,
                kernel_size=kernel,
                stride=stride,
                padding=padding,
                bias=False,
            )
            self.bn = nn.BatchNorm2d(out_planes)
        elif dim == "3d":
            self.conv = nn.Conv3d(
                in_planes,
                out_planes,
                kernel_size=kernel,
                stride=stride,
                padding=padding,
                bias=False,
            )
            # if use_LN:
            #     self.bn=nn.LayerNorm(out_planes)
            # else:
            self.bn = nn.BatchNorm3d(out_planes)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.relu(self.bn(self.conv(x)))
        return out


class UnetBlock(nn.Module):
    def __init__(self, in_planes, out_planes, dim="2d") -> None:
        super().__init__()
        if dim == "2d":
            self.conv1 = nn.Conv2d(
                in_planes, out_planes, kernel_size=3, padding=1, bias=False
            )
            self.bn1 = nn.BatchNorm2d(out_planes)
            self.conv2 = nn.Conv2d(
                out_planes, out_planes, kernel_size=3, padding=1, bias=False
            )
            self.bn2 = nn.BatchNorm2d(out_planes)
        else:
            self.conv1 = nn.Conv3d(
                in_planes, out_planes, kernel_size=3, padding=1, bias=False
            )
            self.bn1 = nn.BatchNorm3d(out_planes)
            self.conv2 = nn.Conv3d(
                out_planes, out_planes, kernel_size=3, padding=1, bias=False
            )
            self.bn2 = nn.BatchNorm3d(out_planes)

        self.relu1 = nn.ReLU(inplace=True)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)
        return out


class stdc_module(nn.Module):
    def __init__(self, in_planes, out_planes, stride=1, dim="3d", use_LN=False):
        super().__init__()
        self.stride = stride
        if stride == 2:
            if dim == "2d":
                self.avd_layer = nn.Sequential(
                    nn.Conv2d(
                        out_planes // 2,
                        out_planes // 2,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        groups=out_planes // 2,
                        bias=False,
                    ),
                    nn.BatchNorm2d(out_planes // 2),
                )
                self.skip = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
            if dim == "3d":
                self.avd_layer = nn.Sequential(
                    nn.Conv3d(
                        out_planes // 2,
                        out_planes // 2,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        groups=out_planes // 2,
                        bias=False,
                    ),
                    nn.BatchNorm3d(out_planes // 2),
                )
                self.skip = nn.AvgPool3d(kernel_size=3, stride=2, padding=1)

        self.block1 = ConvX(
            in_planes, out_planes // 2, kernel=1, stride=1, dim=dim, use_LN=use_LN
        )  # * 1*1 convx
        self.block2 = ConvX(
            out_planes // 2, out_planes // 4, kernel=3, stride=1, dim=dim, use_LN=use_LN
        )
        self.block3 = ConvX(
            out_planes // 4, out_planes // 8, kernel=1, stride=1, dim=dim, use_LN=use_LN
        )
        self.block4 = ConvX(
            out_planes // 8, out_planes // 8, kernel=3, stride=1, dim=dim, use_LN=use_LN
        )
        self.block_list = nn.ModuleList(
            [self.block1, self.block2, self.block3, self.block4]
        )

    def forward(self, x):
        out_list = []
        out1 = self.block_list[0](x)
        for i, block in enumerate(self.block_list[1:]):
            if i == 0:
                if self.stride == 2:
                    out = block(self.avd_layer(out1))
                else:
                    out = block(out1)
            else:
                out = block(out)
            out_list.append(out)
        if self.stride == 2:
            out1 = self.skip(out1)
        out_list.insert(0, out1)
        out = torch.cat(out_list, dim=1)
        return out


class StdcEncoder(nn.Module):
    def __init__(self, in_channels=1, init_features=32, seperate=False):
        """
        Implementations based on the Unet3D paper: https://arxiv.org/abs/1606.06650
        """
        super(StdcEncoder, self).__init__()
        self.seperate = seperate

        self.convx1 = ConvX(in_channels, init_features, stride=1, dim="3d", use_LN=True)
        self.convx2 = ConvX(
            init_features, init_features * 2, stride=2, dim="3d", use_LN=True
        )

        # self.conv_last = ConvX(init_features * 32, max(1024, init_features * 32), kernel=1, stride=1)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        # self.fc = nn.Linear(max(1024, init_features * 32), max(1024, init_features * 32), bias=False)
        # self.bn = nn.BatchNorm2d(max(1024, init_features * 32))
        # self.relu = nn.ReLU(inplace=True)
        # self.drop = nn.Dropout(p=0.2)
        # self.linear = nn.Linear(max(1024, init_features * 32), out_channels, bias=False)

        self.stage3 = nn.Sequential(
            stdc_module(
                in_planes=init_features * 2,
                out_planes=init_features * 4,
                stride=2,
                dim="3d",
                use_LN=True,
            ),
            stdc_module(
                in_planes=init_features * 4,
                out_planes=init_features * 4,
                stride=1,
                dim="3d",
                use_LN=True,
            ),
        )
        self.stage4 = nn.Sequential(
            stdc_module(
                in_planes=init_features * 4,
                out_planes=init_features * 8,
                stride=2,
                dim="3d",
                use_LN=True,
            ),
            stdc_module(
                in_planes=init_features * 8,
                out_planes=init_features * 8,
                stride=1,
                dim="3d",
                use_LN=True,
            ),
        )
        self.stage5_2d = nn.Sequential(
            stdc_module(
                in_planes=init_features * 8,
                out_planes=init_features * 16,
                stride=2,
                dim="2d",
            ),
            stdc_module(
                in_planes=init_features * 16,
                out_planes=init_features * 16,
                stride=1,
                dim="2d",
            ),
        )
        self.stage5_3d = nn.Sequential(
            stdc_module(
                in_planes=init_features * 8,
                out_planes=init_features * 16,
                stride=2,
                dim="3d",
            ),
            stdc_module(
                in_planes=init_features * 16,
                out_planes=init_features * 16,
                stride=1,
                dim="3d",
            ),
        )
        self.neck = UnetBlock(init_features * 8, init_features * 16)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        # * 3d
        out1 = self.convx1(x)
        out2 = self.convx2(out1)
        out3 = self.stage3(out2)
        out4 = self.stage4(out3)  # [1,256,4,4,4] should be [1,256,8,8,8]
        out5_3d = self.stage5_3d(out4)
        out_3d_list = [out1, out2, out3, out4, out5_3d]
        # * 3d project to 2d
        if not self.seperate:
            xy1_plane, xz1_plane, yz1_plane, s1_mask = self.projection(out1)
            xy2_plane, xz2_plane, yz2_plane, s2_mask = self.projection(out2)
            xy3_plane, xz3_plane, yz3_plane, s3_mask = self.projection(out3)
            xy4_plane, xz4_plane, yz4_plane, s4_mask = self.projection(out4)

            xy5_out = self.stage5_2d(xy4_plane)
            xz5_out = self.stage5_2d(xz4_plane)
            yz5_out = self.stage5_2d(yz4_plane)

            xy_planes = [xy1_plane, xy2_plane, xy3_plane, xy4_plane, xy5_out]
            xz_planes = [xz1_plane, xz2_plane, xz3_plane, xz4_plane, xz5_out]
            yz_planes = [yz1_plane, yz2_plane, yz3_plane, yz4_plane, yz5_out]

            # * column stack for decouple
            path1_stack = torch.stack(
                (xy1_plane, xz1_plane, yz1_plane), dim=1
            )  # * after : [bs,3,32,64,64]
            path2_stack = torch.stack((xy2_plane, xz2_plane, yz2_plane), dim=1)
            path3_stack = torch.stack((xy3_plane, xz3_plane, yz3_plane), dim=1)
            path4_stack = torch.stack((xy4_plane, xz4_plane, yz4_plane), dim=1)
            stack_list = [path1_stack, path2_stack, path3_stack, path4_stack]
            mask_list = [s4_mask, s3_mask, s2_mask, s1_mask]
            return xy_planes, xz_planes, yz_planes, mask_list, out_3d_list, stack_list
        else:
            return out_3d_list

    def projection(self, inputs):
        xy_plane = torch.max(inputs, -1).values  # * after projection shape:[1,32,64,64]
        xz_plane = torch.max(inputs, -2).values  # * after projection shape:[1,32,64,64]
        yz_plane = torch.max(inputs, -3).values  # * after projection shape:[1,32,64,64]

        xy_mask = inputs == inputs.max(dim=-1, keepdim=True)[0]
        xz_mask = (
            inputs == inputs.max(dim=-2, keepdim=True)[0]
        )  # * shape [1,32,64,64,64]
        yz_mask = inputs == inputs.max(dim=-3, keepdim=True)[0]

        planes = [xy_plane, xz_plane, yz_plane]
        masks = [xy_mask, xz_mask, yz_mask]
        return xy_plane, xz_plane, yz_plane, masks


class UnetDecoder(nn.Module):
    def __init__(self, base, out_channels, dim="2d") -> None:
        super().__init__()

        if dim == "2d":
            self.upconv4 = nn.ConvTranspose2d(
                base * 16, base * 8, kernel_size=2, stride=2
            )
            self.upconv3 = nn.ConvTranspose2d(
                base * 8, base * 4, kernel_size=2, stride=2
            )
            self.upconv2 = nn.ConvTranspose2d(
                base * 4, base * 2, kernel_size=2, stride=2
            )
            self.upconv1 = nn.ConvTranspose2d(base * 2, base, kernel_size=2, stride=2)

            self.decoder4 = UnetBlock((base * 8) * 2, base * 8)
            self.decoder3 = UnetBlock((base * 4) * 2, base * 4)
            self.decoder2 = UnetBlock((base * 2) * 2, base * 2)
            self.decoder1 = UnetBlock(base * 2, base)
            self.conv = nn.Conv2d(
                in_channels=base, out_channels=out_channels, kernel_size=1
            )
        elif dim == "3d":
            self.upconv4 = nn.ConvTranspose3d(
                base * 16, base * 8, kernel_size=2, stride=2
            )
            self.upconv3 = nn.ConvTranspose3d(
                base * 8, base * 4, kernel_size=2, stride=2
            )
            self.upconv2 = nn.ConvTranspose3d(
                base * 4, base * 2, kernel_size=2, stride=2
            )
            self.upconv1 = nn.ConvTranspose3d(base * 2, base, kernel_size=2, stride=2)

            self.decoder4 = UnetBlock((base * 8) * 2, base * 8, dim="3d")
            self.decoder3 = UnetBlock((base * 4) * 2, base * 4, dim="3d")
            self.decoder2 = UnetBlock((base * 2) * 2, base * 2, dim="3d")
            self.decoder1 = UnetBlock(base * 2, base, dim="3d")
            self.conv = nn.Conv3d(
                in_channels=base, out_channels=out_channels, kernel_size=1
            )

    def forward(self, plane_list):
        plane1, plane2, plane3, plane4, plane5 = plane_list
        dec4 = self.upconv4(plane5)
        dec4 = torch.cat((dec4, plane4), dim=1)
        dec4 = self.decoder4(dec4)

        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, plane3), dim=1)
        dec3 = self.decoder3(dec3)

        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, plane2), dim=1)
        dec2 = self.decoder2(dec2)

        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, plane1), dim=1)
        dec1 = self.decoder1(dec1)
        outputs = self.conv(dec1)
        m_outputs = [dec4, dec3, dec2, dec1]
        return outputs, m_outputs
