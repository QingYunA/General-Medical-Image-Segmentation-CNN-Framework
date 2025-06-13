from .stdc_helper import StdcEncoder, UnetDecoder
import torch.nn as nn
from einops import repeat
import math
import torch


class ProjectionSegNet(nn.Module):
    def __init__(
        self, in_channels=1, out_channels=1, init_features=32, tsne_mode=False
    ) -> None:
        super().__init__()
        self.init_features = init_features
        self.tsne_mode = tsne_mode
        self.conv1 = nn.Conv3d(
            in_channels, init_features, kernel_size=3, stride=1, padding=1
        )  # *  increase chanels, keep feature map size
        self.encoder = StdcEncoder(in_channels, init_features)

        self.path1 = UnetDecoder(init_features, out_channels, dim="2d")
        self.path2 = UnetDecoder(init_features, out_channels, dim="2d")
        self.path3 = UnetDecoder(init_features, out_channels, dim="2d")
        self.seg_path = UnetDecoder(init_features, out_channels, dim="3d")
        self.conv_head = nn.Conv3d(
            in_channels=32,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        # self.rec_head = nn.Conv3d(in_channels=32, out_channels=out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, inputs):
        # * 3d 2d Path
        xy_planes, xz_planes, yz_planes, masks, out_3d_list, stack_list = self.encoder(
            inputs
        )

        # * calculate spatial loss (decouple)
        _, _, _, path4_stack = stack_list
        t_loss = self.decouple_loss(path4_stack)

        # * recover 2d -> 3d
        xy_out, xy_decs = self.path1(xy_planes)
        xz_out, xz_decs = self.path2(xz_planes)
        yz_out, yz_decs = self.path3(yz_planes)  # * shape [1,32,64,64]
        dec4_list, dec3_list, dec2_list, dec1_list = [], [], [], []
        for i in [xy_decs, xz_decs, yz_decs]:
            dec4_list.append(i[0])
            dec3_list.append(i[1])
            dec2_list.append(i[2])
            dec1_list.append(i[3])
        xy_expand = repeat(xy_out, "b c x y -> b c x y z", z=64)
        xz_expand = repeat(xz_out, "b c x z -> b c x y z", y=64)
        yz_expand = repeat(yz_out, "b c y z -> b c x y z", x=64)

        _, _, _, s1_masks = masks
        xy_mask, xz_mask, yz_mask = s1_masks

        xy_rec = xy_expand * xy_mask
        xz_rec = xz_expand * xz_mask
        yz_rec = yz_expand * yz_mask
        # xy_rec = xy_expand
        # xz_rec = xz_expand
        # yz_rec = yz_expand

        out = (xy_rec + xz_rec + yz_rec) / 3
        out = self.conv_head(out)

        # * get 2d middle decoder to 3d
        if self.training:
            decs = [dec4_list, dec3_list, dec2_list, dec1_list]
            m_out = []
            for index, (m, dec) in enumerate(
                zip(masks, decs)
            ):  # * masks [s4_masks,...] s4_masks contain [xy_mask,...]  decs [dec4_list,...] dec4_list: [xy_dec4,xz_dec4...]
                xy_mask, xz_mask, yz_mask = m
                xy_dec, xz_dec, yz_dec = dec

                t_xy_expand = repeat(
                    xy_dec, "b c x y -> b c x y z", z=4 * int(math.pow(2, index + 1))
                )
                t_xz_expand = repeat(
                    xz_dec, "b c x z -> b c x y z", y=4 * int(math.pow(2, index + 1))
                )
                t_yz_expand = repeat(
                    yz_dec, "b c y z -> b c x y z", x=4 * int(math.pow(2, index + 1))
                )
                t_xy_rec = t_xy_expand * xy_mask
                t_xz_rec = t_xz_expand * xz_mask
                t_yz_rec = t_yz_expand * yz_mask
                # t_xy_rec = t_xy_expand
                # t_xz_rec = t_xz_expand
                # t_yz_rec = t_yz_expand
                t_out = (t_xy_rec + t_xz_rec + t_yz_rec) / 3
                m_out.append(t_out)  # * sequence: [4,3,2,1]

            # * seg path for co-train
            out_3d, out_decs = self.seg_path(out_3d_list)
            # out_3d=self.rec_head(out_3d)
            return out, out_3d, m_out, out_decs, t_loss
        else:
            out_3d, out_decs = self.seg_path(out_3d_list)
            if self.tsne_mode:
                return out, path4_stack
            else:
                return out, out_3d

    def decouple_loss(self, feature_map):
        # * input [bs,3,c,h,w] 3: xy,xz,yz path feature map
        bs, path_num, c_num, _, _ = feature_map.shape
        contrast_loss = 0.0
        i1, i2 = 0, 0
        for i in range(path_num):
            pos_vec = feature_map[:, i, ...]  # * shape:[bs,c,h,w]
            for j in range(4, c_num // 2 - 1, 16):
                i1 += 1
                data_list = []
                data_list.append(pos_vec[:, j, ...])
                data_list.append(pos_vec[:, j + c_num // 4, ...])
                # * 放了两个元素,每个元素是取出来的一个channel,shape[bs,8]
                neg_list = [
                    feature_map[:, k, ...][:, j - 4 : j + 5, ...]
                    for k in range(path_num)
                    if k != i
                ]
                # * 取出来 9 个相邻的通道
                data_list.extend(
                    neg_list[idx][:, m, ...]
                    for idx in range(path_num - 1)
                    for m in range(neg_list[0].shape[1])
                )
                contrast_loss += self.one_modality_loss(data_list, t=0.07)
        return contrast_loss / (i1 + i2)

    def one_modality_loss(self, data_list, t=0.07):
        pos_score = self.score(data_list[0], data_list[1], t)
        all_score = 0.0
        for i in range(1, len(data_list)):
            all_score += self.score(data_list[0], data_list[i], t)
        contrast = -torch.log(pos_score / all_score + 1e-5).mean()

        return contrast

    def score(self, fm_1, fm_2, t):
        """
        t: temperature
        """
        # if torch.norm(fm_1, dim=1).mean().item() <= 0.001 or torch.norm(fm_2, dim=1).mean().item() <= 0.001:
        #     print(torch.norm(fm_1, dim=1).mean().item(), torch.norm(fm_2, dim=1).mean().item())

        return torch.exp(
            (fm_1 * fm_2).sum(1)
            / (t * (torch.norm(fm_1, dim=1) * torch.norm(fm_2, dim=1)) + 1e-5)
        )


if __name__ == "__main__":
    import os
    from torchsummary import summary

    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.randn(4, 1, 64, 64, 64).to(device)

    net = ProjectionSegNet()
    net.cuda()
    out, p_out, m_out, out_decs, t_loss = net(x)
    print(t_loss)
    # mask=torch.where(dec4>0)
    # print(out.shape)

    # # onnx_path = './saved_model.onnx'
    # # out, out16, out32 = net(in_ten)
    # # torch.onnx.export(net, in_ten, onnx_path)
    # # netron.start(onnx_path)
    # # torch.save(net.state_dict(), 'STDCNet813.pth')
    # summary(net, (1, 64, 64, 64))
    # import torch
    # a = torch.randn(2, 3, 3)
    # #* help generate a [2,3,3] torch tensor
    # b = a * True
    # print(b)
