import copy
import logging

import torch
import torch.nn as nn

from models.three_d.vt_unet import SwinTransformerSys3D

logger = logging.getLogger(__name__)


class VTUNet(nn.Module):
    def __init__(self, num_classes=2, input_dim=1, zero_head=False, embed_dim=96, win_size=7):
        super(VTUNet, self).__init__()
        self.num_classes = num_classes
        self.input_dim = input_dim
        self.zero_head = zero_head
        self.embed_dim = embed_dim
        self.win_size = win_size
        self.win_size = (self.win_size, self.win_size, self.win_size)

        self.swin_unet = SwinTransformerSys3D(img_size=(128, 128, 128),
                                              patch_size=(4, 4, 4),
                                              in_chans=self.input_dim,
                                              num_classes=self.num_classes,
                                              embed_dim=self.embed_dim,
                                              depths=[2, 2, 2, 1],
                                              depths_decoder=[1, 2, 2, 2],
                                              num_heads=[3, 6, 12, 24],
                                              window_size=self.win_size,
                                              mlp_ratio=4.,
                                              qkv_bias=True,
                                              qk_scale=None,
                                              drop_rate=0.,
                                              attn_drop_rate=0.,
                                              drop_path_rate=0.1,
                                              norm_layer=nn.LayerNorm,
                                              patch_norm=True,
                                              use_checkpoint=False,
                                              frozen_stages=-1,
                                              final_upsample="expand_first")

    def forward(self, x):
        logits = self.swin_unet(x)
        return logits


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    x = torch.Tensor(1, 1, 128, 128, 128)
    x.to(device)
    print("x size: {}".format(x.size()))

    model = VTUNet()

    out = model(x)

    print("out size: {}".format(out.size()))
