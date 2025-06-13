import importlib
import os
from typing import Dict, Any, Optional

# 导入所有模型
from .nets.vc_net import get_vc_net
from .nets.vnet3d import VNet
from .nets.csrnet import CSRNet
from .nets.highresnet import HighResNet
from .nets.stdc2d_v2 import ProjectionSegNet
from .nets.ER_net import ER_Net
from .nets.densevoxelnet3d import DenseVoxelNet

# from .fcn3d import FCN3D
# from .FusionNet import FusionNet
from .nets.RE_net import RE_Net
from .nets.densenet3d import SkipDenseNet3D
from .nets.unetr import UNETR

# from .vtnet import VTUNet
from .nets.residual_unet3d import UNet as ResidualUNet3D
from .nets.unet3d import UNet3D
from .pl_models import BaseModel, TriPlaneModel
# from .vt_unet import


def select_model(model_name: str, model_params, pl_model_params) -> Any:
    model_dict = {
        "unet3d": UNet3D,
        "residual_unet3d": ResidualUNet3D,
        "vnet3d": VNet,
        "densevoxelnet3d": DenseVoxelNet,
        "densenet3d": SkipDenseNet3D,
        # "stdc_unet": SkipDenseNet3D,
        "csrnet": CSRNet,
        "highresnet": HighResNet,
        "vc_net": get_vc_net,
        "er_net": ER_Net,
        "re_net": RE_Net,
        "unetr": UNETR,
        "stdc2d": ProjectionSegNet,
        # "fcn3d": FCN3D,
        # "fusionnet": FusionNet,
        # "vtnet": VTNet,
        # "vt_unet": VTUNet,
    }

    model_name = model_name.lower()
    if model_name not in model_dict:
        raise ValueError(
            f"不支持的模型名称: {model_name}。支持的模型有: {list(model_dict.keys())}"
        )
    model = model_dict[model_name](**model_params)
    # if pl_model_params:
    #     model = pl_model_base(model=model, **pl_model_params)
    if model_name == "stdc2d":
        model = TriPlaneModel(model=model, **pl_model_params)
    else:
        model = BaseModel(model=model, **pl_model_params)

    return model


def get_available_models() -> list:
    return list(select_model.__annotations__["model_name"].__args__)


if __name__ == "__main__":
    # 测试代码
    model = select_model("vc_net")
    print(f"成功创建模型: {type(model).__name__}")

    # 打印所有可用模型
    print("\n可用的模型列表:")
    for model_name in get_available_models():
        print(f"- {model_name}")
