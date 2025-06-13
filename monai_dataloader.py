import os
import numpy as np
import torch
from monai.data import Dataset, DataLoader, CacheDataset
from monai.transforms import (
    LoadImaged,
    EnsureChannelFirstd,
    ScaleIntensityd,
    Compose,
    ToTensor,
    NormalizeIntensityd,
    RandCropByPosNegLabeld,
)
from pathlib import Path
from monai.data import list_data_collate
from typing import List, Optional, Tuple, Union


class MhdDataset:
    def __init__(
        self,
        source_path: str,
        label_path: str,
        test_source_path: str,
        test_label_path: str,
        patch_size: Tuple[int, int, int] = (128, 128, 128),
        batch_size: int = 1,
        num_workers: int = 8,
        num_samples: int = 8,  # patch sample nums
        is_train: bool = True,
        cache: bool = True,
        cache_rate: float = 1,
        pos_ratio: float = 1,
        neg_ratio: float = 1,
        suffix: str = "mhd",
        fast_mode: bool = False,
    ):
        batch_size = 1 if not is_train else batch_size
        self.source_path = source_path if is_train else test_source_path
        self.label_path = label_path if is_train else test_label_path
        self.patch_size = patch_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.is_train = is_train
        self.cache = cache
        self.num_samples = num_samples
        self.pos_ratio = pos_ratio
        self.neg_ratio = neg_ratio
        self.suffix = suffix
        self.fast_mode = fast_mode
        # 获取数据文件列表
        self.data_dicts = self._get_data_files()

        # 设置数据转换
        self.transforms = self._get_transforms()

        # 创建数据集

        self.dataset = CacheDataset(
            data=self.data_dicts,
            transform=self.transforms,
            cache_rate=cache_rate,
            num_workers=num_workers,
        )

        # 创建数据加载器
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=is_train,
            num_workers=num_workers,
            collate_fn=list_data_collate,
            pin_memory=torch.cuda.is_available(),
        )

    def _get_data_files(self) -> list[dict]:
        source_files = list(Path(self.source_path).glob(f"*.{self.suffix}"))
        label_files = list(Path(self.label_path).glob(f"*.{self.suffix}"))
        data_dicts = []
        for file in source_files:
            label_file = label_files[source_files.index(file)]
            data_dicts.append(
                {
                    "image": file,
                    "label": label_file,
                }
            )
        if self.fast_mode:
            return data_dicts[:2]
        return data_dicts

    def _get_transforms(self) -> Compose:
        if self.is_train:
            return Compose(
                [
                    LoadImaged(
                        keys=["image", "label"],
                        image_only=True,
                        ensure_channel_first=True,
                    ),
                    NormalizeIntensityd(
                        keys=["image"],
                        nonzero=True,
                        channel_wise=True,
                    ),
                    RandCropByPosNegLabeld(
                        keys=["image", "label"],
                        label_key="label",
                        num_samples=self.num_samples,
                        spatial_size=self.patch_size,
                        pos=self.pos_ratio,
                        neg=self.neg_ratio,
                    ),
                ]
            )
        else:
            return Compose(
                [
                    LoadImaged(
                        keys=["image", "label"],
                        image_only=True,
                        ensure_channel_first=True,
                    ),
                    NormalizeIntensityd(
                        keys=["image"], nonzero=True, channel_wise=True
                    ),
                ]
            )

    def get_dataloader(self) -> DataLoader:
        return self.dataloader

    def _split_datalist(self) -> list[dict]:
        return self.data_dicts[:2]


if __name__ == "__main__":
    data_dir = "path/to/your/mhd/data"
    dataset = MhdDataset(
        data_path=data_dir,
        image_size=(128, 128, 128),
        batch_size=2,
        num_workers=4,
        is_train=True,
    )

    dataloader = dataset.get_dataloader()

    # 测试数据加载
    for batch in dataloader:
        images = batch["image"]
        labels = batch["label"]
        print(f"Image shape: {images.shape}")
        print(f"Label shape: {labels.shape}")
        break
