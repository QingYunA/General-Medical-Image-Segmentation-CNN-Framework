# Copyright (c) 2022 Huawei Technologies Co., Ltd.
# Licensed under CC BY-NC-SA 4.0 (Attribution-NonCommercial-ShareAlike 4.0 International) (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode
#
# The code is released for academic research use only. For commercial use, please contact Huawei Technologies Co., Ltd.
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# This repository was forked from https://github.com/openai/guided-diffusion, which is under the MIT license

from functools import lru_cache
import os
from utils import imwrite

from collections import defaultdict
from os.path import isfile, expanduser


def to_file_ext(img_names, ext):
    img_names_out = []
    for img_name in img_names:
        splits = img_name.split('.')
        if not len(splits) == 2:
            raise RuntimeError("File name needs exactly one '.':", img_name)
        img_names_out.append(splits[0] + '.' + ext)

    return img_names_out


def write_images(imgs, img_names, dir_path):
    os.makedirs(dir_path, exist_ok=True)

    for image_name, image in zip(img_names, imgs):
        out_path = os.path.join(dir_path, image_name)
        imwrite(img=image, path=out_path)


class NoneDict(defaultdict):

    def __init__(self):
        super().__init__(self.return_None)

    @staticmethod
    def return_None():
        return None

    def __getattr__(self, attr):
        return self.get(attr)


class Default_Conf(NoneDict):

    def __init__(self):
        pass

    def get_default_eval_name(self):
        candidates = self['data']['eval'].keys()
        if len(candidates) != 1:
            raise RuntimeError(f"Need exactly one candidate for {self.name}: {candidates}")
        return list(candidates)[0]

    def pget(self, name, default=None):
        if '.' in name:
            names = name.split('.')
        else:
            names = [name]

        sub_dict = self
        for name in names:
            sub_dict = sub_dict.get(name, default)

            if sub_dict == None:
                return default

        return sub_dict

    def update_from_args(self, args_dict):
        
        for key, value in args_dict.items():
            if value:
                self.update({key: value})

        if type(self.patch_size) == str:
            assert len(self.patch_size.split(',')) <= 3, f'patch size can only be one str or three str but got {len(self.patch_size.split(","))}'
            if len(self.patch_size.split(',')) == 3:
                self.patch_size = tuple(map(int, self.patch_size.split(',')))
            else:
                self.patch_size = int(self.patch_size)