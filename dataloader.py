from pathlib import Path
from torchio.transforms import (
    RandomFlip,
    RandomAffine,
    RandomElasticDeformation,
    RandomNoise,
    RandomMotion,
    RandomBiasField,
    RescaleIntensity,
    Resample,
    RandomSwap,
    ToCanonical,
    ZNormalization,
    CropOrPad,
    HistogramStandardization,
    OneOf,
    Compose,
)
from torchio.data import UniformSampler, LabelSampler
from torchio import ScalarImage, LabelMap, Subject, SubjectsDataset, Queue
import torchio
from torchio import AFFINE, DATA
import torchio as tio
import torch
import sys

sys.path.append('./')


def get_subjects(conf):
    '''
    @description: get the subjects for normal training
    '''
    subjects = []
    if 'predict' in conf.file_name:
        img_path = Path(conf.pred_data_path)
        gt_path = Path(conf.pred_gt_path)
    else:
        img_path = Path(conf.data_path)
        gt_path = Path(conf.gt_path)
    x_generator = sorted(img_path.glob('*.mhd'))
    gt_generator = sorted(gt_path.glob('*.mhd'))
    for i, (source, gt) in enumerate(zip(x_generator, gt_generator)):
        subject = tio.Subject(
            source=tio.ScalarImage(source),
            gt=tio.LabelMap(gt),
        )
        subjects.append(subject)
    return subjects


class Dataset(torch.utils.data.Dataset):

    def __init__(self, conf):
        self.subjects = []

        queue_length = 10
        samples_per_volume = 10

        self.subjects = get_subjects(conf)

        self.transforms = self.transform(conf)

        self.training_set = tio.SubjectsDataset(self.subjects, transform=self.transforms)

        self.queue_dataset = Queue(self.training_set, queue_length, samples_per_volume, UniformSampler(patch_size=conf.patch_size), num_workers=16)

    def transform(self, conf):

        if conf.aug:
            training_transform = Compose([
                # CropOrPad((hp.crop_or_pad_size), padding_mode='reflect'),
                # RandomMotion(),
                RandomBiasField(),
                ZNormalization(),
                RandomNoise(),
                RandomFlip(axes=(0, )),
                OneOf({
                    RandomAffine(): 0.8,
                    RandomElasticDeformation(): 0.2,
                }),
            ])
        else:
            if 'train' in conf.file_name:
                training_transform = Compose([
                    # CropOrPad((hp.crop_or_pad_size), padding_mode='reflect'),
                    # RandomAffine(degrees=20),
                    # RandomNoise(std=0.0001),
                    ZNormalization(),
                ])
            elif 'predict' in conf.file_name:
                training_transform = Compose([
                    ZNormalization(),
                ])
            else:
                training_transform = Compose([
                    # CropOrPad((hp.crop_or_pad_size), padding_mode='reflect'),
                    # RandomAffine(degrees=20),
                    # RandomNoise(std=0.0001),
                    ZNormalization(),
                ])
        return training_transform
