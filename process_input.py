import torch


def process_x(config, batch):
    x = batch["source"]["data"]
    return x


def process_gt(config, batch):
    gt = batch["gt"]["data"]
    return gt
