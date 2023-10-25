import torch


def process_x(args, batch):
    if args.dataset == 'brats':
        flair = batch['flair']['data']  #* shape[bs,1,h,w,d]
        t1 = batch['t1']['data']
        t1ce = batch['t1ce']['data']
        t2 = batch['t2']['data']
        x = torch.cat((flair, t1, t1ce, t2), dim=1)  #* shape [bs,4,h,w,d]
    else:
        x = batch['source']['data']
    return x


def process_gt(args, batch):
    if args.dataset == 'brats':
        flair_gt = batch['flair_gt']['data']
        t1_gt = batch['t1_gt']['data']
        t1ce_gt = batch['t1ce_gt']['data']
        t2_gt = batch['t2_gt']['data']
        gt = torch.cat((flair_gt, t1_gt, t1ce_gt, t2_gt), dim=1)  #* shape[bs,4,h,w,d]
    else:
        gt = batch['gt']['data']
    return gt
