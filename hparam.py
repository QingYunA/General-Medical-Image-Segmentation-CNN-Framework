class hparams:

    def __init__(self):

        #! must set in this file
        self.use_which_net = 'er_net'  #* unet,res_unet,er_net,re_net
        self.scheduler_step_size = 20
        self.scheduler_gamma = 0.8
        self.save_arch = '.mhd'  #* the file format saved when training

        self.dataset = 'else'  #* 'brats19' or 'else'
        self.aug = False  # True or False
        self.network_arch = ''
        self.latest_checkpoint_file = 'latest_checkpoint.pt'
        self.total_epochs = 100
        self.epochs_per_checkpoint = 25
        self.batch_size = 4
        self.ckpt = None
        self.init_lr = 0.01
        self.debug = False  # False True
        # self.mode = '2d'  # '2d or '3d'
        self.in_class = 1
        self.out_class = 1
        # used in data_augment
        self.crop_or_pad_size = 64, 64, 64  # if 2D: 256,256,1  #USELESS
        self.patch_size = 64, 64, 64  # if 2D: 128,128,1
        self.load_mode = 0  #* 0: load nothing 1: load from checkpoint 2: load from pre_trained model(supervised)

        #* for self-supervised random swap size
        self.swap_size = 20
        self.swap_iters = 20

        # for test
        # self.patch_overlap = 4, 4, 4  # if 2D: 4,4,0

        self.fold_arch = '*.mhd'

        self.data_path = '/data/cc/Ying-TOF/train/source'
        self.gt_path = '/data/cc/Ying-TOF/train/label1'
        self.pred_data_path = ''
        self.pred_gt_path = ''
        self.pred_path = ''
        # Constants
        self.output_dir = 'logs/' + self.network_arch  # checkpoint_latest save path

        self.init_type = 'xavier'  # ['normal', 'xavier', 'xavier_uniform', 'kaiming', 'orthogonal', 'none]
