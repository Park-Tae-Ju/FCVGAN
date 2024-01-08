import albumentations as A
from albumentations.pytorch import ToTensorV2

from common.utils.dir_utils import make_folder
import os
import os.path as osp


class Config:
    project_name = 'default'
    model_names = [x + '_default' for x in ['GA', 'GB', 'DA', 'DB']]
    # directories
    cur_dir = osp.dirname(os.path.abspath(__file__))
    root_dir = cur_dir
    output_dir = osp.join(root_dir, 'output')
    model_dir = osp.join(output_dir, 'model_dump', project_name)
    vis_dir = osp.join(output_dir, 'vis')

    gpu_ids = [0]
    continue_train = False

    # hyper parameter
    batch_size = 4
    # batch_size = 1
    num_thread = 8
    # num_thread = 1

    lr = 2e-4
    beta1 = 0.5
    lr_policy = 'linear'

    n_epochs = 100
    n_epochs_decay = 100
    lr_decay_iters = 50
    # for test
    load_epochs = 153

    # related with layer
    ngf = 64
    ndf = 64
    no_dropout = True
    pool_size = 50
    gan_mode = 'lsgan'
    norm = 'instance'
    preprocess = 'resize_and_crop'
    lambda_identity = 0.5
    lambda_A = 10
    lambda_B = 10

    netG = 'fcv'
    netD = 'basic'

    crop_size = 256

    # related with init
    init_type = 'normal'
    init_gain = 0.02

    # augmentation
    data_transforms = {
        "train": A.Compose([
            A.Resize(crop_size, crop_size),
            # A.HorizontalFlip(),
            A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            ToTensorV2()
        ]),
        "test": A.Compose([
            A.Resize(crop_size, crop_size),
            A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            ToTensorV2()
        ]),
    }


cfg = Config
