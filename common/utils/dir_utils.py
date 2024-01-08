import os
import sys
import yaml
from easydict import EasyDict as edict


def make_folder(folder_name):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)


def add_pypath(path):
    if path not in sys.path:
        sys.path.insert(0, path)


def update_config(config_file, config):
    with open(config_file) as f:
        override_config = edict(yaml.load(f, Loader=yaml.FullLoader))
        config.project_name = override_config.PROJECT_NAME
        config.model_dir = os.path.join(config.output_dir, 'model_dump', config.project_name)

        # model
        config.model_names = [x + f'_{override_config.PROJECT_NAME}' for x in ['GA', 'GB', 'DA', 'DB']]
        config.netG = override_config.MODEL.netG
        config.netD = override_config.MODEL.netD

        # optimizer
        config.lambda_A = float(override_config.OPTIMIZER.lambda_A)
        config.lambda_B = float(override_config.OPTIMIZER.lambda_B)
        config.lambda_identity = float(override_config.OPTIMIZER.lambda_identity)

        # hyper param
        config.lr = float(override_config.OPTIMIZER.lr)
        config.beta1 = float(override_config.OPTIMIZER.beta1)
        config.batch_size = int(override_config.TRAINING.batch_size)
        config.n_epochs = int(override_config.TRAINING.n_epochs)

    # Make folder
    make_folder(config.output_dir)
    make_folder(config.model_dir)
    make_folder(config.vis_dir)
    return config