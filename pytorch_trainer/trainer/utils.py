import os
import random
import logging
import os.path as osp
import datetime
from socket import gethostname
from getpass import getuser

import numpy as np
import torch

from pytorch_trainer.trainer.profiling import bcolors


def get_host_info():
    return f'{getuser()}@{gethostname()}'


def get_logger(path='log/', file_name=None, logger_name='trainer'):
    # get logger
    logger = logging.getLogger(logger_name)

    # format logging message
    formater = "[%(levelname)-8s][%(asctime)s][%(name)-8s]: %(message)s"
    formater = logging.Formatter(formater)

    # logging file name
    date_string = datetime.datetime.now().strftime("%Y-%m-%d.log")
    if file_name is None:
        log_filename = date_string
    else:
        log_filename = '{0}_{1}'.format(file_name, date_string)

    if path is not None:
        # make directory
        if not osp.isdir(path):
            os.makedirs(path)
        log_filename = os.path.join(path, log_filename)

        # init file handler
        file_handler = logging.FileHandler(
            log_filename, mode='a', encoding='utf-8')
        file_handler.setFormatter(formater)
        logger.addHandler(file_handler)

    # init stream handler
    console = logging.StreamHandler()
    console.setFormatter(formater)
    logger.addHandler(console)

    logger.setLevel(logging.INFO)

    return logger


class IterDataLoader:

    def __init__(self, dataloader):
        self._dataloader = dataloader
        self.iter_loader = iter(self._dataloader)

    def __next__(self):
        try:
            data = next(self.iter_loader)
        except StopIteration:
            self.iter_loader = iter(self._dataloader)
            data = next(self.iter_loader)

        return data

    def __len__(self):
        return len(self._dataloader)


def sync_counter(func):
    """make sure all iteration and epoch have same value in all hooks"""
    # TODO: write example or document
    def wrap(*args, **kwargs):
        args[0]._iter -= 1
        args[0]._epoch -= 1
        func(*args, **kwargs)
        args[0]._iter += 1
        args[0]._epoch += 1

    return wrap


def set_random_seed(logger, seed, deterministic=False):
    """Set random seed.

    Args:
        logger (:obj:`logging`): logger.
        seed (int): Seed to be used.
        deterministic (bool, optional): Whether to get the deterministic result. Defaults to False.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    logger.warn(f"Using GLOBAL SEED!!! (seed_num={seed})")


def get_device(logger, gpu_ids=[0], deterministic=False):
    """Get CPU or GPU device.

    Args:
        logger (:obj:`logging`): logger
        gpu_ids (List[int]): id of gpus are used for training. Defaults to [0].
        deterministic (bool, optional): deterministic mode in training. Defaults to False.

    Returns:
        torch.device: device for training
    """
    device = torch.device("cuda:"+str(gpu_ids[0])
                          if torch.cuda.is_available()
                          else "cpu")

    if torch.cuda.is_available():
        logger.info(
            f"Using device - GPUs: {bcolors.OKGREEN}{gpu_ids}{bcolors.ENDC}, "
            + f"Main GPU: {bcolors.OKGREEN}{gpu_ids[0]}{bcolors.ENDC}"
        )
    else:
        logger.error(
            "CUDA is not available.\n\n"
            + "Please check:\n"
            + "  1. Is GPU available?\n"
            + "  2. Does CUDA version match the Nvidia driver version?\n"
            + "  3. Is Nvidia driver off-line?\n"
        )
        exit()

    # [NOTE] True for static network, False for dynamic network
    # [NOTE] Turn off if you need a deterministic result.
    torch.backends.cudnn.benchmark = False if deterministic else True

    return device


def create_work_dir(work_dir):
    """Create working directory.

    Args:
        work_dir (str): path to working directory.

    Returns:
        str: modified working directory path with datetime.
    """
    if isinstance(work_dir, str):
        date = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        work_dir = osp.join(work_dir, '')[:-1] + f'_{date}'
        work_dir = osp.abspath(work_dir)
        if not osp.isdir(work_dir):
            os.makedirs(work_dir)
    elif work_dir is None:
        raise ValueError('"work_dir" can not be None.')
    else:
        raise TypeError('Argument "work_dir" must be a string.')

    return work_dir


def load_pretained_weight(model, dir_path, weight_name, logger):
    """Load pretrained weight from path.

    Args:
        model (nn.module): built model
        dir_path (str): directory path to weight file
        model_name (str): name of pretrained weight
        logger (:obj:`logging`): logger
    """
    weight_path = osp.join(dir_path, weight_name)
    if not os.path.isfile(weight_path):
        assert False, "Path to pretrained weight does not exist."

    state_dict = torch.load(weight_path,
                            map_location=lambda storage, loc: storage)

    # [NOTE] Strict mode means the weight should totally match to the model.
    unexpected, missing = model.load_state_dict(state_dict, strict=True)
    if unexpected:
        # ckpt key name not exist in networks
        # [EXAMPLE] network(backbone) + weight(backbone+psp) => unexpected keys (psp)
        logger.info("===unexpected===\n{0}".format(unexpected))
    if missing:
        # networks key name not exist in ckpt
        # [EXAMPLE] network(backbone+psp) + weight(backbone) => missing keys (psp)
        logger.info("===missing===\n{0}".format(missing))

    # [TODO] Handle 'module' prefix of loading distributed weight.
