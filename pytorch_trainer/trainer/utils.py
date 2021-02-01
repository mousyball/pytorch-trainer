import os
import logging
import os.path as osp
import datetime
from socket import gethostname
from getpass import getuser


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
