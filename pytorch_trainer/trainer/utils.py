import os
import logging
import os.path as osp
import datetime
from socket import gethostname
from getpass import getuser


def get_host_info():
    return f'{getuser()}@{gethostname()}'


def get_logger(path='log/', file_name=None, logger_name='trainer'):
    # format logging message
    formater = "[%(levelname)-8s][%(asctime)s][%(name)-8s]: %(message)s"
    formater = logging.Formatter(formater)

    # logging file name
    date_string = datetime.datetime.now().strftime("%Y-%m-%d.log")
    if file_name is None:
        log_filename = date_string
    else:
        log_filename = '{0}_{1}'.format(file_name, date_string)

    # make directory
    if not osp.isdir(path):
        os.makedirs(path)
    log_filename = os.path.join(path, log_filename)

    # init file handler
    file_handler = logging.FileHandler(
        log_filename, mode='a', encoding='utf-8')
    file_handler.setFormatter(formater)

    # init stream handler
    console = logging.StreamHandler()
    console.setFormatter(formater)

    # get logger
    logger = logging.getLogger(logger_name)
    logger.addHandler(console)
    logger.addHandler(file_handler)
    logger.setLevel(logging.INFO)

    return logger
