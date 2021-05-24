from pytorch_trainer.utils import parse_yaml_config
from pytorch_trainer.trainer.utils import (
    get_device, get_logger, create_work_dir, set_random_seed
)
from pytorch_trainer.utils.builder import build
from networks.classification.builder import build_network
from pytorch_trainer.optimizers.builder import build_optimizer
from pytorch_trainer.schedulers.builder import build_scheduler

from .base_trainer import TRAINER


def build_trainer(cfg, **kwargs):
    return build(cfg, TRAINER, **kwargs)


def build_trainer_api(cfg_path):
    # load config
    config = parse_yaml_config(cfg_path)

    # create working directory
    work_dir = './dev/trainer/'
    work_dir = create_work_dir(work_dir)

    # logger
    logger = get_logger(work_dir)

    # set random seed
    seed = config.trainer.seed
    deterministic = config.trainer.deterministic
    set_random_seed(logger,
                    seed=seed,
                    deterministic=deterministic)

    # get device
    gpu_ids = config.trainer.gpu_ids
    device = get_device(logger,
                        gpu_ids=gpu_ids,
                        deterministic=deterministic)

    # model
    model = build_network(config)  # Net()

    # optimizer
    lr = config.optimizer.params.lr
    params_group = [{'params': model.get_lr_params(['backbone']), 'lr': lr * 1},
                    {'params': model.get_lr_params(['criterion']), 'lr': lr * 10}]
    optimizer = build_optimizer(params_group, config.optimizer)

    # scheduler
    scheduler = build_scheduler(optimizer, config.scheduler)

    # initial trainer
    trainer = build_trainer(config.trainer,
                            model=model,
                            optimizer=optimizer,
                            scheduler=scheduler,
                            device=device,
                            work_dir=work_dir,
                            logger=logger,
                            meta=config.trainer.meta)

    # register all callback
    trainer.register_callback(config)

    return trainer, config.trainer.workflow
