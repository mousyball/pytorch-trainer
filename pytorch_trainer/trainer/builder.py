from datasets.builder import build_dataloader
from transforms.builder import build_custom_transform
from pytorch_trainer.utils import parse_yaml_config
from pytorch_trainer.trainer.utils import (
    get_device, get_logger, create_work_dir, set_random_seed,
    load_pretained_weight
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
    work_dir = create_work_dir(config.trainer.work_dir)

    # logger
    logger = get_logger(work_dir)

    # set random seed
    seed = config.trainer.seed
    deterministic = config.trainer.deterministic
    set_random_seed(logger,
                    seed=seed,
                    deterministic=deterministic)

    # create transform
    train_transform = build_custom_transform(config.transform)
    val_transform = train_transform
    transform = [train_transform, val_transform]

    # create dataloader
    dataloader = build_dataloader(config, transform)

    # get device
    gpu_ids = config.trainer.gpu_ids
    device = get_device(logger,
                        gpu_ids=gpu_ids,
                        deterministic=deterministic)

    # model
    model = build_network(config)  # Net()

    # load pretrained weight
    if config.trainer.pretrain.load_weight:
        dir_path = config.trainer.pretrain.dir_path
        weight_name = config.trainer.pretrain.weight_name
        load_pretained_weight(model,
                              dir_path,
                              weight_name,
                              logger)

    # optimizer
    lr = config.optimizer.params.lr
    if config.get('network'):
        group_info = config.network.group_info
    elif config.get('custom'):
        group_info = config.custom.group_info
    else:
        raise ValueError
    params_group = model.get_optimizer_params(group_info, lr)
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

    return trainer, dataloader, config.trainer.workflow
