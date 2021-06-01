import argparse

from pytorch_trainer.trainer import build_trainer_api


def argparser():
    parser = argparse.ArgumentParser(
        description='Trainer demo',
        formatter_class=lambda prog: argparse.HelpFormatter(
            prog,
            max_help_position=40
        )
    )

    parser.add_argument('-cfg',
                        '--config',
                        default='configs/pytorch_trainer/epoch_trainer.yaml',
                        type=str,
                        metavar='PATH',
                        help=r'config path')

    return parser.parse_args()


if __name__ == "__main__":
    parser = argparser()
    trainer, data_loader, workflow = build_trainer_api(parser.config)

    # training: demo will train 15K iteration(3 epoch), run validation 3 time and save 3 weight
    train_loader, val_loader = data_loader
    trainer.fit(data_loaders=[train_loader, val_loader],
                workflow=workflow)
