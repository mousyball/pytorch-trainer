import torch

from pytorch_trainer.utils.config import parse_yaml_config
from networks.classification.builder import build_network


def test_train_inference(cfg_path):
    # Case: Gerneral definition of network
    cfg = parse_yaml_config(cfg_path)

    if cfg.get('NETWORK'):
        n_class = cfg.get('NETWORK').get('BACKBONE').get('NUM_CLASS')
    elif cfg.get('CUSTOM'):
        n_class = cfg.get('CUSTOM').get('MODEL').get('NUM_CLASS')
    else:
        assert False

    net = build_network(cfg=cfg)
    net.train()
    x = torch.rand(4, 3, 32, 32)
    y = torch.randint(low=0, high=n_class, size=(4,))
    print('[LOSS][OUTPUT]', net.train_step((x, y)))
    print('[LOSS][PARAMS]', net.criterion.__dict__, '\n')


if __name__ == "__main__":
    # Case: Gerneral definition of network
    cfg_path = "./configs/networks/classification/lenet.yaml"
    print(f"[INFO] config: {cfg_path}")
    test_train_inference(cfg_path)

    # Case: Custom definition of network
    cfg_path = "./configs/networks/classification/mynet.yaml"
    print(f"[INFO] config: {cfg_path}")
    test_train_inference(cfg_path)
