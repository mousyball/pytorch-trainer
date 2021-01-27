import torch

from pytorch_trainer.utils.config import parse_yaml_config
from networks.classification.builder import build_network


def test_inference(cfg_path):
    # Case: Gerneral definition of network
    cfg = parse_yaml_config(cfg_path)
    net = build_network(cfg=cfg)
    print(net)
    net.eval()
    x = torch.rand(4, 3, 32, 32)
    print(net(x), '\n')


if __name__ == "__main__":
    # Case: Gerneral definition of network
    cfg_path = "./configs/networks/classification/lenet.yaml"
    print(f"[INFO] config: {cfg_path}")
    test_inference(cfg_path)

    # Case: Custom definition of network
    cfg_path = "./configs/networks/classification/mynet.yaml"
    print(f"[INFO] config: {cfg_path}")
    test_inference(cfg_path)
