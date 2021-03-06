import os.path as osp

from pytorch_trainer.utils.config import parse_yaml_config

ROOT_PATH = './tests/components/'


class TestConfig:
    def test_yaml_parser(self):
        # [Case] base.yaml
        PATH = './configs/base.yaml'
        cfg_base = parse_yaml_config(osp.join(ROOT_PATH, PATH))

        assert cfg_base.KEY1 == 'base'
        assert cfg_base.KEY2 == 'base'

    def test_yaml_parser_with_single_base(self):
        # [Case] config.yaml inherits from base.yaml
        PATH = './configs/config.yaml'
        cfg = parse_yaml_config(osp.join(ROOT_PATH, PATH), allow_unsafe=True)

        assert cfg.KEY1 == 'base'
        assert cfg.KEY2 == 'config'
        assert cfg.EXPRESSION == [1, 4, 9]

    def test_yaml_parser_with_multiple_base(self):
        # [Case] config.yaml inherits from multiple yaml
        PATH = './configs/config_inherit.yaml'
        cfg = parse_yaml_config(osp.join(ROOT_PATH, PATH), allow_unsafe=True)

        assert cfg.KEY1 == 'base1'
        assert cfg.KEY2 == 'config'
        assert cfg.EXPRESSION == [1, 4, 9]
