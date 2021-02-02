from networks.loss import LOSSES
from networks.loss.regular import torch_loss


class TestLoss:
    def test_loss_list(self):
        print(LOSSES._obj_map.keys())
        loss_registry = set(LOSSES._obj_map.keys())
        loss_table = set(torch_loss.keys())
        assert loss_table.issubset(loss_registry)
