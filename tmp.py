from core.trainer.hooks import (
    Hook, OptimizerHook, SchedulerHook, EmptyCacheHook)

from core.trainer.log_meter import LossMeter

def summary_writer():
    from torch.utils.tensorboard import SummaryWriter
    sm = SummaryWriter('./')
    # sm.add_text('a', 'b', '123')
    for i in range(10):
        sm.add_scalar("LR/param_group_{}".format(1), i*10, i)
        sm.add_text('aa', 'bb', i)

    sm.close()
    print(SummaryWriter)

def loss_meter_test():
    loss_meter = LossMeter()
    loss_meter.update({'a':1, 'b':2})
    loss_meter.update({'a':3, 'b':4})
    print(loss_meter)
    loss_meter.clear()
    print(loss_meter)

class dummy_trainer:
    def __init__(self):
        losses = dict(loss_1=0, loss_2=10)
        self.outputs = dict(multi_loss=losses)
        self.meter = LossMeter()
    
    def gen_loss(self):
        self.outputs['multi_loss']['loss_1'] += 1
        self.outputs['multi_loss']['loss_2'] += 1

def loss_logger_test():
    from core.trainer.hooks.logger import LossLoggerHook
    loss_hook = LossLoggerHook()
    trainer = dummy_trainer()
    print(trainer.meter)

    loss_hook.after_train_iter(trainer)
    print(trainer.meter)

    trainer.gen_loss()
    loss_hook.after_train_iter(trainer)
    print(trainer.meter)

    loss_hook.before_train_epoch(trainer)
    print(trainer.meter)

if __name__ == "__main__":
    print(Hook, OptimizerHook, SchedulerHook, EmptyCacheHook)

    loss_logger_test()

    # from collections import defaultdict
