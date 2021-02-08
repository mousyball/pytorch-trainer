# priority

## priority table

* The priority table is designed for controlling hook execution order
    * HIGHEST means execution first
* HIGHEST, VERY_HIGH, HIGH: hooks which requests data from the trainer before data been overwrite
* NORMAL: orderless hook
* LOW, VERY_LOW, LOWEST: hooks which changes the trainer data

| Level     |                 Hooks                 |
|-----------|:-------------------------------------:|
| HIGHEST   |                                       |
| VERY_HIGH | LossLoggerHook                        |
| HIGH      | TensorboardLoggerHook, TextLoggerHook |
| NORMAL    | CheckpointHook                        |
| LOW       | OptimizerHook                         |
| VERY_LOW  | SchedulerHook                         |
| LOWEST    |                                       |

## Hook Brief

* OptimizerHook: loss backward and optimizer step
* SchedulerHook: scheduler step
* CheckpointHook: save checkpoint
* LossLoggerHook: log model output losses
* TensorboardLoggerHook: log losses and learning rate into tensorboard
* TextLoggerHook: log losses and learning rate on console

## Design


