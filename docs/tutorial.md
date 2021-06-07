# Tutorial of setup to trainer

## Introduction

In this repo, two type of trainer are supported, namely `epoch-based` and `iteration-based` trainer. You can use one of them on demand.

All of the parameters are controlled by config yaml. The structure of config is hierarchical inheritance which starts from trainer, components, dataset, transfrom to network settings. We recommend you to override the parent setting at the leaf config in `configs/custom` directory.

Default configs could be checked in `configs/` directory and the final config is located at `configs/custom/`.

## Tutorials

In the following steps, we will show you how to go through all of the settings.

### 1. Trainer

Pick up the trainer you want to use. In practical use, iteration trainer usually is used for large dataset whose epoch contains a huge amount of batches.

* `configs/pytorch_trainer/epoch_trainer.yaml`
* `configs/pytorch_trainer/iter_trainer.yaml`

### 2. Transform

* Program your transform at `transform/custom/<your_transform>.py`
* Register customized transform class into `TRANSFORMS` registry.
* Call by its name in config.

```yaml
transform:
  name: 'LeNet'
```

### 3. Dataset

* Program your transform at `datasets/custom/<your_dataset>.py`
* Register customized transform class into `DATASETS` registry. Make sure to follow the rule of pytorch dataset.
* Inherit your custom transform.
* Setup the parameters of dataset and dataloader in train and val respectively.

```yaml
_BASE_: ../transforms/custom_transform.yaml

dataset:
  train:
    name: custom_cifar10
    params:
      root: "./dev/data"
      train: True
      download: True
  val:
    name: custom_cifar10
    params:
      root: "./dev/data"
      train: False
      download: False

dataloader:
  train:
    name: dataloader
    params:
      batch_size: 4
      shuffle: False
      num_workers: 2
  val:
    name: dataloader
    params:
      batch_size: 4
      shuffle: False
      num_workers: 2
```

### 4. Network

* Program your network at `networks/<task>/customs/<your_network>.py`.
* Register customized network class into `CUSTOMS` registry.
* Setup your network settings in `configs/networks/<task>/<your_config>.yaml`

```yaml
custom:
  name: 'MyLeNet'
  model:
    name: 'LeNet'
    num_class: 10
  loss:
    name: 'MyLeNetLoss'
    weight: null
    reduction: 'mean'
    ignore_index: -87
  group_info:
    - [[model], 1]
```

### 5. Loss

* Program your network at `networks/loss/<your_loss>.py`.
* Register customized network class into `LOSSES` registry.
* Setup your loss settings in `configs/networks/<task>/<your_config>.yaml` described in preceding network settings.

### 6. Gather all configs

* Finalize your config at `configs/custom/<task>/<your_network>/<final_config>.yaml` which should include the following configs.
  * `trainer.yaml`
  * `dataset.yaml` included `transform.yaml`
  * `network.yaml` included `loss setting`
* Then, override the parent setting according to your training need in the final config.
* At last, enjoy the training if everything works fine.
