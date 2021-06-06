# pytorch-trainer

## Requirements

* `Python 3.7`

```bash
pip install -r requirements/runtime.txt
# For development
pip install -r requirements/dev.txt
```

### Pycocotool

```bash
pip install "git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI"
```

### Pre-commit Hook

* for development

```bash
pip install pre-commit
pre-commit install
```

### Config

* Fix the version of `fvcore` to `0.1.2.post20210128`
* Support multiple inheritance of config

## Demo

* Epoch-based trainer

```bash
python examples/train_cifar10.py -cfg configs/custom/classification/lenet/epoch_trainer.yaml
```

* Iteration-based trainer

```bash
python examples/train_cifar10.py -cfg configs/custom/classification/lenet/iter_trainer.yaml
```

## Get started

* Check the step-by-step [tutorial](./docs/tutorial.md) in the docs.
