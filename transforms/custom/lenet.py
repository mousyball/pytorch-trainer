import torchvision.transforms as transforms

from ..builder import CUSTOM_TRANSFORMS


class BaseCustomTransform:
    def __init__(self) -> None:
        self._train_transforms = None
        self._test_transforms = None

    def get_train_transforms(self):
        return self._train_transforms

    def get_test_transforms(self):
        return self._test_transforms


@CUSTOM_TRANSFORMS.register(name='LeNet')
class BaseLenet(BaseCustomTransform):
    def __init__(self) -> None:
        self._train_transforms = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        )

        self._test_transforms = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        )
