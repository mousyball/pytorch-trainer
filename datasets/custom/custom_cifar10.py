from typing import Any, Dict, Callable, Optional

from torchvision import datasets

from ..builder import DATASETS


@DATASETS.register(name='custom_cifar10')
class CustomTransform(datasets.CIFAR10):
    def __init__(self,
                 root: str,
                 train: bool = True,
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None,
                 download: bool = False,
                 ) -> None:
        super().__init__(root=root,
                         train=train,
                         transform=transform,
                         target_transform=target_transform,
                         download=download)

    def __getitem__(self, index: int) -> Dict[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            dict: (inputs:tensor, targets:tensor)
        """
        img, target = self.data[index], self.targets[index]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return dict(inputs=img,
                    targets=target)
