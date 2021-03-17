from typing import List

import cv2
import albumentations as A

from ..builder import TRANSFORMS
from ..transform import BaseTransform


@TRANSFORMS.register(name='AlbumRandomAffine')
class AlbumRandomAffine(BaseTransform):
    def __init__(self,
                 degree: List = [-5, +5],
                 prob: float = 0.5,
                 bbox_format: str = 'pascal_voc',
                 border_mode: int = cv2.BORDER_REFLECT_101,
                 value: int = 0,
                 input_key=('image', 'bboxes'),
                 output_key=('image', 'bboxes'),
                 visualize=False) -> None:
        """Albumentations Random Affine.

        Args:
            degree (List, optional): rotated degree. Defaults to [-5, +5].
            prob (float, optional): activated probability. Defaults to 0.5.
            bbox_format (str, optional): bbox format.
                Check albumentaion docs for more options. Defaults to 'pascal_voc'.
            border_mode (int, optional): padding mode.
                Check albumentations docs for more options. Defaults to cv2.BORDER_REFLECT_101.
            value (int, optional): padding value. Defaults to 0.
            input_key (tuple, optional): input status. Defaults to ('image', 'bboxes').
            output_key (tuple, optional): output status. Defaults to ('image', 'bboxes').
            visualize (bool, optional): visualize transform. Defaults to False.
        """
        super().__init__(input_key=input_key,
                         output_key=output_key,
                         visualize=visualize)
        self._degree = degree
        self._prob = prob
        self._bbox_format = bbox_format
        self._border_mode = border_mode
        self._value = value

        self._transform = self._transform_function(input_key)

    def _transform_function(self, input_key):
        """Albumentations tranform.

        Returns:
            :obj:`A.Compose`: A wrapped composed function.
        """
        bbox_params = None
        if 'bboxes' in input_key:
            bbox_params = A.BboxParams(format=self._bbox_format,
                                       label_fields=[])

        transform = A.Compose(
            [A.Rotate(limit=self._degree,
                      border_mode=self._border_mode,
                      value=self._value,
                      p=self._prob)],
            bbox_params=bbox_params
        )

        return transform

    def __repr__(self):
        return '{0}: degree={1}, prob={2}, bbox_format={3}, border_mode={4}, padding_value={5}'.format(
            self.__class__.__name__,
            self._degree,
            self._prob,
            self._bbox_format,
            self._border_mode,
            self._value
        )


@TRANSFORMS.register(name='AlbumRandomScale')
class AlbumRandomScale(BaseTransform):
    def __init__(self,
                 scale_limit: List[float] = [0.75, 1.25],
                 interpolation: int = cv2.INTER_LINEAR,
                 prob: float = 0.5,
                 input_key=('image', 'mask'),
                 output_key=('image', 'mask'),
                 visualize=False) -> None:
        """Albumentations Random Scale.

        Args:
            scale_limit (List[float], optional): scaling range. Defaults to [0.75, 1.25].
            interpolation (int, optional): interpolation mode in opencv. Defaults to cv2.INTER_LINEAR.
            prob (float, optional): activated probability. Defaults to 0.5.
            input_key (tuple, optional): input status. Defaults to ('image', 'mask').
            output_key (tuple, optional): output status. Defaults to ('image', 'mask').
            visualize (bool, optional): visualize transform. Defaults to False.
        """
        super().__init__(input_key=input_key,
                         output_key=output_key,
                         visualize=visualize)
        self._scale_limit = scale_limit
        self._interpolation = interpolation
        self._prob = prob

        self._transform = self._transform_function(input_key)

    def _transform_function(self, input_key):
        """Albumentations tranform.

        Returns:
            :obj:`A.Compose`: A wrapped composed function.
        """
        bbox_params = None
        if 'bboxes' in input_key:
            bbox_params = A.BboxParams(format=self._bbox_format,
                                       label_fields=[])

        transform = A.Compose(
            [A.RandomScale(scale_limit=self._scale_limit,
                           interpolation=self._interpolation,
                           p=self._prob)],
            bbox_params=bbox_params
        )

        return transform

    def __repr__(self):
        return '{0}: scale_limit={1}, prob={2}, interpolation_mode={3}'.format(
            self.__class__.__name__,
            self._scale_limit,
            self._prob,
            self._interpolation
        )


@TRANSFORMS.register(name='AlbumResize')
class AlbumResize(BaseTransform):
    def __init__(self,
                 height: int = 512,
                 width: int = 512,
                 interpolation: int = cv2.INTER_LINEAR,
                 prob: float = 1,
                 input_key=('image', 'mask'),
                 output_key=('image', 'mask'),
                 visualize=False) -> None:
        """Albumentations Resize.

        Args:
            height (int, optional): target height. Defaults to 512.
            width (int, optional): target width. Defaults to 512.
            interpolation (int, optional): interpolation mode in opencv. Defaults to cv2.INTER_LINEAR.
            prob (float, optional): activated probability. Defaults to 1.
            input_key (tuple, optional): input status. Defaults to ('image', 'mask').
            output_key (tuple, optional): output status. Defaults to ('image', 'mask').
            visualize (bool, optional): visualize transform. Defaults to False.
        """
        super().__init__(input_key=input_key,
                         output_key=output_key,
                         visualize=visualize)
        self._height = height
        self._width = width
        self._interpolation = interpolation
        self._prob = prob

        self._transform = self._transform_function(input_key)

    def _transform_function(self, input_key):
        """Albumentations tranform.

        Returns:
            :obj:`A.Compose`: A wrapped composed function.
        """
        bbox_params = None
        if 'bboxes' in input_key:
            bbox_params = A.BboxParams(format=self._bbox_format,
                                       label_fields=[])

        transform = A.Compose(
            [A.Resize(height=self._height,
                      width=self._width,
                      interpolation=self._interpolation,
                      p=self._prob)],
            bbox_params=bbox_params
        )

        return transform

    def __repr__(self):
        return '{0}: height={1}, width={2}, interpolation={3}, prob={4}'.format(
            self.__class__.__name__,
            self._height,
            self._width,
            self._interpolation,
            self._prob
        )
