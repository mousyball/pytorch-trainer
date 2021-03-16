from copy import deepcopy
from typing import List
from collections import OrderedDict

import cv2
import numpy as np
import torch
import albumentations as A

from .builder import TRANSFORMS
from .helpers import dextr_helper
from .visualization import TransformVisualization as TVis


class BaseTransform:
    def __init__(self,
                 input_key=('image', 'bboxes'),
                 output_key=('image', 'bboxes'),
                 visualize=False) -> None:
        self._input_key = input_key
        self._output_key = output_key
        self._transform = self._transform_function

        self.visualize = visualize
        if self.visualize:
            self.tvis = TVis()
        self._visualized_images = None

    @staticmethod
    def _get_input_keys(sample):
        """Remap keys to pre-defined keys in the albumentation.

        Args:
            sample (dict): intermediate status among transforms.

        Returns:
            dict: remapped dictionary.
        """
        image = sample.get('image', None)
        bboxes = sample.get('bboxes', None)
        mask = sample.get('mask', None)
        masks = sample.get('masks', None)
        keypoints = sample.get('keypoints', None)

        _dict = OrderedDict()
        if image is not None:
            _dict['image'] = image
        if bboxes is not None:
            _dict['bboxes'] = bboxes
        if mask is not None:
            _dict['mask'] = mask
        if masks is not None:
            _dict['masks'] = masks
        if keypoints is not None:
            _dict['keypoints'] = keypoints

        return _dict

    def _transform_function(self, *args, **kwargs):
        """Wrap a transform function.
        Note that the function should be wrapped by Albumentation compose.
        Coordinate boundary is handled by the compose functionality.
        """
        raise NotImplementedError()

    def visualization_hook(self, sample, transformed):
        """Hook to activate visualized functions.

        Args:
            sample (dict): input status
            transformed (dict): output status
        """
        if not self.visualize:
            return

        # Only visualize the pre-defined keys.
        self.tvis.apply_to_input(sample, None)
        for key in sample.keys():
            getattr(self.tvis, f'apply_to_{key}')(sample, transformed)

        self._visualized_images = self.tvis.visualized_images

    def __call__(self, sample) -> dict:
        """Transform flow.

        Args:
            sample (dict): input status

        Returns:
            dict: output status
        """
        # Get pre-defined keys.
        input_dict = self._get_input_keys(sample)

        # Transform image by the input keys.
        transformed = self._transform(**input_dict)

        # Visualize transformed result if needed.
        self.visualization_hook(sample, transformed)

        return transformed

    def clear_visualized_cache(self):
        """Clear visualized cache for next image."""
        self._visualized_images = None

    @property
    def visualized_images(self):
        """Get visualized images.

        Returns:
            dict: visualized images stored in a dictionary.
        """
        return self._visualized_images


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
        """Albumentation tranform.

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


@TRANSFORMS.register(name='AlbumRandomScale')
class AlbumRandomScale(BaseTransform):
    def __init__(self,
                 scale_limit: List[float] = [0.75, 1.25],
                 interpolation: int = cv2.INTER_LINEAR,
                 prob: float = 0.5,
                 input_key=('image', 'mask'),
                 output_key=('image', 'mask'),
                 visualize=False) -> None:
        super().__init__(input_key=input_key,
                         output_key=output_key,
                         visualize=visualize)
        self._scale_limit = scale_limit
        self._interpolation = interpolation
        self._prob = prob

        self._transform = self._transform_function(input_key)

    def _transform_function(self, input_key):
        """Albumentation tranform.

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


@TRANSFORMS.register(name='CustomCropFromMask')
class CustomCropFromMask(BaseTransform):
    def __init__(self,
                 relax: int = 50,
                 zero_pad: bool = True,
                 mask_key: str = 'mask',
                 input_key=('image', 'mask'),
                 output_key=('image', 'mask'),
                 visualize=False) -> None:
        super().__init__(input_key=input_key,
                         output_key=output_key,
                         visualize=visualize)
        self._relax = relax
        self._zero_pad = zero_pad
        self._mask_key = mask_key

    def apply_to_mask(self, sample, transformed):
        """Self-defined visualizd function.

        Args:
            sample (dict): input status
            transformed (dict): output status

        Returns:
            np.array: visualized image.
        """
        image = transformed['image'] / 255
        mask = transformed['mask'] / 255

        # Require floating image
        draw_image = dextr_helper.overlay_mask(
            im=image,
            ma=mask,
            colors=None,
            alpha=0.5
        ) * 255

        return draw_image

    def visualization_hook(self, sample, transformed):
        """Customized visualization layers.

        Args:
            sample (dict): input status
            transformed (dict): output status
        """
        if not self.visualize:
            return

        self.tvis.apply_to_input(sample, None)
        self.tvis.apply_to_image(None, transformed)
        self.tvis.update_custom_function(
            name='mask_is',
            func=self.apply_to_mask
        )
        self.tvis.apply_to_custom_function(
            sample=sample,
            transformed=transformed
        )
        self._visualized_images = self.tvis.visualized_images

    def __call__(self, sample) -> dict:
        """Custom transform flow.

        Args:
            sample (dict): input status

        Returns:
            dict: output status
        """
        transformed = deepcopy(sample) if self.visualize else sample

        _mask = sample[self._mask_key]
        if _mask.ndim == 2:
            _mask = np.expand_dims(_mask, axis=-1)

        for input_key, output_key in zip(self._input_key, self._output_key):
            _crop = []
            target = sample[input_key]
            if target.ndim == 2:
                target = np.expand_dims(target, axis=-1)

            for k in range(0, _mask.shape[-1]):
                if np.max(_mask[..., k]) == 0:
                    _crop.append(np.zeros(target.shape, dtype=target.dtype))
                else:
                    _crop.append(
                        dextr_helper.crop_from_mask(target,
                                                    _mask[..., k],
                                                    relax=self._relax,
                                                    zero_pad=self._zero_pad))
            if len(_crop) == 1:
                # make sure gt dimension is 2
                transformed[output_key] = np.squeeze(_crop[0])
            else:
                transformed[output_key] = _crop

        self.visualization_hook(sample, transformed)

        return transformed

    def __str__(self):
        return '{0}: relax={1} zero_pad={2}'.format(
            self.__class__.__name__,
            self._relax,
            self._zero_pad
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
        super().__init__(input_key=input_key,
                         output_key=output_key,
                         visualize=visualize)
        self._height = height
        self._width = width
        self._interpolation = interpolation
        self._prob = prob

        self._transform = self._transform_function(input_key)

    def _transform_function(self, input_key):
        """Albumentation tranform.

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


@TRANSFORMS.register(name='CustomExtremePoints')
class CustomExtremePoints(BaseTransform):
    def __init__(self,
                 pert: int = 5,
                 sigma: int = 10,
                 input_key=('mask'),
                 output_key=('extreme_points'),
                 visualize=False) -> None:
        """Generate the four extreme points (left, right, top, bottom) (with some random perturbation) in a given binary mask.

        Args:
            sigma (int): sigma of Gaussian to create a heatmap from a point
            pert (int): number of pixels fo the maximum perturbation
            input_key  (tuple[str]): input key
            output_key (tuple[str]): output key
            visualize (bool): visualize transform
        """
        super().__init__(input_key=input_key,
                         output_key=output_key,
                         visualize=visualize)
        self._pert = pert
        self._sigma = sigma

    def apply_to_extreme_points(self, sample, transformed):
        """Self-defined visualizd function.

        Args:
            sample (dict): input status
            transformed (dict): output status

        Returns:
            np.array: visualized image.
        """
        draw_image = transformed['image'].copy()
        mask = transformed['mask']
        extreme_points = transformed['extreme_points']

        # Overlay mask
        draw_image = draw_image * 0.5 + (mask[..., np.newaxis] * 255) * 0.5

        # Draw extreme points (with random disturbance).
        index = extreme_points > 0.5
        draw_image[index] = [255, 0, 0]

        return draw_image

    def visualization_hook(self, sample, transformed):
        """Customized visualization layers.

        Args:
            sample (dict): input status
            transformed (dict): output status
        """
        if not self.visualize:
            return

        self.tvis.apply_to_input(sample, None)
        self.tvis.update_custom_function(
            name='extreme_points',
            func=self.apply_to_extreme_points
        )
        self.tvis.apply_to_custom_function(
            sample=sample,
            transformed=transformed
        )
        self._visualized_images = self.tvis.visualized_images

    def __call__(self, sample) -> dict:
        """Custom transform flow.

        Args:
            sample (dict): input status

        Returns:
            dict: output status
        """
        transformed = deepcopy(sample) if self.visualize else sample

        for input_key, output_key in zip(self._input_key, self._output_key):
            _target = sample[input_key]
            if _target.ndim == 3:
                raise ValueError(
                    'ExtremePoints not implemented for multiple object per image.')
            if np.max(_target) == 0:
                # TODO: handle one_mask_per_point case
                transformed[output_key] = np.zeros(
                    _target.shape, dtype=_target.dtype
                )
            else:
                _points = dextr_helper.extreme_points(_target, self._pert)
                transformed[output_key] = dextr_helper.make_gt(
                    _target,
                    _points,
                    sigma=self._sigma,
                    one_mask_per_point=False
                )

        self.visualization_hook(sample, transformed)

        return transformed

    def __str__(self):
        return '{0} generate from {1} to {2}, pert={3}, sigma={4}'.format(
            self.__class__.__name__,
            self._input_key,
            self._output_key,
            self._pert,
            self._sigma
        )


@TRANSFORMS.register(name='CustomPoints')
class CustomPoints(BaseTransform):
    def __init__(self,
                 input_key=('image', 'bboxes'),
                 output_key=('points'),
                 visualize=False) -> None:
        super().__init__(input_key=input_key,
                         output_key=output_key,
                         visualize=visualize)

    def apply_to_points(self, sample, transformed):
        """Self-defined visualizd function.

        Args:
            sample (dict): input status
            transformed (dict): output status

        Returns:
            np.array: visualized image.
        """
        draw_image = transformed['image'].copy()

        for bbox in transformed['bboxes']:
            bbox_int = [int(pt) for pt in bbox[:4]]
            pt0 = tuple(bbox_int[:2])
            pt1 = tuple(bbox_int[2:])
            cv2.rectangle(draw_image, pt0, pt1, (0, 255, 0))

        for point in transformed['points']:
            point_int = [int(pt) for pt in point]
            cv2.circle(draw_image,
                       tuple(point_int),
                       radius=1,
                       color=(255, 0, 0),
                       thickness=4)

        return draw_image

    def visualization_hook(self, sample, transformed):
        """Customized visualization layers.

        Args:
            sample (dict): input status
            transformed (dict): output status
        """
        if not self.visualize:
            return

        self.tvis.apply_to_input(sample, None)
        self.tvis.apply_to_bboxes(None, transformed)
        self.tvis.update_custom_function(
            name='points',
            func=self.apply_to_points
        )
        self.tvis.apply_to_custom_function(
            sample=sample,
            transformed=transformed
        )
        self._visualized_images = self.tvis.visualized_images

    def __call__(self, sample) -> dict:
        """Custom transform flow.

        Args:
            sample (dict): input status

        Returns:
            dict: output status
        """
        # Mapping variables
        bboxes = sample['bboxes']
        transformed = sample

        # Transform
        points = [bbox[:2] for bbox in bboxes]
        transformed['points'] = points

        self.visualization_hook(sample, transformed)

        return transformed


@TRANSFORMS.register(name='CustomToImage')
class CustomToImage(BaseTransform):
    def __init__(self,
                 custom_max=255.,
                 input_key=None,
                 output_key=None,
                 visualize=False) -> None:
        """Mapping the value range to (0, custom_max).

        Args:
            custom_max (int or float): max value of pixel
            input_key (tuple[str]): input key
            output_key (tuple[str]): output key
            visualize (bool): visualize transform
        """
        super().__init__(input_key=input_key,
                         output_key=output_key,
                         visualize=visualize)
        self._max = custom_max
        self._input_key = [input_key] if isinstance(
            input_key, str) else input_key
        self._output_key = [output_key] if isinstance(
            output_key, str) else output_key

    def visualization_hook(self, sample, transformed):
        """Customized visualization layers.

        Args:
            sample (dict): input status
            transformed (dict): output status
        """
        if not self.visualize:
            return

        self._visualized_images = self.tvis.visualized_images

    def __call__(self, sample) -> dict:
        """Custom transform flow.

        Args:
            sample (dict): input status

        Returns:
            dict: output status
        """
        if self._input_key is None:
            self._input_key = list(sample.keys())
        if self._output_key is None:
            self._output_key = list(sample.keys())
        assert len(self._input_key) == len(self._output_key)

        transformed = sample

        for input_key, output_key in zip(self._input_key, self._output_key):
            target = sample[input_key]
            sample[output_key] = self._max \
                * (target - target.min()) \
                / (target.max() - target.min() + 1e-10)

        self.visualization_hook(sample, transformed)

        return transformed

    def __str__(self):
        return '{0} from {1} to {2}'.format(
            self.__class__.__name__,
            self._input_key,
            self._output_key
        )


@TRANSFORMS.register(name='CustomConcatInputs')
class CustomConcatInputs(BaseTransform):
    def __init__(self,
                 input_key=('image', 'extreme_points'),
                 output_key=('concat'),
                 visualize=False) -> None:
        """Concate multiple inputs

        Args:
            input_key (tuple[str]): input key
            output_key (tuple[str]): output key
            visualize (bool): visualize transform
        """
        super().__init__(input_key=input_key,
                         output_key=output_key,
                         visualize=visualize)
        self._input_key = [input_key] if isinstance(
            input_key, str) else input_key
        self._output_key = [output_key] if isinstance(
            output_key, str) else output_key

    def visualization_hook(self, sample, transformed):
        """Customized visualization layers.

        Args:
            sample (dict): input status
            transformed (dict): output status
        """
        if not self.visualize:
            return

        self._visualized_images = self.tvis.visualized_images

    def __call__(self, sample) -> dict:
        """Custom transform flow.

        Args:
            sample (dict): input status

        Returns:
            dict: output status
        """

        transformed = sample

        concat = []
        target_size = sample[self._input_key[0]].shape
        for input_key in self._input_key:
            assert sample[input_key].shape[:2] == target_size[:2], 'concat size not match'
            target = sample[input_key]
            if target.ndim == 2:
                target = target[..., np.newaxis]
            concat.append(target)

        transformed[self._output_key[0]] = np.concatenate(concat, axis=2)

        self.visualization_hook(sample, transformed)

        return transformed

    def __str__(self):
        return '{0} from {1} to {2}'.format(
            self.__class__.__name__,
            self._input_key,
            self._output_key
        )


@TRANSFORMS.register(name='CustomCollect')
class CustomCollect(BaseTransform):
    def __init__(self,
                 input_key=None,
                 output_key=None,
                 visualize=False) -> None:
        """Collect the needed elements.

        Args:
            input_key (tuple[str]): input key
            output_key (tuple[str]): output key
            visualize (bool): visualize transform
        """
        super().__init__(input_key=input_key,
                         visualize=visualize)
        assert input_key is not None
        self._input_key = [input_key] if isinstance(
            input_key, str) else input_key

    def apply_to_train_batch(self, sample, transformed):
        mask = transformed['mask'].copy()
        concat = sample['concat'].copy()
        draw_image = concat[..., :3]
        extreme_points = concat[..., -1]

        # Overlay mask - Require floating image
        draw_image = dextr_helper.overlay_mask(
            im=draw_image / 255,
            ma=mask,
            colors=None,
            alpha=0.5
        ) * 255

        # Draw extreme points (with random disturbance).
        index = extreme_points > 0.5 * 255  # half of pixel range
        draw_image[index] = [255, 0, 0]

        return draw_image

    def visualization_hook(self, sample, transformed):
        """Customized visualization layers.

        Args:
            sample (dict): input status
            transformed (dict): output status
        """
        if not self.visualize:
            return

        self.tvis.update_custom_function(
            name='train_batch',
            func=self.apply_to_train_batch
        )
        self.tvis.apply_to_custom_function(
            sample=sample,
            transformed=transformed
        )

        self._visualized_images = self.tvis.visualized_images

    def __call__(self, sample) -> dict:
        """Custom transform flow.

        Args:
            sample (dict): input status

        Returns:
            dict: output status
        """
        transformed = {}
        for key in self._input_key:
            transformed[key] = sample[key]

        self.visualization_hook(sample, transformed)

        return transformed

    def __repr__(self):
        return '{0} Pick {1}'.format(
            self.__class__.__name__,
            self._input_key
        )


@TRANSFORMS.register(name='CustomToTensor')
class CustomToTensor(BaseTransform):
    def __init__(self,
                 input_key=None,
                 output_key=None,
                 except_key=('meta'),
                 visualize=False) -> None:
        """Collect the needed elements.

        Args:
            input_key (tuple[str]): input key
            output_key (tuple[str]): output key
            except_key (tuple[str]): keys discarded in training
            visualize (bool): visualize transform
        """
        super().__init__(input_key=input_key,
                         output_key=output_key,
                         visualize=visualize)
        self._input_key = [input_key] if isinstance(
            input_key, str) else input_key
        self._output_key = [output_key] if isinstance(
            output_key, str) else output_key
        self._except_key = [except_key] if isinstance(
            except_key, str) else except_key

    def visualization_hook(self, sample, transformed):
        """Customized visualization layers.

        Args:
            sample (dict): input status
            transformed (dict): output status
        """
        if not self.visualize:
            return

        self._visualized_images = self.tvis.visualized_images

    def __call__(self, sample) -> dict:
        """Custom transform flow.

        Args:
            sample (dict): input status

        Returns:
            dict: output status
        """
        if self._input_key is None:
            self._input_key = list(sample.keys())
        if self._output_key is None:
            self._output_key = list(sample.keys())
        assert len(self._input_key) == len(self._output_key)

        transformed = sample

        for input_key, output_key in zip(self._input_key, self._output_key):
            if input_key in self._except_key:
                continue

            target = sample[input_key]
            if input_key == 'bbox':
                transformed[output_key] = torch.from_numpy(target)
                continue

            if target.ndim == 2:
                target = target[:, :, np.newaxis]

            # [Color Axis Order]
            # numpy image: H x W x C
            # torch image: C X H X W
            target = target.transpose((2, 0, 1))
            transformed[output_key] = torch.from_numpy(target)

        self.visualization_hook(sample, transformed)

        return transformed

    def __repr__(self):
        if self._input_key is None:
            return '{0} transfer all element to torch tensor except {1}'.format(
                self.__class__.__name__,
                self._except_key
            )
        else:
            return '{0} from {1} to {2} except {3}'.format(
                self.__class__.__name__,
                self._input_key,
                self._output_key,
                self._except_key
            )
