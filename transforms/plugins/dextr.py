from copy import deepcopy

import numpy as np
import torch

from ..builder import TRANSFORMS
from ..helpers import dextr_helper
from ..transform import BaseTransform


@TRANSFORMS.register(name='CustomCropFromMask')
class CustomCropFromMask(BaseTransform):
    def __init__(self,
                 relax: int = 50,
                 zero_pad: bool = True,
                 mask_key: str = 'mask',
                 input_key=('image', 'mask'),
                 output_key=('image', 'mask'),
                 visualize=False) -> None:
        """Custom transfrom of DEXTR - CropFromMask.

        Args:
            relax (int, optional): margin pixels to the border. Defaults to 50.
            zero_pad (bool, optional): zero padding mode. Defaults to True.
            mask_key (str, optional): extracted mask key. Defaults to 'mask'.
            input_key (tuple, optional): input status. Defaults to ('image', 'mask').
            output_key (tuple, optional): output status. Defaults to ('image', 'mask').
            visualize (bool, optional): visualize transform. Defaults to False.
        """
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

    def __repr__(self):
        return '{0}: relax={1} zero_pad={2}'.format(
            self.__class__.__name__,
            self._relax,
            self._zero_pad
        )


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

    def __repr__(self):
        return '{0} generate from {1} to {2}, pert={3}, sigma={4}'.format(
            self.__class__.__name__,
            self._input_key,
            self._output_key,
            self._pert,
            self._sigma
        )


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

    def __repr__(self):
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

    def __repr__(self):
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
