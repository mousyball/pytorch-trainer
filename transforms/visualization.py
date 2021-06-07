from enum import Enum
from collections import OrderedDict

import cv2

from .helpers import dextr_helper


class MaskVisual(Enum):
    STRAIGHT_MIX = 0
    DEXTR = 1


class ITransformVisualization:
    def __init__(self) -> None:
        pass

    @property
    def visualized_images(self):
        raise NotImplementedError()

    def apply_to_input(self, sample, transformed):
        raise NotImplementedError()

    def apply_to_image(self, sample, transformed):
        raise NotImplementedError()

    def apply_to_bboxes(self, sample, transformed):
        raise NotImplementedError()

    def apply_to_mask(self, sample, transformed):
        raise NotImplementedError()

    def apply_to_masks(self, sample, transformed):
        raise NotImplementedError()

    def apply_to_keypoints(self, sample, transformed):
        raise NotImplementedError()


class BaseTransformVisualization(ITransformVisualization):
    def __init__(self) -> None:
        super().__init__()
        self._visualized_images = OrderedDict()
        self._mask_display_method = MaskVisual.DEXTR

    @property
    def visualized_images(self):
        return self._visualized_images

    def apply_to_input(self, sample, transformed):
        self._visualized_images['input'] = sample['image']

    def apply_to_image(self, sample, transformed):
        self._visualized_images['image'] = transformed['image']

    def apply_to_bboxes(self, sample, transformed):
        draw_image = transformed['image'].copy()
        for bbox in transformed['bboxes']:
            bbox_int = [int(pt) for pt in bbox[:4]]
            pt0 = tuple(bbox_int[:2])
            pt1 = tuple(bbox_int[2:])
            cv2.rectangle(draw_image, pt0, pt1, (0, 255, 0))

        self._visualized_images['bboxes'] = draw_image

    def apply_to_mask(self, sample, transformed):
        draw_image = transformed['image'].copy()
        mask = transformed['mask'].copy()

        if self._mask_display_method == MaskVisual.STRAIGHT_MIX:
            draw_image = draw_image[:, :, 0] * 0.5 + (mask * 255) * 0.5
        elif self._mask_display_method == MaskVisual.DEXTR:
            # Require floating image
            draw_image = dextr_helper.overlay_mask(
                im=draw_image / 255,
                ma=mask / 255,
                colors=None,
                alpha=0.5
            ) * 255

        self._visualized_images['mask'] = draw_image

    def apply_to_masks(self, sample, transformed):
        pass

    def apply_to_keypoints(self, sample, transformed):
        pass


class TransformVisualization(BaseTransformVisualization):
    def __init__(self) -> None:
        super().__init__()
        self._custom_table = dict()

    def apply_to_custom_function(self, sample, transformed):
        """Apply the visualized functions in the table.

        Args:
            sample (dict): input status
            transformed (dict): output status
        """
        for name, func in self._custom_table.items():
            self._visualized_images[name] = func(sample, transformed)

    def update_custom_function(self, name, func):
        """Update custom visualized function into a table.

        Args:
            name (str): key name
            func (object): visualized function
        """
        self._custom_table[name] = func
