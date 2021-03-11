from typing import List

import cv2
import albumentations as A

from .utils import get_input_keys
from .builder import TRANSFORMS
from .visualization import TransformVisualization as TVis


class BaseTransform:
    def __init__(self,
                 input_key=('image', 'bboxes'),
                 output_key=('image', 'bboxes'),
                 visualize=False) -> None:
        self._input_key = input_key
        self._output_key = output_key
        self._transfrom = self._transform_function

        self.visualize = visualize
        if self.visualize:
            self.tvis = TVis()
            self._visualized_images = None

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
        input_dict = get_input_keys(sample)

        # Transform image by the input keys.
        transformed = self._transfrom(**input_dict)

        # Visualize transformed result if needed.
        self.visualization_hook(sample, transformed)

        # Same keys as the sample.
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
                 bbox_format='pascal_voc',
                 input_key=('image', 'bboxes'),
                 output_key=('image', 'bboxes'),
                 visualize=False) -> None:
        super().__init__(input_key=input_key,
                         output_key=output_key,
                         visualize=visualize)
        self._degree = degree
        self._prob = prob
        self._bbox_format = bbox_format

        self._transfrom = self._transform_function()

    def _transform_function(self):
        """Albumentation tranform.

        Returns:
            :obj:`A.Compose`: A wrapped composed function.
        """
        transform = A.Compose(
            [A.Rotate(limit=self._degree, p=self._prob)],
            bbox_params=A.BboxParams(format=self._bbox_format,
                                     label_fields=[])
        )

        return transform


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
