import cv2

from ..builder import TRANSFORMS
from ..transform import BaseTransform


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
