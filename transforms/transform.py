from collections import OrderedDict

from .visualization import TransformVisualization as TVis


class ITransform:
    def __init__(self,
                 input_key=('image', 'bboxes'),
                 output_key=('image', 'bboxes'),
                 visualize=False) -> None:
        raise NotImplementedError()

    @staticmethod
    def _get_input_keys(sample):
        raise NotImplementedError()

    def _transform_function(self, *args, **kwargs):
        raise NotImplementedError()

    def visualization_hook(self, sample, transformed):
        raise NotImplementedError()

    def __call__(self, sample) -> dict:
        raise NotImplementedError()

    def clear_visualized_cache(self):
        raise NotImplementedError()

    @property
    def visualized_images(self):
        raise NotImplementedError()


class BaseTransform(ITransform):
    """Basic functionailty of transform.
    """

    def __init__(self,
                 input_key=('image', 'bboxes'),
                 output_key=('image', 'bboxes'),
                 visualize=False) -> None:
        """Initialize necessary variables.

        Args:
            input_key (tuple, optional): input status. Defaults to ('image', 'bboxes').
            output_key (tuple, optional): output status. Defaults to ('image', 'bboxes').
            visualize (bool, optional): visualize transform. Defaults to False.
        """
        self._input_key = input_key
        self._output_key = output_key
        self._transform = self._transform_function

        self.visualize = visualize
        if self.visualize:
            self.tvis = TVis()
        self._visualized_images = None

    @staticmethod
    def _get_input_keys(sample):
        """Remap keys to pre-defined keys in the albumentations.

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
        Note that the function should be wrapped by albumentations compose.
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
