from collections import OrderedDict


def get_input_keys(sample):
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
