import numbers
from PIL import Image
import torch
from .bounding_box import BBox

def _is_pil_image(img):
    return isinstance(img, Image.Image)

def _is_bbox(bboxes):
    return isinstance(bboxes, BBox)


def pad(obj, padding):
    if not _is_bbox(obj):
        raise TypeError('img should be PIL Image or BBox. Got {}'.format(type(img)))

    if not isinstance(padding, (numbers.Number, tuple)):
        raise TypeError('Got inappropriate padding arg')
    if not isinstance(fill, (numbers.Number, str, tuple)):
        raise TypeError('Got inappropriate fill arg')

    return BBox.pad(padding)





