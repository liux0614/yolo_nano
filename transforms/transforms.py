import random
import math
import numbers
import torch
from PIL import Image

from torchvision.transforms import functional as F
from . import bounding_box 

class Compose(object):
    """Composes several transforms together.

    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.

    Example:
        >>> transforms.Compose([
        >>>     transforms.Pad(),
        >>>     transforms.ToTensor(),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, bboxes):
        for t in self.transforms:
            img, bboxes = t(img, bboxes)
        return img, bboxes

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string


class RandomHorizontalFlip(object):
    """horizontally flip the image and bounding boxes
    """
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image, bboxes):
        if random.random() < self.p:
            return F.hflip(image), bboxes.hflip()
        else:
            return image, bboxes

class ToTensor(object):
    def __init__(self):
        pass
    
    def __call__(self, image, bboxes):
        return F.to_tensor(image), bboxes.to_tensor()