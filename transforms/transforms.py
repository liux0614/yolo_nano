import random
import math
import numbers
import numpy as np
import torch
from PIL import Image

from torchvision.transforms import functional as F
from . import functional as T
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

    def randomize_parameters(self):
        pass

class MultiScale(object):
    """resize the image
    """
    def __init__(self, size, interpolation=Image.BILINEAR):
        self.size = size
        self.min_size = size - 3*32
        self.max_size = size + 3*32
        self.interpolation = interpolation


    def __call__(self, image, bboxes):
        size = random.choice(range(self.min_size, self.max_size + 1, 32))
        resized_image = F.resize(image, (size, size), self.interpolation)
        return resized_image, bboxes
    
    def randomize_parameters(self):
        pass

class ToTensor(object):
    def __init__(self):
        pass
    
    def __call__(self, image, bboxes):
        return F.to_tensor(image), bboxes.to_tensor()
    
    def randomize_parameters(self):
        pass