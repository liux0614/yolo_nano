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


class Resize(object):
    """Resize image to a desired size
    """
    def __init__(self, size, interpolation=Image.BILINEAR):
        assert isinstance(size, int) or (isinstance(size, Iterable) and len(size) == 2)
        self.size = size
        self.interpolation = interpolation

    def __call__(self, image, bboxes):
        return F.resize(image, self.size, self.interpolation), bboxes


class PadToSquare(object):
    """Pad image to square shape and adjust the bounding box coordinates
    """
    def __init__(self, fill=0, padding_mode='constant'):
        assert isinstance(fill, (numbers.Number, str, tuple))
        assert padding_mode in ['constant', 'edge', 'reflect', 'symmetric']
        self.fill = fill
        self.padding_mode = padding_mode

    def __call__(self, image, bboxes):
        w, h = image.size
        if w == h:
            return image, bboxes
        
        dim_diff = abs(w - h)
        padding_1, padding_2 = dim_diff // 2, dim_diff - dim_diff // 2
        padding = (0, padding_1, 0, padding_2) if w > h else (padding_1, 0, padding_2, 0)
        image = F.pad(image, padding, fill=self.fill, padding_mode=self.padding_mode)
        bboxes = bboxes.pad(padding)
        return image, bboxes


class RandomCrop(object):
    def __init__(
        self, size, padding=None, pad_if_needed=False,
        fill=0, padding_mode='constant'):

        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.padding = padding
        self.pad_if_needed = pad_if_needed
        self.fill = fill
        self.padding_mode = padding_mode

    @staticmethod
    def get_params(image, output_size):
        w, h = image.size
        th, tw = output_size
        if w == tw and h == th:
            return 0, 0, h, w

        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)
        return i, j, th, tw

    def __call__(self, image, bboxes):
        if self.padding is not None:
            image = F.pad(image, self.padding, self.fill, self.padding_mode)
            bboxes = bboxes.pad(padding)

        # pad the width if needed
        if self.pad_if_needed and image.size[0] < self.size[1]:
            image = F.pad(image, (self.size[1] - image.size[0], 0), self.fill, self.padding_mode)
            bboxes = bboxes.pad((self.size[1] - image.size[0], 0))
        # pad the height if needed
        if self.pad_if_needed and image.size[1] < self.size[0]:
            image = F.pad(image, (0, self.size[0] - image.size[1]), self.fill, self.padding_mode)
            bboxes = bboxes.pad((0, self.size[0] - image.size[1]))
        
        i, j, h, w = self.get_params(image, self.size)
        return F.crop(image, i, j, h, w), bboxes.crop((i, j, i+h, j+w))


class ToTensor(object):
    def __init__(self):
        pass
    
    def __call__(self, image, bboxes):
        return F.to_tensor(image), bboxes.to_tensor()