import numbers
import numpy as np
import collections
import torch

FLIP_LEFT_RIGHT = 0
FLIP_TOP_BOTTOM = 1

def _validate_bboxes(bboxes):
    if isinstance(bboxes, tuple) or isinstance(bboxes, np.ndarray):
        bboxes = torch.tensor(bboxes).float()
    elif isinstance(bboxes, torch.Tensor):
        bboxes = bboxes.clone().detach().float()
    else:
        raise ValueError("Type of bboxes should be `tuple`, `numpy.ndarray` or `torch.Tensor`. Got {}".format(type(bboxes)))

    if bboxes.ndimension() != 2:
        raise ValueError("Dimensions of bbox should be 2. Got {}".format(bboxes.ndimension()))
    # if bboxes.size(0) == 0:
    #     raise ValueError("There should be at least one bounding box. Got {}".format(bboxes.size(0)))
    if bboxes.size(-1) != 5:
        raise ValueError("Last dimenion of bboxes should be 5 (including classes). Got {}".format(bboxes.size(-1)))

    return bboxes

def _validate_size(size):
    if isinstance(size, numbers.Number):
        size = (int(size), int(size))

    return size

class BBox(object):
    def __init__(self, bboxes, image_size):

        self.size = image_size
        self.bboxes = bboxes

    @property 
    def bbox_sizes(self):
        return self.bboxes.size()


    @staticmethod
    def from_xyxy(bboxes, image_size):
        bboxes = _validate_bboxes(bboxes)
        image_size = _validate_size(image_size)
        
        return BBox(bboxes, image_size)

    @staticmethod
    def from_xyhw(bboxes, image_size, normalized=False):
        # x y is the center, not the lower-bottom corner
        bboxes = _validate_bboxes(bboxes)
        image_size = _validate_size(image_size)
        
        w_factor, h_factor = image_size if normalized else (1, 1)
        classes, tx, ty, w, h = bboxes.split(1, dim=-1)
        xmin = w_factor * (tx - w/2)
        ymin = h_factor * (ty - h/2)
        xmax = w_factor * (tx + w/2)
        ymax = h_factor * (ty + h/2)
        
        return BBox(torch.cat((classes, xmin, ymin, xmax, ymax), dim=-1), image_size)

    @staticmethod
    def from_yolo(bboxes, image_size):
        return BBox.from_xyhw(bboxes, image_size, normalized=True)

    @staticmethod
    def from_coco(bboxes, image_size):
        bboxes = _validate_bboxes(bboxes)
        image_size = _validate_size(image_size)
        
        classes, xmin, ymin, w, h = bboxes.split(1, dim=-1)
        xmax = xmin + w
        ymax = ymin + h
        
        return BBox(torch.cat((classes, xmin, ymin, xmax, ymax), dim=-1), image_size)

    def _split(self, mode='xyxy'):
        if mode == 'xyxy':
            return self.bboxes.split(1, dim=-1)
        elif mode == 'xyhw':
            classes, xmin, ymin, xmax, ymax = bboxes.split(1, dim=-1)
            tx = xmin + xmax
            ty = ymin + ymax
            w = xmax - xmin
            h = ymax - ymin
            return classes, tx, ty, w, h

    def to_tensor(self, mode='yolo'):
        if mode not in ('yolo', 'xyhw', 'xyxy', 'coco'):
            raise ValueError("BBox only supports mode: `yolo`, `xyhw`, `xyxy`, `coco`. Got {}".format(mode))

        if mode == 'xyxy':
            return self.bboxes
        elif mode == 'coco':
            classes, xmin, ymin, xmax, ymax = self._split()
            return torch.cat((classes, xmin, ymin, xmax-xmin, ymax-ymin), -1)
        else:
            w_factor, h_factor = self.size if mode == 'yolo' else (1, 1)
            classes, xmin, ymin, xmax, ymax = self._split()
            tx = (xmin + xmax) / (2*w_factor)
            ty = (ymin + ymax) / (2*h_factor)
            w = (xmax - xmin) / w_factor
            h = (ymax - ymin) / h_factor
        
            return torch.cat((classes, tx, ty, w, h), -1)

    def to_numpy(self, mode='yolo'):
        if mode not in ('yolo', 'xyhw', 'xyxy', 'coco'):
            raise ValueError("BBox only supports mode: `yolo`, `xyhw`, `xyxy`, `coco`. Got {}".format(mode))

        if mode == 'xyxy':
            return self.bboxes.numpy()


    def crop(self, box):
        """Crop the bboxes.
        Args:
            box: left, top, left+width, top+height
        """
        w, h = box[2] - box[0], box[3] - box[1]
        classes, xmin, ymin, xmax, ymax = self._split()
        cropped_xmin = (xmin - box[0]).clamp(min=0, max=w)
        cropped_ymin = (ymin - box[1]).clamp(min=0, max=h)
        cropped_xmax = (xmax - box[0]).clamp(min=0, max=w)
        cropped_ymax = (ymax - box[1]).clamp(min=0, max=h)

        cropped_bboxes = torch.cat(
            (classes, cropped_xmin, cropped_ymin, cropped_xmax, cropped_ymax), dim=-1)
        
        is_empty = (cropped_xmin == cropped_xmax) | (cropped_ymin == cropped_ymax)
        is_empty = is_empty.view(-1)
        cropped_bboxes = cropped_bboxes[is_empty == 0]

        return BBox(cropped_bboxes, (w, h))
    
    def resize(self, box_size):
        w, h = self.size
        box_w, box_h = _validate_size(box_size)
        
        classes, xmin, ymin, xmax, ymax = self._split()
        resized_xmin = (xmin - box_w).clamp(min=0, max=w)
        resized_ymin = (ymin - box_h).clamp(min=0, max=h)
        resized_xmax = (xmax - box_w).clamp(min=0, max=w)
        resized_ymax = (ymax - box_h).clamp(min=0, max=h)

        resized_bboxes = torch.cat(
            (classes, resized_xmin, resized_ymin, resized_xmax, resized_ymax), dim=-1)

        return BBox(resized_bboxes, self.size)

    
    def pad(self, padding):
        # If len(padding) == 2: padding on left/right and top/bottom respectively
        # If len(padding) == 4: padding on left, top, right and bottom respectively
        if isinstance(padding, numbers.Number):
            left, top, right, bottom = padding, padding, padding, padding
        elif isinstance(padding, tuple) and len(padding) == 2:
            left, top, right, bottom = padding[0], padding[1], padding[0], padding[1]
        elif isinstance(padding, tuple) and len(padding) == 4:
            left, top, right, bottom = padding
        else:
            raise ValueError("Padding must be an int or a 2, or 4 element tuple, not a " +
                             "{} element tuple".format(len(padding)))
        
        classes, xmin, ymin, xmax, ymax = self._split()
        padded_xmin = xmin + left
        padded_ymin = ymin + top
        padded_xmax = xmax + left
        padded_ymax = ymax + top

        padded_bboxes = torch.cat((classes, padded_xmin, padded_ymin, padded_xmax, padded_ymax), -1)
        w, h = self.size
        padded_w = w + left + right
        padded_h = h + top + bottom

        return BBox(padded_bboxes, (padded_w, padded_h))

    def rotate(self, angle):
        w, h = self.size
        classes, xmin, ymin, xmax, ymax = self._split()
        tx = w / 2 # center of image
        ty = h / 2
        
        rotated_xmin = torch.cos(angle) * (xmin-tx) - torch.sin(angle) * (ymin - ty) + tx
        rotated_ymin = torch.sin(angle) * (xmin-tx) + torch.cos(angle) * (ymin - ty) + ty
        rotated_xmax = torch.cos(angle) * (xmax-tx) - torch.sin(angle) * (ymax - ty) + tx
        rotated_ymax = torch.sin(angle) * (xmax-tx) + torch.cos(angle) * (ymax - ty) + ty

        rotated_bboxes = torch.cat(
            (classes, rotated_xmin, rotated_ymin, rotated_xmax, rotated_ymax), -1)

    def hflip(self):
        w, h = self.size
        classes, xmin, ymin, xmax, ymax = self._split()
        transposed_xmin = w - xmax
        transposed_xmax = w - xmin

        transposed_bboxes = torch.cat(
                (classes, transposed_xmin, ymin, transposed_xmax, ymax), dim=-1)

        return BBox(transposed_bboxes, (w, h))

    def vflip(self):
        w, h = self.size
        classes, xmin, ymin, xmax, ymax = self._split()
        transposed_ymin = h - ymax
        transposed_ymax = h - ymin

        transposed_bboxes = torch.cat(
                (classes, xmin, transposed_ymin, xmax, transposed_ymax), dim=-1)
            
        return BBox(transposed_bboxes, (w, h))

    def transpose(self, method):
        if method not in (FLIP_LEFT_RIGHT, FLIP_TOP_BOTTOM):
            raise NotImplementedError("Only horizontal and vertical flipping are supported. Got {}".format(method))
        
        if method == FLIP_LEFT_RIGHT:
            return self.hflip()
        else:
            return self.vflip()

