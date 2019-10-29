# this module implements an abstract base class

import random
import numpy as np 
import torch.utils.data as data
from PIL import Image
import torchvision.transforms as transforms

import torch
import torch.nn.functional as F 
import os

class BaseDataset(data.Dataset):
    def __init__(self, opt):
        super(BaseDataset, self).__init__()

    # implement the data processing method from yolov3 implementation
    def pad_to_square(self, img, pad_value):
        # this function makes an image into square shape
        c, h, w = img.shape
        dim_diff = np.abs(h-w)
        pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
        pad = (0,0, pad1, pad2) if h <= w else (pad1, pad2, 0,0)
        img = F.pad(img, pad, 'constant', value=pad_value)
        return img, pad

    def resize(self, image, size, mode='bilinear'):
        img_size = image.size()
        #print("img size is: ", len(img_size))
        if (len(img_size) == 3):
            image = image.unsqueeze(0)
        image = F.interpolate(image, size=size, mode=mode).squeeze(0)
        return image 

    def random_resize(self, images, min_size=288, max_size=488, mode='bilinear'):
        new_size = random.sample(list(range(min_size, max_size+1, 32)), 1)[0]
        print('new_size: ', new_size)
        images = F.interpolate(images, size=new_size, mode=mode)
        return images

    def get_paths(self, path, suffix='.jpg'):
        image_paths = []
        for dirpath, subdirs, files in os.walk(path):
            for file in files:
                if suffix in file:
                    image_paths.append('/'.join([dirpath, file]))
        return image_paths

    def replace(self, paths):
        label_path = []
        for path in paths:
            p = path[:-4]+'.txt'
            p = p.replace('image/', 'label/')
            label_path.append(p)
        return label_path

    def horisontal_flip(self, images, targets):
        images = torch.flip(images, [-1])
        targets[:,2] = 1 - targets[:, 2]
        return images, targets

    def name(self):
        return "BaseDataset"