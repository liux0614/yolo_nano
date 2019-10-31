import random
import numpy as np 
from PIL import Image
import torchvision.transforms as transforms

import torch
import os

from data.base_dataset import BaseDataset


class ListDataset(BaseDataset):
    def __init__(self, folder_path, img_size=416, augment=True, multiscale=True, normalized_labels=True):
        # read all the images paths
        self.img_paths = self.get_paths(folder_path, '.jpg')
        self.label_paths = self.replace(self.img_paths)
        self.img_size = img_size
        self.max_objects = 100
        self.augment = False #augment
        self.multiscale = False #multiscale
        self.normalized_labels = normalized_labels
        self.min_size = self.img_size - 3 * 32
        self.max_size = self.img_size + 3 * 32

    def __getitem__(self, index):
        # image
        img_path = self.img_paths[index]
        if not os.path.exists(img_path):
            print('cannot find the image_path')
        img = transforms.ToTensor()(Image.open(img_path).convert('RGB'))

        # handle gray scale channels
        if len(img.shape) != 3:
            img = img.unsqueeze(0)
            img = img.expand((3, img.shape[1:]))

        _, h, w = img.shape
        h_factor, w_factor = (h,w) if self.normalized_labels else (1,1)
        img, pad = self.pad_to_square(img, 0)
        _, padded_h, padded_w = img.shape

        # label
        label_path = self.label_paths[index]
        targets = None
        if os.path.exists(label_path):
            boxes = torch.from_numpy(np.loadtxt(label_path).reshape(-1,5))
            # extract coordinates
            x1 = w_factor * (boxes[:,1] - boxes[:,3] / 2)
            x2 = w_factor * (boxes[:,1] + boxes[:,3] / 2)
            y1 = h_factor * (boxes[:,2] - boxes[:,4] / 2)
            y2 = h_factor * (boxes[:,2] + boxes[:,4] / 2)
            #Adjust for added pading
            x1 += pad[0]
            x2 += pad[1]
            y1 += pad[2]
            y2 += pad[3]

            # boxes with the format of (x,y,w,h)
            boxes[:, 1] = ((x1 + x2) / 2) / padded_w
            boxes[:, 2] = ((y1 + y2) / 2) / padded_h
            boxes[:, 3] *= w_factor / padded_w
            boxes[:, 4] *= h_factor / padded_h

            targets = torch.zeros((len(boxes), 6))
            targets[:, 1:] = boxes
        else:
            print(f'cannot find the labels txt path: {label_path}')

        if self.augment:
            if np.random.random() < 0.5:
                img, targets = self.horisontal_flip(img, targets)

        return img, targets

    def __len__(self):
        return len(self.img_paths)

    def name(self):
        return "ListDataset"

    def collate_fn(self, batch):
        imgs, targets = list(zip(*batch))
        targets = [boxes for boxes in targets if boxes is not None]
        for i, boxes in enumerate(targets):
            boxes[:, 0] = i
        targets = torch.cat(targets, 0)
        if self.multiscale:
            self.img_size = random.choice(range(self.min_size, self.max_size + 1, 32))
        # Resize images to input shape
        imgs = torch.stack([self.resize(img, self.img_size) for img in imgs])
        return imgs, targets