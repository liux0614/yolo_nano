import torch
from .transforms import Compose, RandomHorizontalFlip, ToTensor

def get_transforms(opt):
    trasforms = []
    if not opt.no_hflip:
        trasforms.append(RandomHorizontalFlip(opt.hflip_prob))

    trasforms.append(ToTensor())

    return Compose(trasforms)