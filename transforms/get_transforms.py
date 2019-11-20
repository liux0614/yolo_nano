import torch
from .transforms import (
    Compose, PadToSquare, RandomCrop, RandomHorizontalFlip, ToTensor)

def get_train_transforms(opt):
    trasforms = []
    if not opt.no_pad2square:
        trasforms.append(PadToSquare())
    # if not opt.no_crop:
    #     trasforms.append(RandomCrop(opt.crop_size, pad_if_needed=True))
    if not opt.no_hflip:
        trasforms.append(RandomHorizontalFlip(opt.hflip_prob))

    trasforms.append(ToTensor())

    return Compose(trasforms)


def get_val_transforms(opt):
    trasforms = []
    if not opt.no_pad2square:
        trasforms.append(PadToSquare())
    trasforms.append(ToTensor())

    return Compose(trasforms)


def get_test_transforms(opt):
    trasforms = []

    trasforms.append(ToTensor())

    return Compose(trasforms)